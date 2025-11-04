import pandas as pd
import os
import re
import logging
import base64
import joblib
import html
from typing import List, Tuple, Optional
from email.header import decode_header

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ---------- Config ----------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_FILE = "credentials.json"  # Path to OAuth client JSON
TOKEN_FILE = "token.json"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
VEC_PATH = os.environ.get("VEC_PATH", "models/vectorizer.joblib")
PREDICTIONS_CSV = "predictions.csv"
SPAM_PROB_THRESHOLD = float(os.environ.get("SPAM_PROB_THRESHOLD", 0.5))
QUARANTINE_SAVE = True  # saves .eml copies of moved messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------- Preprocess (same as trainer) ----------
def simple_preprocess(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'(^>.*$)', ' ', s, flags=re.MULTILINE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def strip_html_and_unescape(raw_html: str) -> str:
    if not raw_html:
        return ""
    if "<html" not in raw_html.lower() and "<body" not in raw_html.lower():
        return raw_html
    raw = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', ' ', raw_html)
    raw = re.sub(r'(?i)<br\s*/?>', '\n', raw)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = html.unescape(raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def decode_mime_words(s: Optional[str]) -> str:
    if not s:
        return ""
    parts = decode_header(s)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            try:
                out.append(part.decode(enc or "utf-8", errors="ignore"))
            except Exception:
                out.append(part.decode("utf-8", errors="ignore"))
        else:
            out.append(part)
    return "".join(out)


# ---------- Gmail API helpers ----------
def get_gmail_service(credentials_file: str = CREDENTIALS_FILE, token_file: str = TOKEN_FILE):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"{credentials_file} not found. Create OAuth credentials and save to this file.")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_file, "w") as f:
                f.write(creds.to_json())
    service = build("gmail", "v1", credentials=creds)
    return service


def extract_text_from_payload(payload) -> str:
    if not payload:
        return ""

    def decode_part(p):
        body = p.get("body", {})
        data = body.get("data")
        if not data:
            return ""
        try:
            decoded = base64.urlsafe_b64decode(data.encode("ASCII"))
            try:
                return decoded.decode("utf-8", errors="ignore")
            except Exception:
                return decoded.decode("latin1", errors="ignore")
        except Exception:
            return ""

    if "parts" not in payload:
        return decode_part(payload) or ""
    # prefer text/plain, fallback to text/html
    texts = []
    for part in payload.get("parts", []):
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            txt = decode_part(part)
            if txt:
                texts.append(txt)
    if texts:
        return "\n".join(texts)
    for part in payload.get("parts", []):
        mime = part.get("mimeType", "")
        if mime == "text/html":
            txt = decode_part(part)
            if txt:
                return strip_html_and_unescape(txt)
    # nested fallback
    for part in payload.get("parts", []):
        if "parts" in part:
            txt = extract_text_from_payload(part)
            if txt:
                return txt
    return ""


def message_to_text(msg_full) -> Tuple[str, str]:
    payload = msg_full.get("payload", {})
    headers = payload.get("headers", [])
    subj = ""
    for h in headers:
        if h.get("name", "").lower() == "subject":
            subj = decode_mime_words(h.get("value"))
            break
    raw_body = extract_text_from_payload(payload)
    clean_body = strip_html_and_unescape(raw_body)
    combined = (subj + " " + clean_body).strip()
    combined = re.sub(r'\s+', ' ', combined)
    return subj, combined


# ---------- Model helpers ----------
def load_model_and_vectorizer(model_path: str = MODEL_PATH, vec_path: str = VEC_PATH):
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Model or vectorizer not found. Run trainer first.")
    logging.info("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# ---------- Actions ----------
def fetch_unread_inbox_message_ids(service) -> List[str]:
    resp = service.users().messages().list(userId="me", q="in:inbox is:unread").execute()
    msgs = resp.get("messages", []) or []
    ids = [m["id"] for m in msgs]
    logging.info("Found %d unread inbox messages", len(ids))
    return ids


def get_full_message(service, msg_id: str):
    return service.users().messages().get(userId="me", id=msg_id, format="full").execute()


def add_labels_move_to_spam(service, msg_id: str):
    mods = {"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX", "UNREAD"]}
    try:
        service.users().messages().modify(userId="me", id=msg_id, body=mods).execute()
        logging.info("Moved message %s to SPAM", msg_id)
    except Exception as e:
        logging.error("Failed to modify labels for %s: %s", msg_id, e)


def save_eml_quarantine(service, msg_id: str, dest_dir: str = "gmail_debug"):
    try:
        resp = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
    except Exception as e:
        logging.error("Failed to fetch raw message for %s: %s", msg_id, e)
        return
    raw_b64 = resp.get("raw") if isinstance(resp, dict) else None
    if not raw_b64:
        logging.warning("No raw data for %s, skipping quarantine save.", msg_id)
        return
    try:
        decoded = base64.urlsafe_b64decode(raw_b64.encode("ASCII"))
        os.makedirs(dest_dir, exist_ok=True)
        with open(os.path.join(dest_dir, f"{msg_id}.eml"), "wb") as f:
            f.write(decoded)
        logging.info("Saved quarantine copy for %s", msg_id)
    except Exception as e:
        logging.error("Failed to save quarantine for %s: %s", msg_id, e)


# ---------- Processing ----------
def process_messages(service, model, vectorizer, message_ids: List[str]):
    rows = []
    for mid in message_ids:
        full = get_full_message(service, mid)
        if full is None:
            continue
        subj, combined = message_to_text(full)
        if not combined.strip():
            logging.info("Skipping empty message %s", mid)
            rows.append({"message_id": mid, "subject": subj, "spam_prob": 0.0, "pred": 0})
            continue
        text_proc = simple_preprocess(combined)
        X = vectorizer.transform([text_proc])
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            # fallback: if model lacks predict_proba, default 0.0
            prob = 0.0
        pred = int(prob >= SPAM_PROB_THRESHOLD)
        logging.info("Msg %s | subj=%s | prob=%.3f pred=%d", mid, subj[:80], prob, pred)
        rows.append({"message_id": mid, "subject": subj, "spam_prob": prob, "pred": pred})
        if pred == 1:
            if QUARANTINE_SAVE:
                save_eml_quarantine(service, mid)
            add_labels_move_to_spam(service, mid)
    # append predictions
    df = pd.DataFrame(rows)
    if not df.empty:
        if os.path.exists(PREDICTIONS_CSV):
            df.to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)
        else:
            df.to_csv(PREDICTIONS_CSV, index=False)
        logging.info("Saved predictions to %s", PREDICTIONS_CSV)
        CSV = pd.read_csv(PREDICTIONS_CSV)
        print("Number of Spam: ", int(CSV["pred"].sum()))
        print("Number of Preprocessed Messages: ", int(CSV["message_id"].count()))


# ---------- Main ----------
def main():
    model, vectorizer = load_model_and_vectorizer()
    service = get_gmail_service()
    msg_ids = fetch_unread_inbox_message_ids(service)
    if not msg_ids:
        logging.info("No unread messages.")
        return
    process_messages(service, model, vectorizer, msg_ids)
    logging.info("Run complete.")


if __name__ == "__main__":
    main()
