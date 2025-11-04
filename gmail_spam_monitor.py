#!/usr/bin/env python3
"""
gmail_spam_monitor.py - Robust version with bug fixes.

Primary changes from previous:
 - Fixed `raw` local-variable bug when saving .eml quarantine copies.
 - Defensive checks for Gmail API responses (explicit None checks).
 - Better exception logging to avoid referencing unassigned locals.
 - Keeps features: OAuth2/Gmail API, model/vectorizer loading, spam prediction,
   move to SPAM label, quarantine .eml save, predictions CSV, debug token dump.

Requirements:
 pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib joblib scikit-learn pandas
"""

import os
import re
import logging
import base64
import joblib
import html
import datetime
from typing import List, Optional, Tuple
import pandas as pd
import traceback

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.header import decode_header
from google.auth.transport.requests import Request


# ---------- Config ----------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_FILE = os.environ.get("GMAIL_OAUTH_CREDENTIALS", "credentials.json")
TOKEN_FILE = os.environ.get("GMAIL_OAUTH_TOKEN", "token.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
VEC_PATH = os.environ.get("VEC_PATH", "models/vectorizer.joblib")
PREDICTIONS_CSV = os.environ.get("PRED_CSV", "predictions.csv")
DEBUG_DIR = os.environ.get("DEBUG_DIR", "gmail_debug")
SPAM_PROB_THRESHOLD = float(os.environ.get("SPAM_PROB_THRESHOLD", 0.5))
QUARANTINE_SAVE = os.environ.get("QUARANTINE_SAVE", "1") in ("1", "true", "True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------- Helpers ----------
def simple_preprocess(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r'(^>.*$)', ' ', s, flags=re.MULTILINE)
    s = re.sub(r'\b(from|sent|to|subject):.*', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def strip_html_and_unescape(raw_html: str) -> str:
    if not raw_html:
        return ""
    if "<html" not in raw_html.lower() and "<body" not in raw_html.lower():
        return raw_html
    raw = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', ' ', raw_html)
    raw = re.sub(r'(?i)<br\s*/?>', '\n', raw)
    raw = re.sub(r'(?i)</(p|div|li|tr)>', '\n', raw)
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


# ---------- Gmail API / OAuth ----------
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
            logging.info("Saved OAuth token to %s (do NOT commit).", token_file)
    service = build("gmail", "v1", credentials=creds)
    return service


# ---------- Model loading ----------
def load_model_and_vectorizer(model_path: str = MODEL_PATH, vec_path: str = VEC_PATH):
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Model or vectorizer not found. Expected {model_path} and {vec_path}")
    logging.info("Loading model and vectorizer from %s and %s ...", model_path, vec_path)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# ---------- Gmail message extraction ----------
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


# ---------- Gmail actions ----------
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
        logging.info("Message %s: added SPAM label and removed INBOX/UNREAD", msg_id)
    except Exception as e:
        logging.error("Failed to modify labels for %s: %s", msg_id, e)


def save_eml_quarantine(service, msg_id: str, dest_dir: str = DEBUG_DIR):
    """
    Robustly save raw RFC822 (.eml) content. Avoid referencing 'raw' unless it exists.
    """
    try:
        resp = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()
    except Exception as e:
        logging.error("Failed to fetch raw message for %s: %s", msg_id, e)
        return
    # resp may be None or not contain 'raw'
    if not resp or not isinstance(resp, dict):
        logging.warning("No response or unexpected format when fetching raw for %s", msg_id)
        return
    raw_b64 = resp.get("raw")
    if not raw_b64:
        logging.warning("No 'raw' field present in Gmail API response for %s; skipping quarantine save.", msg_id)
        return
    try:
        decoded = base64.urlsafe_b64decode(raw_b64.encode("ASCII"))
    except Exception as e:
        logging.error("Failed to decode raw base64 for %s: %s", msg_id, e)
        return
    try:
        os.makedirs(dest_dir, exist_ok=True)
        fn = os.path.join(dest_dir, f"{msg_id}.eml")
        with open(fn, "wb") as f:
            f.write(decoded)
        logging.info("Saved quarantine copy to %s", fn)
    except Exception as e:
        logging.error("Failed to write quarantine file for %s: %s", msg_id, e)


# ---------- Processing ----------
def process_messages(service, model, vectorizer, message_ids: List[str], save_csv: Optional[str] = PREDICTIONS_CSV):
    rows = []
    for mid in message_ids:
        try:
            full = get_full_message(service, mid)
            if full is None:
                logging.warning("Message %s returned None for full fetch — skipping.", mid)
                continue
            subj, combined = message_to_text(full)
            if not combined.strip():
                logging.info("Message %s appears empty after extraction; skipping.", mid)
                rows.append({"message_id": mid, "subject": subj, "spam_prob": 0.0, "pred": 0})
                continue
            text_proc = simple_preprocess(combined)
            X = vectorizer.transform([text_proc])
            try:
                prob = float(model.predict_proba(X)[:, 1][0])
            except Exception:
                # fallback to decision_function (sigmoid)
                try:
                    score = model.decision_function(X)
                    import math
                    prob = 1.0 / (1.0 + math.exp(-float(score)))
                except Exception:
                    logging.exception("Model doesn't support predict_proba or decision_function; defaulting prob=0.0")
                    prob = 0.0
            pred = int(prob >= SPAM_PROB_THRESHOLD)
            logging.info("Msg %s | subj=%s | prob=%.4f pred=%d", mid, subj[:80], prob, pred)
            rows.append({"message_id": mid, "subject": subj, "spam_prob": prob, "pred": pred})
            if pred == 1:
                if QUARANTINE_SAVE:
                    save_eml_quarantine(service, mid)
                add_labels_move_to_spam(service, mid)
        except Exception as e:
            logging.error("Failed processing message %s: %s\n%s", mid, e, traceback.format_exc())
            # continue to next message
    df = pd.DataFrame(rows)
    if not df.empty and save_csv:
        try:
            if os.path.exists(save_csv):
                df.to_csv(save_csv, mode="a", header=False, index=False)
            else:
                df.to_csv(save_csv, index=False)
            logging.info("Saved predictions to %s", save_csv)
        except Exception as e:
            logging.error("Failed to save predictions CSV: %s", e)
    # debug when nothing flagged
    if not df.empty and df['pred'].sum() == 0:
        logging.warning("All predictions are 0 (no spam detected). Writing debug artifacts.")
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            df.to_csv(os.path.join(DEBUG_DIR, f"predictions_debug_{stamp}.csv"), index=False)
            try:
                coef = None
                if hasattr(model, "coef_"):
                    coef = model.coef_
                elif hasattr(model, "named_steps"):
                    final = list(model.named_steps.items())[-1][1]
                    if hasattr(final, "coef_"):
                        coef = final.coef_
                if coef is not None:
                    feat_names = vectorizer.get_feature_names_out() if hasattr(vectorizer,
                                                                               "get_feature_names_out") \
                        else vectorizer.get_feature_names()
                    import numpy as np
                    coefs = coef.ravel()
                    top_pos_idx = np.argsort(-coefs)[:40]
                    top_neg_idx = np.argsort(coefs)[:40]
                    with open(os.path.join(DEBUG_DIR, f"token_debug_{stamp}.txt"), "w", encoding="utf-8") as f:
                        f.write("Top positive tokens (spam):\n")
                        for i in top_pos_idx:
                            f.write(f"{feat_names[i]}\t{coefs[i]:.6f}\n")
                        f.write("\nTop negative tokens (ham):\n")
                        for i in top_neg_idx:
                            f.write(f"{feat_names[i]}\t{coefs[i]:.6f}\n")
                    logging.info("Wrote token debug to %s", os.path.join(DEBUG_DIR, f"token_debug_{stamp}.txt"))
            except Exception:
                logging.exception("Failed while writing token debug.")
        except Exception:
            logging.exception("Failed while writing debug artifacts.")
    return df


# ---------- Main ----------
def main():
    try:
        model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VEC_PATH)
    except Exception as e:
        logging.error("Model load error: %s", e)
        return
    try:
        service = get_gmail_service()
    except Exception as e:
        logging.error("Gmail service error: %s", e)
        return
    try:
        msg_ids = fetch_unread_inbox_message_ids(service)
        if not msg_ids:
            logging.info("No unread messages found.")
            return
        df = process_messages(service, model, vectorizer, msg_ids)
        logging.info("Processing complete. %d messages handled. Spam flagged: %d", len(df), int(df['pred'].sum()))
    except Exception as e:
        logging.error("Processing error: %s\n%s", e, traceback.format_exc())


if __name__ == "__main__":
    main()
