#!/usr/bin/env python3
"""
train_spam_detector.py

Usage examples:
  # Train on default public dataset and save model
  python train_spam_detector.py --out_dir models

  # Train using your own CSV (must contain either 'text' or 'subject'/'body' plus 'label' for training)
  python train_spam_detector.py --data path/to/your_emails.csv --out_dir models

  # Predict only (load saved model and vectorizer)
  python train_spam_detector.py --predict_only --model models/model.joblib --vectorizer models/vectorizer.joblib
  --test_csv my_email_to_check.csv
"""

import argparse
import os
import sys
import re
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# --------- Utilities and preprocessing (important parts commented) ---------
def download_uciml_sms_dataset():
    """
    Downloads the UCI SMS Spam Collection dataset (text file).
    Returns a pandas DataFrame with columns: label (spam/ham), text.
    Source: UCI ML repository.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Downloading UCI SMS dataset from:", url)
    data = urllib.request.urlopen(url).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    # the file inside zip is 'SMSSpamCollection'
    raw = z.read('SMSSpamCollection').decode('utf-8', errors='ignore')
    rows = [line.split('\t', 1) for line in raw.splitlines() if '\t' in line]
    df = pd.DataFrame(rows, columns=['label', 'text'])
    return df


def load_csv_as_dataframe(path):
    """
    Tries to load a CSV and normalizes it to columns ['text','label'] if possible.
    Accepts CSVs with:
      - 'text' and 'label', OR
      - 'subject' and 'body' (will concatenate), and optionally 'label'.
    If no label column present, returns dataframe with label=None (useful for prediction).
    """
    df = pd.read_csv(path)
    # unify text column
    if 'text' in df.columns:
        text = df['text'].astype(str)
    else:
        # combine subject + body if present
        parts = []
        if 'subject' in df.columns:
            parts.append(df['subject'].fillna('').astype(str))
        if 'body' in df.columns:
            parts.append(df['body'].fillna('').astype(str))
        if parts:
            text = parts[0]
            for p in parts[1:]:
                text = text + ' ' + p
        else:
            # fallback: use first text-like column
            text_col = None
            for c in df.columns:
                if df[c].dtype == object and c.lower() not in ('label', 'id'):
                    text_col = c
                    break
            if text_col is None:
                raise ValueError("Couldn't find a text column in CSV. Expected 'text' or 'subject'/'body'.")
            text = df[text_col].astype(str)
    out = pd.DataFrame({'text': text})
    if 'label' in df.columns:
        out['label'] = df['label']
    return out


def simple_preprocess(s):
    """Lowercase + remove repeated whitespace + strip."""
    s = s.lower()
    # optional: remove email headers / quoted lines (simple heuristic)
    s = re.sub(r'(^>.*$)', ' ', s, flags=re.MULTILINE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# --------- Main pipeline ---------
def train_and_save(df, out_dir, model_name='model.joblib', vectorizer_name='vectorizer.joblib'):
    # require 'label' column to train
    if 'label' not in df.columns:
        raise ValueError("Dataframe must contain 'label' column to train.")
    # normalize labels to 0/1
    df = df.copy()
    df['label'] = df['label'].astype(str)
    df['label'] = df['label'].str.lower().map(lambda x: 1 if x in ('spam', '1', 'true', 't', 'yes', 'y') else 0)
    df['text'] = df['text'].astype(str).map(simple_preprocess)

    X = df['text'].values
    y = df['label'].values

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF vectorizer: unigrams + bigrams, min_df to ignore very rare tokens
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # simple but strong baseline classifier
    clf = LogisticRegression(max_iter=200, solver='liblinear')

    # cross-validation score (5-fold)
    print("Running 5-fold CV (accuracy)...")
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print("CV accuracy scores:", np.round(cv_scores, 4))
    print("CV accuracy mean: {:.4f}".format(cv_scores.mean()))

    # fit on train and evaluate on test
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    print("\n=== Test set evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # save artifacts
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, model_name)
    vec_path = os.path.join(out_dir, vectorizer_name)
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")
    return clf, vectorizer


def predict_from_csv(model, vectorizer, csv_path, out_csv=None):
    """
    Load a CSV (no label required), runs predictions and prints top examples.
    Requires the CSV to contain either 'text' or 'subject'/'body' columns — see load_csv_as_dataframe.
    """
    df = load_csv_as_dataframe(csv_path)
    df['text_proc'] = df['text'].astype(str).map(simple_preprocess)
    X = df['text_proc'].values
    X_tfidf = vectorizer.transform(X)
    probs = model.predict_proba(X_tfidf)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df_out = df.copy()
    df_out['spam_prob'] = probs
    df_out['pred_label'] = preds
    if out_csv:
        df_out.to_csv(out_csv, index=False)
        print("Saved predictions to", out_csv)
    # show top suspicious emails
    top = df_out.sort_values('spam_prob', ascending=False).head(10)[['text', 'spam_prob', 'pred_label']]
    print("\nTop 10 highest-spam-prob messages:")
    for i, row in top.iterrows():
        snippet = row['text'][:200].replace('\n', ' ')
        print(f"prob={row['spam_prob']:.3f} pred={row['pred_label']}  text_snippet={snippet}")
    return df_out


# --------- CLI and orchestration ---------
def main(args):
    # load data
    if args.data:
        print("Loading dataset from:", args.data)
        df = load_csv_as_dataframe(args.data)
        # if no label present and user wants training, error out
        if (not args.predict_only) and ('label' not in df.columns):
            print("Error: provided CSV doesn't contain 'label' column required for training.")
            sys.exit(1)
    else:
        if args.predict_only:
            print("Predict-only mode requires --test_csv and saved model/vectorizer.")
            sys.exit(1)
        print("No dataset provided — downloading UCI SMS Spam Collection example dataset.")
        df = download_uciml_sms_dataset()

    if args.predict_only:
        # load provided model and vectorizer
        print("Loading model and vectorizer...")
        model = joblib.load(args.model)
        vectorizer = joblib.load(args.vectorizer)
        predict_from_csv(model, vectorizer, args.test_csv, out_csv=args.out_csv)
        return

    # train
    clf, vectorizer = train_and_save(df, args.out_dir)

    # if user provided their own test CSV to check, run prediction now
    if args.test_csv:
        print("\nNow running predictions on user-provided test CSV:", args.test_csv)
        predict_from_csv(clf, vectorizer, args.test_csv, out_csv=args.out_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a simple spam detector and test on your email CSV.")
    p.add_argument("--data", type=str, default=None,
                   help="Path to CSV for training (must contain 'label' and 'text' or 'subject'/'body')."
                        "If absent, downloads public SMS spam dataset.")
    p.add_argument("--out_dir", type=str, default="models", help="Directory to save model + vectorizer.")
    p.add_argument("--predict_only", action="store_true", help="Only run prediction using saved model/vectorizer.")
    p.add_argument("--model", type=str, help="Path to saved model.joblib (required if --predict_only).")
    p.add_argument("--vectorizer", type=str, help="Path to saved vectorizer.joblib (required if --predict_only).")
    p.add_argument("--test_csv", type=str, default=None, help="CSV of emails to test/predict (no label required).")
    p.add_argument("--out_csv", type=str, default=None, help="Optional path to write prediction outputs.")
    args = p.parse_args()
    main(args)
