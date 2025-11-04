# GmailSpamDetection
Detects spam on your Gmail.

# Gmail Spam Monitor + Trainer

This repository contains two tools:
- `train_spam_detector.py` — trains a TF-IDF + Logistic Regression spam classifier and saves `models/model.joblib` and `models/vectorizer.joblib`.
- `gmail_spam_monitor.py` — uses Gmail API (OAuth2) to fetch unread emails, predicts spam with the saved model, quarantines and moves spam to Gmail's Spam label, and logs predictions.

## Quick start

1. Create a Python venv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Train a model (or use provided model artifacts):
   Train with the default demo SMS dataset:
     ```bash
     python train_spam_detector.py --out_dir models
     ```
    Or train on your labeled CSV (must contain text and label):
     ```bash
     python train_spam_detector.py --data my_labeled_emails.csv --out_dir models
     ```

3. Set up Gmail API OAuth credentials:
  Create a Google Cloud project, enable Gmail API, create an OAuth Client ID (Desktop app), and download credentials.json.
  Configure the OAuth consent screen and add your email as a test user (if using testing mode).

4. Run Gmail monitor once to authenticate:
  ```bash
  python gmail_spam_monitor.py
  ```
  Approve the consent screen in the browser. This creates token.json.

5. 

