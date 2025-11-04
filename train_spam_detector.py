import os
import re
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# ---------- Simple preprocessing ----------
def simple_preprocess(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'(^>.*$)', ' ', s, flags=re.MULTILINE)  # remove quoted lines
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------- Dataset loader uses UCI SMS collection Dataset ----------
def download_uciml_sms_dataset() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Downloading UCI SMS dataset from:", url)
    data = urllib.request.urlopen(url).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    raw = z.read('SMSSpamCollection').decode('utf-8', errors='ignore')
    rows = [line.split('\t', 1) for line in raw.splitlines() if '\t' in line]
    df = pd.DataFrame(rows, columns=['label', 'text'])
    return df


# ---------- Training ----------
def train_and_save(out_dir: str = "models"):
    df = download_uciml_sms_dataset()
    print(f"Head:\n{df.head()}")

    # normalize and preprocess
    df['label'] = df['label'].astype(str).str.lower().map(lambda x: 1 if x == 'spam' else 0)
    df['text'] = df['text'].astype(str).map(simple_preprocess)

    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=200, solver='liblinear')

    print("Running 5-fold CV (accuracy)...")
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print("CV accuracy scores:", np.round(cv_scores, 4))
    print("CV accuracy mean: {:.4f}".format(cv_scores.mean()))

    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    print("\n=== Test set evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    vec_path = os.path.join(out_dir, "vectorizer.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")


if __name__ == "__main__":
    train_and_save("models")
