# sms_spam_tfidf_logreg.py
# TF-IDF + Logistic Regression spam detector. Prints precision/recall/F1 and
# top positive/negative features. Saves confusion matrix figure.
#
# If offline: put the UCI "SMSSpamCollection" file (tab-separated "label<TAB>message")
# next to this script and set LOCAL_PATH below. Otherwise the script tries a known URL.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

LOCAL_PATH = "SMSSpamCollection"  # fallback local filename (optional)
REMOTE_TSV = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv"


def load_data():
    """Load SMS spam dataset from local file or remote URL."""
    # Try local file first
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH, sep="\t", header=None, names=["label", "message"])
        return df

    # Try remote URL
    try:
        df = pd.read_csv(REMOTE_TSV, sep="\t", header=None, names=["label", "message"])
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load SMS dataset. Place 'SMSSpamCollection' next to this script "
            "or ensure internet is available."
        ) from e


def main():
    os.makedirs("outputs/sms", exist_ok=True)

    df = load_data()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"].values,
        df["label"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["label"].values,
    )

    # TF-IDF vectorizer
    vect = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
    )

    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    # Logistic Regression
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )
    clf.fit(Xtr, y_train)

    # Predictions
    y_pred = clf.predict(Xte)

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
    disp.plot(cmap="Blues")

    plt.title("SMS Spam Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/sms/cm_sms.png", dpi=200)
    plt.close()
    print("Saved: outputs/sms/cm_sms.png")

    # Top positive/negative coefficients (spam indicators)
    feature_names = np.array(vect.get_feature_names_out())
    spam_idx = np.where(clf.classes_ == "spam")[0][0]
    coefs = clf.coef_[spam_idx]

    top_pos = np.argsort(coefs)[-20:][::-1]
    top_neg = np.argsort(coefs)[:20]

    print("\nTop 20 SPAM-indicative features:")
    for f, w in zip(feature_names[top_pos], coefs[top_pos]):
        print(f"{f:30s} {w: .3f}")

    print("\nTop 20 HAM-indicative features:")
    for f, w in zip(feature_names[top_neg], coefs[top_neg]):
        print(f"{f:30s} {w: .3f}")


if __name__ == "__main__":
    main()
