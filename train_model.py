import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOP = set(stopwords.words('english'))

def clean_text(s):
    if pd.isnull(s):
        return ""
    s = s.lower()
    s = re.sub(r'http\S+','',s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [t for t in s.split() if t not in STOP and len(t)>1]
    return " ".join(tokens)

def main():
    df = pd.read_csv("data/fake_job_postings.csv")  # change if filename different

    text_cols = ['title','company_profile','description','requirements','benefits']
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)

    target_col = 'fraudulent'
    y = df[target_col].astype(int)
    X = df['text']

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_vec = vec.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_vec, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(clf, "models/model.joblib")
    joblib.dump(vec, "models/vectorizer.joblib")
    print("Model saved in /models folder")

if __name__ == "__main__":
    main()
