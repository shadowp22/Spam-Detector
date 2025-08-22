# Author: Ali Amini |----> aliamini9728@gmail.com

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ===============================
# NOTE FOR DEVELOPERS:
# Instead of downloading NLTK data every time (stopwords, punkt, etc.),
# you can download them once and keep them in a local folder (e.g., ./nltk_data).
# Then, set the NLTK_DATA environment variable or use nltk.data.path.append("path").
#
# Later in your code:
#   import nltk
#   nltk.data.path.append('nltk_data')
#
# This way, new developers won’t need to redownload data on every machine.
# ===============================

try:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except Exception as download_err:
    raise RuntimeError(f"Failed to download NLTK data: {str(download_err)}. Please connect to the internet and run the script again.")
else:
    print("NLTK data downloaded successfully.")

data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'sms_spam_collection.csv'), encoding='latin-1')
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

data['processed_message'] = data['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions):.4f}')
print('Classification Report:')
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))

try:
    joblib.dump(model, os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl'))
    joblib.dump(vectorizer, os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl'))
    print('Model and vectorizer saved to models/')
except Exception as e:
    raise RuntimeError(f"Failed to save model or vectorizer: {str(e)}")