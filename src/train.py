import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

df = pd.read_csv('D:\\Deep_Learning\\Language Detection\\data\\raw\\dataset\\Language Detection.csv')

x = np.array(df['Text'])
y = np.array(df['language'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with open('D:\\Deep_Learning\\Language Detection\\saved_models\\vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training the SVM model...")
model_svm = SVC(kernel='linear')
model_svm.fit(X_train_vec, y_train)
print("Training complete!")

y_pred_svm = model_svm.predict(X_test_vec)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nSVM Model Accuracy: {accuracy_svm:.4f}")

print("\nClassification Report (SVM):\n", classification_report(y_test, y_pred_svm))

with open('D:\\Deep_Learning\\Language Detection\\saved_models\\model_svm.pkl', 'wb') as f:
    pickle.dump(model_svm, f)