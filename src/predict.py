import pickle
import re
import sys

with open('../saved_models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('../saved_models/model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_language(text):
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)

    return prediction[0]

if __name__ == "__main__":
    input_text = input("Please enter the text: ")
    if input_text.strip():
        lang = predict_language(input_text)
        print(f"\nInput text: '{input_text}'")
        print(f"Predicted Language: {lang}")
    else:
        print("Please provide text as a command-line argument.")
        print("Example: python predict.py This is a test sentence.")
