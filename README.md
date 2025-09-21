# Multilingual Language Detection Model

![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning model for identifying the language of a given text snippet from a set of 22 languages. This project leverages classical NLP techniques and a Support Vector Machine (SVM) classifier to achieve high accuracy and reliable performance.


## 📝<a name="-project-overview"></a> Project Overview

The goal of this project is to build an accurate and efficient language detection system. The model is trained on a balanced dataset containing text from 22 different languages. The process involves comprehensive exploratory data analysis, text preprocessing using TF-IDF vectorization with character n-grams, and training multiple machine learning models to find the optimal solution. The final model is a Support Vector Classifier (SVC) that demonstrates superior performance in distinguishing between languages with similar character sets.


## ✨Features

- **Identifies 22 Languages:** Accurately classifies text into languages including English, Spanish, French, Russian, Arabic, and more.
- **High Accuracy:** Achieves a final accuracy of **97.65%** on the test set.
- **Efficient Preprocessing:** Uses a TF-IDF Vectorizer with character n-grams, which is highly effective for language identification tasks.
- **Robust Model:** Built with a Support Vector Machine (SVM) for balanced and reliable predictions.
- **Ready for Inference:** Includes a simple command-line interface to predict the language of new text.


## 💻Technologies Used

- **Programming Language:** Python 3.9+
- **Machine Learning:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Lab, Git & GitHub


## 📊Dataset

The model was trained on the **Language Detection** dataset from Kaggle, contributed by Zara Khan. It contains over 22,000 text samples, with 1000 samples for each of the 22 languages, ensuring a perfectly balanced training environment.

- **Link to Dataset:** [Language Detection on Kaggle](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)


## 📁Project Structure

The project is organized into a modular structure for clarity and scalability:
```
language-detection/
├── data/              # Contains raw and processed datasets
├── notebooks/         # Jupyter notebook for EDA and experimentation
├── saved_models/      # Stores the trained model and vectorizer
├── src/               # Source code for training and prediction
│   ├── train.py       # Script to run the full training pipeline
│   └── predict.py     # Script to make predictions on new text
├── .gitignore         # Specifies files for Git to ignore
├── README.md          # Project documentation (this file)
└── requirements.txt   # Python package dependencies
```

## 🛠️Setup and Installation

To get this project running on your local machine, follow these steps:

### **1. Clone the Repository**

```bash
git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name
```

### **2. Create and Activate a Virtual Environment**

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **3. Install Dependencies**

Install all the required Python packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

## 🚀 Usage
You can use the project in two ways: training the model from scratch or using the pre-trained model for predictions.

### **1. Training the Model**

To retrain the model, run the train.py script from the src directory. This will process the data, train a new SVM model, and save the model and vectorizer artifacts to the saved_models/ directory.

```bash
cd src
python train.py
```

### **2. Making a Prediction**

Use the predict.py script to classify a new sentence. Pass the text as a command-line argument.

```bash
cd src
python predict.py "This is a sentence written in English."
# Expected Output: Predicted Language: English

python predict.py "Ceci est une phrase écrite en français."
# Expected Output: Predicted Language: French
```
**Note:** The model performs best on sentences with sufficient length to capture unique linguistic features. Very short or ambiguous text may result in less accurate predictions.



## 📈Model Performance
Two models were evaluated for this task: **Multinomial Naive Bayes** and a **Support Vector Machine (SVM)**. The Multinomial Naive Bayes achieved an overall accuracy of ***97.65%***, while the Support Vector Machine achieved an overall accuracy of ***98.32%***. The SVM was chosen as the final model due to its superior performance in handling challenging cases.

The initial Naive Bayes model struggled with low precision for English (0.72), frequently misclassifying other Latin-alphabet languages as English. The SVM model significantly improved this, raising the precision for English to 0.83, indicating a more robust and reliable classifier.

Confusion Matrix (SVM Model)
The following confusion matrix shows the predictions of the final SVM model on the test set. The strong diagonal line indicates a high number of correct predictions across all languages.


## 📄 License
This project is licensed under the MIT License.

