# Importing the libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # Updated import
from sklearn.naive_bayes import MultinomialNB # Better for text than GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download NLTK data
nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv('/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords and stem
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to all reviews
corpus = [preprocess_text(review) for review in dataset['Review']]

# Creating the Bag of Words model, convert words to numbers vectors
cv = CountVectorizer(max_features=1500) 
X = cv.fit_transform(corpus).toarray() 
y = dataset.iloc[:, 1].values

# Splitting the dataset (using modern train_test_split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=0,
    stratify=y  # Maintain class distribution
)

# Using MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting and evaluating
y_pred = classifier.predict(X_test)

# Evaluation metrics
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))