import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources if not already downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (same as in training)
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\b(salary|experience|openings|not disclosed|yrs|pa)\b', '', text)  # Remove boilerplate
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Function to classify a new job description
def classify_job_description(job_description):
    # Preprocess the input
    processed_text = preprocess_text(job_description)
    
    # Transform the text using the loaded vectorizer
    text_tfidf = vectorizer.transform([processed_text])
    
    # Predict the category
    predicted_category = model.predict(text_tfidf)[0]
    
    return predicted_category

# Example usage in runtime
if __name__ == "__main__":
    print("Enter a job description to classify (or type 'exit' to quit):")
    while True:
        job_description = input("Job Description: ")
        if job_description.lower() == 'exit':
            break
        category = classify_job_description(job_description)
        print(f"Predicted Category: {category}\n")
