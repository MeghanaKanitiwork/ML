import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import joblib

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function to assign initial category labels
def assign_category(job_title, key_skills, functional_area, industry):
    job_title = job_title.lower()
    key_skills = key_skills.lower() if pd.notna(key_skills) else ""
    functional_area = functional_area.lower() if pd.notna(functional_area) else ""
    industry = industry.lower() if pd.notna(industry) else ""
    
    # IT keywords
    it_keywords = ['software', 'developer', 'engineer', 'technical', 'programming', 'webmaster', 'datawarehousing']
    it_skill_keywords = ['azure', 'docker', 'aws', 'sql', 'python', 'java']
    
    # Marketing keywords
    marketing_keywords = ['marketing', 'advertising', 'media planning', 'social media', 'seo', 'campaign']
    marketing_skill_keywords = ['google analytics', 'content marketing', 'email marketing', 'sem']
    
    # Healthcare keywords
    healthcare_keywords = ['nurse', 'doctor', 'medical', 'clinical', 'healthcare', 'patient', 'pharma', 'hospital', 'physician', 'therapy', 'nursing', 'surgery', 'medic']
    healthcare_skill_keywords = ['patient care', 'clinical research', 'medical coding', 'health informatics', 'nursing skills', 'surgical assistance', 'health management']    

    if (any(keyword in job_title for keyword in it_keywords) or 
        any(keyword in key_skills for keyword in it_skill_keywords) or 
        'it-software' in industry or 'software services' in industry):
        return 'IT'
    elif (any(keyword in job_title for keyword in marketing_keywords) or 
          any(keyword in key_skills for keyword in marketing_skill_keywords) or 
          any(keyword in functional_area for keyword in ['marketing', 'advertising'])):
        return 'Marketing'
    elif (any(keyword in job_title for keyword in healthcare_keywords) or 
          any(keyword in key_skills for keyword in healthcare_skill_keywords) or 
          'healthcare' in industry or 'pharma' in industry):
        return 'Healthcare'
    else:
        return 'Others'

# Enhanced text preprocessing
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

# Load dataset
try:
    data = pd.read_csv('B:/job_ml_assignment/categorized_jobs.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('B:/job_ml_assignment/categorized_jobs.csv', encoding='latin1')

# Verify data
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)

# Handle missing values
data['Key Skills'] = data['Key Skills'].fillna('')
data['Job Title'] = data['Job Title'].fillna('')
data['Functional Area'] = data['Functional Area'].fillna('')
data['Industry'] = data['Industry'].fillna('')

# Apply rule-based labeling
data['Category'] = data.apply(lambda x: assign_category(x['Job Title'], x['Key Skills'], x['Functional Area'], x['Industry']), axis=1)

# Combine text fields
data['Combined_Text'] = data['Job Title'] + ' ' + data['Key Skills'] + ' ' + data['Functional Area'] + ' ' + data['Industry']
data['Processed_Text'] = data['Combined_Text'].apply(preprocess_text)

# Simulate partial labeling: Sample 500 rows to ensure category balance
labeled_data = data.sample(n=500, random_state=42).copy()

# Check category distribution
category_counts = labeled_data['Category'].value_counts()
print("Category Distribution in Labeled Data:\n", category_counts)

# Filter categories with at least 2 samples
valid_categories = category_counts[category_counts >= 2].index
labeled_data = labeled_data[labeled_data['Category'].isin(valid_categories)].copy()
print("Filtered Labeled Data Shape:", labeled_data.shape)
print("Filtered Category Distribution:\n", labeled_data['Category'].value_counts())

# Prepare features and labels
X = labeled_data['Processed_Text']
y = labeled_data['Category']

# Safety check
if len(labeled_data) < 10 or y.nunique() < 2:
    raise ValueError("Insufficient samples or categories after filtering. Increase sample size or adjust categories.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train and evaluate multiple models
models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'SVM': LinearSVC(random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate
    print(f"\nResults for {name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Cross-validation
    scores = cross_val_score(model, vectorizer.transform(X), y, cv=5)
    print(f"Cross-Validation Scores ({name}):", scores, "Mean:", scores.mean())

# Select best model (Naive Bayes for simplicity in saving)
best_model = MultinomialNB(alpha=0.1)
best_model.fit(X_train_tfidf, y_train)

# Predict for unlabeled data
unlabeled_data = data.drop(labeled_data.index).copy()
if not unlabeled_data.empty:
    X_unlabeled = vectorizer.transform(unlabeled_data['Processed_Text'])
    unlabeled_data['Predicted_Category'] = best_model.predict(X_unlabeled)
    print("\nSample Predictions for Unlabeled Data:\n", unlabeled_data[['Job Title', 'Predicted_Category']].head())

# Save model and vectorizer
joblib.dump(best_model, 'nb_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Save predictions
data['Predicted_Category'] = data['Category']
if not unlabeled_data.empty:
    data.loc[unlabeled_data.index, 'Predicted_Category'] = unlabeled_data['Predicted_Category']
data[['Job Title', 'Key Skills', 'Functional Area', 'Industry', 'Predicted_Category']].to_csv('categorized_jobs.csv', index=False)
print("\nPredictions saved to 'categorized_jobs.csv'")

# Feature importance (top words per category)
feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(best_model.classes_):
    top_words = feature_names[best_model.feature_log_prob_[i].argsort()[-10:]]
    print(f"\nTop words for {category}: {top_words}")
