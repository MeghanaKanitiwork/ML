# import joblib
# import pandas as pd
# import re
# import nltk
# import spacy
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import csr_matrix

# # Download necessary NLTK resources
# nltk.download('stopwords')

# # Load NLP model for better tokenization & lemmatization
# nlp = spacy.load("en_core_web_sm")

# # Load saved model assets
# vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure this matches training
# job_data = pd.read_parquet("B:/job_ml_assignment/categorized_jobs.parquet")  # Faster loading than CSV

# # Preprocess text efficiently
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"[^a-z\s]", "", text)
#     doc = nlp(text)
#     tokens = [token.lemma_ for token in doc if token.text not in nltk.corpus.stopwords.words("english")]
#     return " ".join(tokens)

# # Cache processed job skills
# job_data["Processed_Skills"] = job_data["Key Skills"].apply(preprocess_text)
# job_vectors = csr_matrix(vectorizer.transform(job_data["Processed_Skills"]))  # Convert to sparse matrix

# # Recommend jobs based on resume
# def recommend_jobs(resume_text, top_n=5):
#     try:
#         processed_resume = preprocess_text(resume_text)
#         resume_vector = csr_matrix(vectorizer.transform([processed_resume]))  # Sparse matrix for efficiency
#         similarities = cosine_similarity(resume_vector, job_vectors).flatten()
#         top_indices = similarities.argsort()[::-1][:top_n]
#         return job_data.iloc[top_indices][["Job Title", "Key Skills"]]
#     except Exception as e:
#         print(f"Error processing resume: {e}")
#         return pd.DataFrame()

# # Main script execution
# if __name__ == "__main__":
#     print("Paste your resume text below (press Enter twice to submit):\n")
#     print("-" * 50)

#     resume_lines = []
#     while True:
#         line = input()
#         if line.strip() == "":
#             break
#         resume_lines.append(line)

#     resume_text = "\n".join(resume_lines).strip()
#     print("\nResume received. Matching jobs...\n")

#     results = recommend_jobs(resume_text)
#     if results.empty:
#         print("No matching jobs found or something went wrong.")
#     else:
#         for idx, row in results.iterrows():
#             print(f"🔹 Job Title: {row['Job Title']}")
#             print(f"   Key Skills: {row['Key Skills']}\n{'-'*50}")

import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load saved model assets
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Must match training
job_data = pd.read_csv('B:/job_ml_assignment/categorized_jobs.csv')  # Must have 'Job Title' and 'Key Skills'

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b(salary|experience|openings|not disclosed|yrs|pa)\b', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess the job listings' key skills
job_data['Processed_Skills'] = job_data['Key Skills'].apply(preprocess_text)
job_vectors = vectorizer.transform(job_data['Processed_Skills'])

# Recommend jobs based on resume
def recommend_jobs(resume_text, top_n=5):
    try:
        processed_resume = preprocess_text(resume_text)
        resume_vector = vectorizer.transform([processed_resume])
        similarities = cosine_similarity(resume_vector, job_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        return job_data.iloc[top_indices][['Job Title', 'Key Skills']]
    except Exception as e:
        print(f" Error processing resume: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Paste your resume text below (press Enter twice to submit):\n")
    print("--------------------------------------------------------------")

    resume_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        resume_lines.append(line)

    resume_text = "\n".join(resume_lines).strip()

    print("\nResume received. Matching jobs...\n")

    results = recommend_jobs(resume_text)
    if results.empty:
        print("No matching jobs found or something went wrong.")
    else:
        for idx, row in results.iterrows():
            print(f"🔹 Job Title: {row['Job Title']}")
            print(f"   Key Skills: {row['Key Skills']}\n{'-'*50}")








