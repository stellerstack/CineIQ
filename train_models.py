import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from data_loader import load_data
import mlflow
from surprise.model_selection import train_test_split
from surprise import accuracy

def train_and_save_models():
    print("Loading data...")
    movies, ratings = load_data()
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')

    # ==========================================
    # 1. CONTENT-BASED MODEL (TF-IDF)
    # ==========================================
    print("Training Content-Based Model (TF-IDF)...")
    # We use TF-IDF to convert the cleaned genres into a mathematical matrix
    tfidf = TfidfVectorizer(stop_words='english')
    # Fill NaN values with empty strings just in case
    movies['genres_cleaned'] = movies['genres_cleaned'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['genres_cleaned'])
    
    # Compute the cosine similarity matrix
    # This matrix contains the similarity score of every movie with every other movie
    print("Computing cosine similarity...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
# Save ONLY the TF-IDF matrix, not the Pandas DataFrame
    with open('models/content_model.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)

    # ==========================================
    # 2. COLLABORATIVE MODEL (SVD) with MLflow
    # ==========================================
    print("Training Collaborative Model (SVD) and logging to MLflow...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Set up the MLflow experiment
    mlflow.set_experiment("CINEIQ_SVD_Training")
    
    with mlflow.start_run():
        # Define and log our hyperparameter
        n_factors = 100
        mlflow.log_param("n_factors", n_factors)
        
        # Split data to calculate RMSE (Root Mean Square Error)
        trainset, testset = train_test_split(data, test_size=0.2)
        
        svd = SVD(n_factors=n_factors, random_state=42)
        svd.fit(trainset)
        
        # Test the model and log the accuracy metric
        predictions = svd.test(testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        mlflow.log_metric("rmse", rmse)
        
        # Retrain on the FULL dataset for actual production use
        full_trainset = data.build_full_trainset()
        final_svd = SVD(n_factors=n_factors, random_state=42)
        final_svd.fit(full_trainset)
        
        # Save the model locally and log the file to MLflow
        with open('models/svd_model.pkl', 'wb') as f:
            pickle.dump(final_svd, f)
            
        mlflow.log_artifact('models/svd_model.pkl')
        
    print("\nSuccess! Models trained and logged.")

if __name__ == "__main__":
    train_and_save_models()