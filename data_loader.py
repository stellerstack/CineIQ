import pandas as pd
import os
import urllib.request
import zipfile

def download_and_extract_data():
    """Downloads MovieLens small dataset for rapid development and testing."""
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "ml-latest-small.zip"
    extract_dir = "data"

    # Create data directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Download only if we don't already have the files
    target_csv = os.path.join(extract_dir, "ml-latest-small", "movies.csv")
    if not os.path.exists(target_csv):
        print("Downloading MovieLens dataset (this might take a moment)...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        os.remove(zip_path) # Clean up the zip file
        print("Download and extraction complete.")
    else:
        print("Dataset already exists locally. Skipping download.")

def load_data():
    """Loads CSVs into Pandas DataFrames and applies initial transformations."""
    movies_path = 'data/ml-latest-small/movies.csv'
    ratings_path = 'data/ml-latest-small/ratings.csv'
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Feature Engineering: 
    # The 'genres' column uses pipes (Action|Adventure). 
    # We replace them with spaces (Action Adventure) so our TF-IDF model can parse them as individual "words".
    movies['genres_cleaned'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    return movies, ratings

if __name__ == "__main__":
    download_and_extract_data()
    movies_df, ratings_df = load_data()
    
    print(f"\nSuccess! Loaded {len(movies_df)} movies and {len(ratings_df)} ratings.")
    print("\nPreview of movies data:")
    print(movies_df[['movieId', 'title', 'genres_cleaned']].head(3))