from fastapi import FastAPI, HTTPException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from recommender import CineiqEngine
import pandas as pd

# Initialize the API, the ML Engine, and the Sentiment Analyzer
app = FastAPI(title="CINEIQ Recommendation API")
engine = CineiqEngine()
analyzer = SentimentIntensityAnalyzer()

print("Loading user tags for sentiment analysis...")
try:
    tags_df = pd.read_csv('data/ml-latest-small/tags.csv')
except FileNotFoundError:
    print("Warning: tags.csv not found. Sentiment will be neutral.")
    tags_df = pd.DataFrame(columns=['movieId', 'tag'])

def get_sentiment_multiplier(movie_id):
    """Analyzes user tags using VADER to create a score multiplier (0.8x to 1.2x)"""
    movie_tags = tags_df[tags_df['movieId'] == movie_id]['tag'].dropna().tolist()
    
    if not movie_tags:
        return 1.0 # Neutral multiplier if no tags exist
    
    # Calculate sentiment for each tag (-1.0 to 1.0)
    compound_scores = [analyzer.polarity_scores(str(tag))['compound'] for tag in movie_tags]
    avg_sentiment = sum(compound_scores) / len(compound_scores)
    
    # Scale the sentiment from a [-1, 1] range to a multiplier of [0.8, 1.2]
    multiplier = 1.0 + (avg_sentiment * 0.2)
    return multiplier

@app.get("/")
def health_check():
    return {"status": "CINEIQ Engine is online and ready!"}

@app.get("/recommend")
def recommend_endpoint(user_id: int, title: str, top_n: int = 5):
    # 1. Fetch a wider pool of base recommendations from our Hybrid Engine
    base_recs = engine.get_hybrid_recommendations(user_id, title, top_n=top_n * 2)
    
    if isinstance(base_recs, dict) and "error" in base_recs:
        raise HTTPException(status_code=404, detail=base_recs["error"])

    # 2. The Sentiment Re-Ranker Layer
    reranked_recs = []
    for rec in base_recs:
        # Find the movie ID to look up its tags
        matched_movie = engine.movies_df[engine.movies_df['title'] == rec['title']]
        if not matched_movie.empty:
            movie_id = matched_movie.iloc[0]['movieId']
            
            # Apply the VADER sentiment multiplier
            multiplier = get_sentiment_multiplier(movie_id)
            final_score = rec['hybrid_score'] * multiplier
            
            # Enhance the explainability layer
            explanation = rec['reason']
            if multiplier > 1.05:
                explanation += " (Boosted by highly positive audience sentiment!)"
            elif multiplier < 0.95:
                explanation += " (Penalized slightly due to mixed audience sentiment.)"

            reranked_recs.append({
                "title": rec['title'],
                "final_score": round(final_score, 4),
                "explanation": explanation
            })

    # 3. Sort by the new Sentiment-Adjusted score and return the exact requested amount
    final_results = sorted(reranked_recs, key=lambda x: x['final_score'], reverse=True)[:top_n]
    
    return {
        "user_id": user_id, 
        "reference_movie": title, 
        "recommendations": final_results
    }

@app.get("/similar")
def similar_endpoint(title: str, top_n: int = 5):
    """Endpoint to fetch similar movies based on content metadata."""
    recs = engine.get_similar_movies(title, top_n)
    
    if isinstance(recs, dict) and "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
        
    return {
        "reference_movie": title, 
        "similar_movies": recs
    }