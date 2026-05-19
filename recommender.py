import pandas as pd
import pickle
import difflib
from data_loader import load_data

class CineiqEngine:
    def __init__(self):
        print("Initializing CINEIQ Engine...")
        
        # Load the dataframe directly from our data loader to avoid pickle bugs
        self.movies_df, _ = load_data()
        
        # Load ONLY the math matrix from the pickle
        with open('models/content_model.pkl', 'rb') as f:
            self.cosine_sim = pickle.load(f)
        
        with open('models/svd_model.pkl', 'rb') as f:
            self.svd_model = pickle.load(f)

        # Create a reverse lookup table
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()

    def get_hybrid_recommendations(self, user_id, movie_title, weight_content=0.5, weight_collab=0.5, top_n=10):
        # 1. Handle fuzzy matching (so "toy story" finds "Toy Story (1995)")
        titles = self.movies_df['title'].tolist()
        close_matches = difflib.get_close_matches(movie_title, titles, n=1, cutoff=0.5)
        
        if not close_matches:
            return {"error": f"Movie '{movie_title}' not found in database."}
        
        exact_title = close_matches[0]
        idx = self.indices[exact_title]

        # 2. Get Content-Based Scores (Genre similarity)
        # Handle cases where a movie might have multiple entries
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        results = []
        for i, content_score in sim_scores:
            movie_id = self.movies_df.iloc[i]['movieId']
            title = self.movies_df.iloc[i]['title']
            genres = self.movies_df.iloc[i]['genres']
            
            # Skip the reference movie itself
            if title == exact_title:
                continue

            # 3. Get Collaborative Score (SVD predicted rating)
            predicted_rating = self.svd_model.predict(user_id, movie_id).est
            
            # Normalize the 0-5 rating to a 0-1 scale so it blends cleanly with the content score
            collab_score = predicted_rating / 5.0
            
            # 4. The Weighted Hybrid Score
            hybrid_score = (content_score * weight_content) + (collab_score * weight_collab)
            
            # 5. The Explainability Layer (Rule-based templates)
            if content_score > collab_score:
                reason = f"Based on its genres ({genres.replace('|', ', ')}), it's highly similar to {exact_title}."
            else:
                reason = "Based on our matrix factorization, users with your specific taste patterns rated this highly."

            results.append({
                'title': title,
                'hybrid_score': float(hybrid_score),
                'reason': reason
            })

        # Sort the results by the final hybrid score in descending order
        results = sorted(results, key=lambda x: x['hybrid_score'], reverse=True)[:top_n]
        return results

    def get_similar_movies(self, movie_title, top_n=5):
        """Returns movies similar to the target based strictly on content (TF-IDF)."""
        titles = self.movies_df['title'].tolist()
        titles_lower = [str(t).lower() for t in titles]
        search_query = str(movie_title).lower()
        
        close_matches = difflib.get_close_matches(search_query, titles_lower, n=1, cutoff=0.4)
        if not close_matches:
            return {"error": f"Movie '{movie_title}' not found in database."}
            
        match_index = titles_lower.index(close_matches[0])
        exact_title = titles[match_index]
        idx = self.indices[exact_title]
        
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
            
        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort them by highest similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Skip the first one (it's the exact same movie you searched for)
        sim_scores = sim_scores[1:top_n+1]
        
        results = []
        for i, score in sim_scores:
            title = self.movies_df.iloc[i]['title']
            results.append({
                "title": title, 
                "similarity_score": round(float(score), 4)
            })
            
        return results

if __name__ == "__main__":
    # Quick test to make sure it works!
    engine = CineiqEngine()
    print("\n--- Testing Hybrid Engine ---")
    print("User ID: 1 | Target Movie: 'Matrix'")
    
    recs = engine.get_hybrid_recommendations(user_id=1, movie_title="Matrix", top_n=5)
    
    for r in recs:
        print(f"\n🎬 {r['title']} (Score: {r['hybrid_score']:.3f})")
        print(f"💡 Why? {r['reason']}")