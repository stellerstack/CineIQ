import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# Page config must be the very first Streamlit command
st.set_page_config(page_title="CINEIQ Dashboard", layout="wide")

@st.cache_data
def load_and_prep_data():
    movies = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    
    # Extract year from title, e.g., "Toy Story (1995)" -> 1995
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    # Create decade column (1995 -> 1990)
    movies['decade'] = (movies['year'] // 10) * 10
    
    return movies, ratings

# Load the data
movies_df, ratings_df = load_and_prep_data()

st.title("🎥 CINEIQ: User Taste & Discovery Dashboard")
st.markdown("An explainable, hybrid movie recommendation engine.")

# ==========================================
# SIDEBAR: Dynamic User Settings
# ==========================================
st.sidebar.header("User Settings")

# Find the actual maximum User ID in your dataset so it updates automatically
max_user_id = int(ratings_df['userId'].max())

# Use that dynamic max_user_id in the input
user_id = st.sidebar.number_input(
    "Enter User ID", 
    min_value=1, 
    max_value=max_user_id, 
    value=1
)

# Filter data for the active user
user_ratings = ratings_df[ratings_df['userId'] == user_id].merge(movies_df, on='movieId')

if user_ratings.empty:
    st.warning("No rating history found for this user.")
else:
    # ==========================================
    # VISUALIZATION 1: Genre Radar Chart
    # ==========================================
    st.header(f"Taste Profile: User {user_id}")
    
    # Explode genres so "Action|Adventure" becomes two separate rows
    user_ratings['genres_split'] = user_ratings['genres'].str.split('|')
    exploded_genres = user_ratings.explode('genres_split')
    
    # Count genres for movies the user actually liked (rated >= 3.5)
    liked_movies = exploded_genres[exploded_genres['rating'] >= 3.5]
    genre_counts = liked_movies['genres_split'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    top_genres = genre_counts.head(6) # Take top 6 for a clean radar chart
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Genre Affinity")
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=top_genres['Count'],
          theta=top_genres['Genre'],
          fill='toself',
          line_color='#E50914'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ==========================================
    # VISUALIZATION 2: Decade Preferences
    # ==========================================
    with col2:
        st.subheader("Decade Preferences")
        decade_counts = user_ratings['decade'].value_counts().sort_index().reset_index()
        decade_counts.columns = ['Decade', 'Movies Watched']
        
        # Drop NaN decades
        decade_counts = decade_counts.dropna()
        
        fig_bar = px.bar(decade_counts, x='Decade', y='Movies Watched', 
                         color_discrete_sequence=['#E50914'])
        fig_bar.update_layout(xaxis_type='category')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # THE RECOMMENDATION ENGINE (Via FastAPI)
    # ==========================================
    st.header("Discovery Engine")
    
    # Let the user pick a movie they liked to base recommendations on
    top_rated_titles = user_ratings[user_ratings['rating'] >= 4.0]['title'].tolist()
    
    if top_rated_titles:
        selected_movie = st.selectbox("Pick a movie you loved:", top_rated_titles)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Querying Hybrid Engine & Analyzing Sentiment..."):
                # Call our FastAPI backend!
                api_url = f"http://127.0.0.1:8000/recommend?user_id={user_id}&title={selected_movie}&top_n=5"
                try:
                    response = requests.get(api_url)
                    if response.status_code == 200:
                        data = response.json()
                        recs = data.get("recommendations", [])
                        
                        st.success("Recommendations Generated!")
                        for i, rec in enumerate(recs):
                            # Use Streamlit expanders for a clean UI
                            with st.expander(f"{i+1}. {rec['title']} (Score: {rec['final_score']})", expanded=(i==0)):
                                st.markdown(f"**Why we recommend this:** {rec['explanation']}")
                    else:
                        st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Is your Uvicorn server running?")
    else:
        st.info("Rate some movies highly (4+) to unlock recommendations!")