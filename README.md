# 🎬 CineIQ: Hybrid NLP Recommendation Architecture

An explainable, high-throughput movie recommendation backend. This project shifts away from standard cosine-similarity tutorials to implement a multi-layered, hybrid algorithmic pipeline, designed to solve recommendation granularity and context-blindness.

## 🧠 Engineering Implementation

- **Algorithmic Blending:** Developed a pipeline that fuses Content-Based Filtering (TF-IDF matrix factorization on metadata) with Collaborative Filtering (SVD) for predictive user-item interactions.
- **Sentiment Re-Ranking Pipeline:** Integrated VADER NLP to dynamically analyze and re-rank the final $k$-suggestion pool. Instead of just matching genre tags, the engine evaluates the semantic sentiment of user preferences to surface highly nuanced matches.
- **Asynchronous Concurrency:** Engineered the backend using FastAPI and Uvicorn to handle concurrent I/O operations and API requests with sub-second latency, preventing UI thread blocking during heavy matrix calculations.

## 🛠️ Tech Stack & Execution

- **Stack:** Python, FastAPI, Pandas, Scikit-learn, NLTK, Streamlit

**To run the service locally:**

```bash
# 1. Boot the asynchronous backend
uvicorn api.main:app --reload

# 2. Initialize the interactive dashboard (Run in a separate terminal)
streamlit run app.py
```
