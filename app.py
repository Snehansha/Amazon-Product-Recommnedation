import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load Data
df = pd.read_csv('amazon.csv')
df['about_product'] = df['about_product'].fillna('')

# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['about_product'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend Function
def recommend(index, num=5):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices][['product_name', 'actual_price', 'rating']]

# Streamlit App
st.title("Amazon Product Recommender")

product_list = df['product_name'].tolist()
selected_product = st.selectbox("Select a product to get recommendations:", product_list)

if st.button('Recommend'):
    index = df[df['product_name'] == selected_product].index[0]
    results = recommend(index)
    st.write(results)
