import streamlit as st
from recommendation import load_data, recommend_movies

st.title("Movie Recommendation System")
df = load_data()

st.markdown("Select a user to get movie recommendations based on collaborative filtering.")

user_ids = sorted(df['user_id'].unique())
selected_user = st.selectbox("Select User ID", user_ids)

if st.button("Recommend"):
    recommendations = recommend_movies(df, selected_user)
    st.write("Top Movie Recommendations (Movie IDs):")
    st.write("Top Movie Recommendations:")
    st.table(recommendations.reset_index(drop=True))

