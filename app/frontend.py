import streamlit as st
import pandas as pd
import backend  # This imports your backend.py functions

# Load user profile data (same as backend)
profiles_df = backend.profiles_df

# Streamlit UI
st.set_page_config(page_title="Friend Recommender", layout="centered")
st.title("🤝 Friend Recommendation System")

# Sidebar: User Selection
st.sidebar.header("Select User")
user_ids = profiles_df['user_id'].unique()
selected_user_id = st.sidebar.selectbox("Choose a User ID", user_ids)

# Main Area
st.write(f"### Profile for User {selected_user_id}")
user_profile = profiles_df[profiles_df['user_id'] == selected_user_id]
st.dataframe(user_profile)

# Button to trigger recommendations
if st.button("Find Friends"):
    recommendations = backend.get_recommendations(selected_user_id)
    
    if not recommendations.empty:
        st.write("### 🔗 Top Friend Recommendations")
        st.dataframe(recommendations[['user_id', 'final_score'] + 
                                     [col for col in recommendations.columns if col not in ['user_id', 'final_score']]])
    else:
        st.warning("No recommendations found.")
