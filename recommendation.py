import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('data/u.data', sep='\t', names=columns)
    print("\n Loaded Data:")
    print(df.head())
    print("Shape:", df.shape)
    return df

def create_user_item_matrix(df):
    matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    print("\n User-Item Matrix:")
    print(matrix.head())
    print("Shape:", matrix.shape)
    return matrix

def load_movie_titles():
    item_cols = ['item_id', 'title']
    movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=item_cols, header=None)
    return movies.set_index('item_id')['title']




def get_similar_users(user_item_matrix, user_id):
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    return similarity_df[user_id].sort_values(ascending=False)

def recommend_movies(df, user_id, top_n=5):
    user_item_matrix = create_user_item_matrix(df)
    
    # Calculate similarity
    similar_users = get_similar_users(user_item_matrix, user_id)
    similar_users = similar_users.drop(user_id)  # remove self

    # Filter user_item_matrix to only include similar users
    similar_user_ratings = user_item_matrix.loc[similar_users.index]

    # Multiply each user's ratings by their similarity score
    weighted_ratings = similar_user_ratings.T.dot(similar_users)

    # Normalize by total similarity score
    recommendation_scores = weighted_ratings / similar_users.sum()

    # Get movies the current user hasn't rated yet
    user_rated = user_item_matrix.loc[user_id]
    unrated_movies = user_rated[user_rated == 0].index

    # Return top N recommended movie IDs
    recommendations = recommendation_scores.loc[unrated_movies]
    top_recommendations = recommendations.sort_values(ascending=False).head(top_n)

# Map movie IDs to titles
    movie_titles = load_movie_titles()
    top_recommendations.index.name = 'item_id'
    top_recommendations = top_recommendations.rename('score').to_frame()
    top_recommendations['title'] = top_recommendations.index.map(movie_titles)

    return top_recommendations[['title', 'score']]

