import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    'restaurant': ['A', 'B', 'C', 'A', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'B'],
    'entree': ['Pasta', 'Pizza', 'Burger', 'Pasta', 'Burger', 'Pizza', 'Pasta', 'Burger', 'Pizza', 'Pasta', 'Burger', 'Pizza'],
    'rating': [5, 4, 3, 5, 4, 4, 5, 3, 4, 5, 4, 5]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a pivot table
pivot_table = df.pivot_table(index='user_id', columns='restaurant', values='rating').fillna(0)
print("\nPivot Table:")
print(pivot_table)

# Convert the data into a format Surprise can use
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'restaurant', 'rating']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Test the algorithm on the testset
predictions = algo.test(testset)

# Compute and print RMSE
print("\nRMSE:")
accuracy.rmse(predictions)

def get_top_n_recommendations(algo, user_id, n=3):
    # Get a list of all restaurants
    all_restaurants = df['restaurant'].unique()
    
    # Get the restaurants the user has already rated
    rated_restaurants = df[df['user_id'] == user_id]['restaurant'].unique()
    
    # Get the list of restaurants the user hasn't rated
    unrated_restaurants = [restaurant for restaurant in all_restaurants if restaurant not in rated_restaurants]
    
    # Predict ratings for the unrated restaurants
    predictions = [algo.predict(user_id, restaurant) for restaurant in unrated_restaurants]
    
    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top n recommendations
    top_n_recommendations = predictions[:n]
    
    return [(pred.iid, pred.est) for pred in top_n_recommendations]

# Get recommendations for user 1
recommendations = get_top_n_recommendations(algo, user_id=1, n=3)
print("\nTop recommendations for user 1:")
for restaurant, rating in recommendations:
    print(f"Restaurant: {restaurant}, Estimated Rating: {rating:.2f}")
