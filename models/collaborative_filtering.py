from .base import RecSys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
from torch.utils.data import Dataset

class CollaborativeFilteringRecSys(RecSys):
    """
    A recommendation system class implementing collaborative filtering (user-based or item-based) with optional
    content-based filtering using cosine similarity.

    Args:
        method: 'user' for user-based CF, 'item' for item-based CF.
        k: Number of nearest neighbors (users/items) to consider for making predictions.
        content_based: If True, content-based filtering is applied instead of collaborative filtering.

    Methods:
        - fit(user_data, item_data, train_data): Prepares the user-item interaction matrix and computes similarity matrix.
        - predict(user_id, item_id): Predicts the rating or target value for a specific user-item pair using collaborative filtering.
        - run_experiment(user_data, item_data, train_data, test_data): Runs an MLflow experiment and logs parameters and evaluation metrics.
    """
    def __init__(self, method: str = 'user', k: int = 5, content_based: bool = False):
        super().__init__()
        self.method = method
        self.k = k
        self.content_based = content_based

    def fit(self, user_data: Dataset, item_data: Dataset, train_data: Dataset):
        """
        Fits the collaborative filtering model by creating the user-item interaction matrix and computing the
        similarity matrix based on users or items.

        Args:
            user_data: Dataset containing user-related features.
            item_data: Dataset containing item/product-related features.
            train_data: Dataset containing user-item interaction data used for training.
        """
        train_data = train_data.data

        # Create global user-item matrix (rows: users, columns: items, values: target)
        unique_users = user_data.data['user_id'].unique()
        unique_items = item_data.data['product_id'].unique()
        self.user_item_matrix = pd.DataFrame(np.nan, index=unique_users, columns=unique_items)

        # Fill the user-item matrix with target values (e.g., ratings)
        self.user_item_matrix.update(
            train_data.pivot_table(
                index='user_id',
                columns='product_id',
                values=self.target
            )
        )

        # Fill missing values with 0
        self.user_item_matrix.fillna(0, inplace=True)


        # Calculate similarity matrix
        if self.content_based:

            # Content-based similarity: Compute similarity between users or items based on their features
            user_data = user_data.data.set_index('user_id')
            item_data = item_data.data.set_index('product_id')


            self.similarity_matrix = cosine_similarity(user_data if self.method == 'user' else item_data)
        else:
            
            # Collaborative filtering similarity: Compute similarity between users or items based on their interactions
            self.similarity_matrix = cosine_similarity(self.user_item_matrix if self.method == 'user' else self.user_item_matrix.T)            
        
        # Store the similarity matrix as a DataFrame for easy access
        self.similarity_matrix = pd.DataFrame(
            self.similarity_matrix, 
            index=self.user_item_matrix.index if self.method == 'user' else self.user_item_matrix.columns, 
            columns=self.user_item_matrix.index if self.method == 'user' else self.user_item_matrix.columns
        )
        

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts the target value (e.g., rating) for a given user-item pair.

        This method uses collaborative filtering to find similar users or items and predicts the target value
        by aggregating their interactions.

        Args:
            user_id: ID of the user for whom to predict the target value.
            item_id: ID of the item for which to predict the target value.

        Returns:
            float: The predicted target value (e.g., rating).
        """
        if self.method == 'user':
            # Step 1: Get the similarity vector for the target user
            similarity_vector = self.similarity_matrix.loc[user_id].copy()
            similarity_vector[user_id] = 0  # Set self-similarity to 0

            # Step 2: Find the top-k most similar users
            similar_users = similarity_vector.nlargest(self.k).index

            # Step 3: Filter out users who have not rated the item
            valid_users = self.user_item_matrix[item_id] != 0
            neighbors = similar_users[similar_users.isin(valid_users[valid_users].index)]
            neighbor_values = self.user_item_matrix.loc[neighbors, item_id]


        elif self.method == 'item':
            # Step 1: Get the similarity vector for the target item
            similarity_vector = self.similarity_matrix.loc[item_id].copy()
            similarity_vector[item_id] = 0  # Set self-similarity to 0

            # Step 2: Find the top-k most similar items
            similar_items = similarity_vector.nlargest(self.k).index

            # Step 3: Filter out items not rated by the user
            valid_items = self.user_item_matrix.loc[user_id] != 0
            neighbors = similar_items[similar_items.isin(valid_items[valid_items].index)]
            neighbor_values = self.user_item_matrix.loc[user_id, neighbors]
        
        else:
            raise ValueError("Invalid method. Choose 'user' or 'item'.")

        
        # Step 4: Compute the weighted sum of ratings if similar users exist
        if len(neighbors) > 0:
            similarities = similarity_vector[neighbors]
            weighted_sum = (neighbor_values * similarities).sum()
            total_weight = similarities.sum()

            pred = weighted_sum / total_weight if total_weight > 0 else 0

        else:
            # If no similar users or items are found, return the mean rating of the target item
            pred = self.user_item_matrix[item_id].mean()  

        return pred   

    def run_experiment(self, user_data: Dataset, item_data: Dataset, train_data: Dataset, test_data: Dataset) -> dict:
        """
        Runs an experiment with MLflow to track model parameters and evaluation metrics.

        This method trains the model, evaluates it on the test set, and logs the results to MLflow.

        Args:
            user_data: Dataset containing user-related features.
            item_data: Dataset containing item/product-related features.
            train_data: Dataset containing training interaction data.
            test_data: Dataset containing testing interaction data.

        Returns:
            dict: A dictionary of evaluation metrics ('MSE', 'RMSE', 'MAE').
        """
        with mlflow.start_run():
            # Log model parameters to MLflow
            mlflow.log_params({
                'method': self.method,
                'k': self.k,
                'content_based': self.content_based
            })

            # Train the model
            self.fit(user_data, item_data, train_data)

            # Evaluate the model and log metrics
            metrics = self.evaluate(test_data)
            mlflow.log_metrics(metrics)

        return metrics
