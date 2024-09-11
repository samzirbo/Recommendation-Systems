from .base import RecSys
from surprise import SVD, SVDpp, Reader, Dataset as SurpriseDataset
import mlflow
from typing import Tuple
from torch.utils.data import Dataset

class SVDRecSys(RecSys):
    """
    A recommendation system class using Singular Value Decomposition (SVD) from the Surprise library.

    This class supports both SVD and SVD++ algorithms for collaborative filtering, providing an implementation
    that integrates with MLflow for experiment tracking and model evaluation.

    Args:
        n_components: Number of latent factors (dimensions) for matrix factorization.
        n_epochs: Number of epochs for training the SVD model.
        plus: If True, use SVD++ algorithm, otherwise use SVD.
        rating_scale: Tuple specifying the minimum and maximum values for the rating scale.
        random_state: Random seed for reproducibility.

    Methods:
        - fit(user_data, item_data, train_data): Trains the SVD or SVD++ model using the training data.
        - predict(user_id, item_id): Predicts the rating or target value for a given user-item pair.
        - run_experiment(user_data, item_data, train_data, test_data): Runs an MLflow experiment and logs parameters and evaluation metrics.
    """
    def __init__(self, n_components: int, n_epochs: int, plus: bool = False, rating_scale: Tuple[int] = (1, 5), random_state: int = 42):
        super().__init__()
        # Initialize Surprise's Reader to interpret the rating scale
        self.reader = Reader(rating_scale=rating_scale)

        # Choose SVD or SVD++ based on the 'plus' parameter
        if plus:
            self.svd = SVDpp(n_factors=n_components, n_epochs=n_epochs, random_state=random_state)
        else:
            self.svd = SVD(n_factors=n_components, n_epochs=n_epochs, random_state=random_state)


    def fit(self, user_data: Dataset, item_data: Dataset, train_data: Dataset):
        """
        Trains the SVD or SVD++ model using the user-item interaction data.

        The training data is converted into a format compatible with Surprise's SVD algorithms.

        Args:
            user_data: Dataset containing user-related features (not directly used by SVD).
            item_data: Dataset containing item/product-related features (not directly used by SVD).
            train_data: Dataset containing user-item interaction data used for training.
        """
        # Extract interaction data and format it for Surprise's SVD
        train_data = train_data.data
        train_data = SurpriseDataset.load_from_df(
            train_data[['user_id', 'product_id', self.target]],
            self.reader
        ).build_full_trainset()

        # Fit the SVD model on the training dataset
        self.svd.fit(train_data)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts the target value (e.g., rating) for a given user-item pair using the trained SVD model.

        Args:
            user_id: ID of the user for whom to predict the target value.
            item_id: ID of the item for which to predict the target value.

        Returns:
            float: The predicted target value (e.g., rating) for the user-item pair.
        """
        return self.svd.estimate(user_id, item_id)
    
    def run_experiment(self, user_data: Dataset, item_data: Dataset, train_data: Dataset, test_data: Dataset) -> dict:
        """
        Runs an experiment with MLflow to track model parameters and evaluation metrics.

        This method trains the SVD model, evaluates it on the test set, and logs the results to MLflow.

        Args:
            user_data: Dataset containing user-related features.
            item_data: Dataset containing item/product-related features.
            train_data: Dataset containing training interaction data.
            test_data: Dataset containing testing interaction data.

        Returns:
            dict: A dictionary of evaluation metrics ('MSE', 'RMSE', 'MAE').
        """
        with mlflow.start_run():
            mlflow.log_params({
                'n_components': self.svd.n_factors,
                'n_epochs': self.svd.n_epochs,
                'plus': isinstance(self.svd, SVDpp)
            })

            self.fit(user_data, item_data, train_data)
            metrics = self.evaluate(test_data)

            mlflow.log_metrics(metrics)
        return metrics
