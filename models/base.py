from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
import torch
import mlflow

class RecSys(ABC):
    """
    An abstract base class for a recommendation system (RecSys) model.

    This class provides a common interface for building recommendation systems with methods for fitting
    the model, making predictions, and evaluating performance on a test dataset. Subclasses should implement
    the `fit` and `predict` methods according to the specific model used.

    Args:
        target: The target field to predict (default is 'rating').

    Methods:
        - fit(user_data, item_data, train_data): Train the recommendation model with user, item, and interaction data.
        - predict(user_id, item_id): Make a prediction for a specific user and item.
        - evaluate(test_data): Evaluate the model's performance on a test dataset using MSE, RMSE, and MAE metrics.
    """
    def __init__(self, target: str ='rating', **kwargs):
        super().__init__()
        self.target = target

    @abstractmethod
    def fit(self, user_data: Dataset, item_data: Dataset, train_data: Dataset):
        """
        Abstract method for fitting the model using user data, item data, and training data.

        Args:
            user_data: Dataset containing user-related features.
            item_data: Dataset containing item/product-related features.
            train_data: Dataset containing user-item interaction data used for training.
        """
        pass

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Abstract method for predicting a target value for a given user and item.

        Args:
            user_id: The ID of the user for whom to make the prediction.
            item_id: The ID of the item for which to make the prediction.

        Returns:
            float: The predicted value (e.g., rating).
        """
        pass

    def evaluate(self, test_data: Dataset) -> dict:
        """
        Evaluates the model on a test dataset by comparing predicted values with actual values.

        This method iterates over the test data, generates predictions for each user-item pair,
        and calculates evaluation metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
        and Mean Absolute Error (MAE).

        Args:
            test_data: Dataset containing user-item interactions with actual target values (e.g., ratings).

        Returns:
            dict: A dictionary of evaluation metrics with rounded values for 'MSE', 'RMSE', and 'MAE'.
        """
        # Extract data from the test dataset
        test_data = test_data.data
        predictions = []
        actuals = []

        # Iterate through the test dataset
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            actual_value = row[self.target]

            # Get the predicted value from the model
            predicted_value = self.predict(user_id, item_id)

            # Append the predicted and actual values to lists
            predictions.append(predicted_value)
            actuals.append(actual_value)

        # Calculate evaluation metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)

        # Compile metrics dictionary with rounded values
        metrics =  {
            'MSE': mse,
            'RMSE': mse ** 0.5,
            'MAE': mae
        }
        return {name: round(value, 2) for name, value in metrics.items()}
    

class NeuralNetRecSys(nn.Module, RecSys):
    """
    A neural network-based recommendation system (RecSys) model.

    This class inherits from both `nn.Module` (for building neural networks) and `RecSys` (for recommendation systems),
    providing an implementation of a neural network recommender with customizable learning rate, number of epochs,
    batch size, and weight decay for regularization. It defines abstract methods for the network forward pass
    and implements training (fit) and prediction functionality.

    Args:
        lr: Learning rate for the optimizer (default: 0.001).
        n_epochs: Number of training epochs (default: 10).
        batch_size: Batch size for training (default: 32).
        weight_decay: L2 regularization parameter (default: 1e-5).

    Methods:
        - forward(x): Abstract method for the forward pass of the neural network.
        - fit(user_data, item_data, train_data): Trains the model using user, item, and interaction data.
        - predict(user_id, item_id): Makes a prediction for a specific user-item pair.
        - evaluate(test_data): Evaluates the model on a test dataset.
    """
    def __init__(self, lr=0.001, n_epochs=10, batch_size=32, weight_decay=1e-5, **kwargs):
        RecSys.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def fit(self, user_data: Dataset, item_data: Dataset, train_data: Dataset):
        """
        Trains the neural network using the training dataset.

        This method defines the training loop, where batches of user and item data are processed, loss is computed,
        and the model parameters are updated via backpropagation. The loss function used is Mean Squared Error (MSE).

        Args:
            user_data: Dataset containing user-related features (not used directly in this function).
            item_data: Dataset containing item/product-related features (not used directly in this function).
            train_data: Dataset containing user-item interaction data used for training.
        """
        # Set the model to training mode
        self.train()

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # DataLoader for training data
        dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for _ in range(self.n_epochs):
            running_loss = 0
            for batch in dl:
                user_ids = batch['user_id'].long() - 1001 # Subtract 1001 to start user_id from 0
                item_ids = batch['product_id'].long() - 2001 
                target = batch[self.target].float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.forward(user_ids, item_ids)

                # Compute loss
                loss = criterion(output, target)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                running_loss += loss.item()


    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts the target value (e.g., rating) for a given user-item pair.

        This method uses the trained neural network to make a prediction for a specific user and item.

        Args:
            user_id: The ID of the user for whom to make the prediction.
            item_id: The ID of the item for which to make the prediction.

        Returns:
            float: The predicted value (e.g., rating) for the user-item pair.
        """
        # Convert user_id and item_id to tensors and adjust indices
        user_id = torch.tensor([user_id], dtype=torch.long) - 1001
        item_id = torch.tensor([item_id], dtype=torch.long) - 2001
        return self.forward(user_id, item_id).item()

    def evaluate(self, test_data: Dataset) -> dict:
        """
        Evaluates the model on a test dataset by computing metrics like MSE, RMSE, and MAE.

        The method uses the `evaluate` method from the `RecSys` parent class to calculate the performance metrics
        and returns them. The model is temporarily set to evaluation mode during this process.

        Args:
            test_data: The dataset containing user-item interaction data used for testing.

        Returns:
            dict: A dictionary of evaluation metrics ('MSE', 'RMSE', 'MAE') with rounded values.
        """
        # Set the model to evaluation mode
        self.eval()

        # Call the parent class evaluation method to calculate the metrics
        metrics = super().evaluate(test_data)

        # Set the model back to training mode
        self.train()
        
        return metrics