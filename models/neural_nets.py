from torch import nn
import torch
from torch.utils.data import Dataset
from .base import NeuralNetRecSys
import mlflow
from torch import Tensor


class MLPRecSys(NeuralNetRecSys):
    """
    A Multi-Layer Perceptron (MLP) based recommendation system.

    This model uses user and item embeddings, followed by fully connected layers to predict the target value (e.g., rating).
    The embeddings of users and items are concatenated and passed through multiple fully connected layers to predict the final output.

    Args:
        n_users: Number of unique users in the dataset.
        n_items: Number of unique items in the dataset.
        embedding_dim: Dimensionality of the embedding vectors for users and items.
        hidden_size: Number of units in the hidden layers.
        n_layers: Number of fully connected layers.
        dropout: Dropout rate for regularization.

    Methods:
        - forward(user_ids, item_ids): Defines the forward pass of the MLP, combining user and item embeddings and passing them through fully connected layers.
        - run_experiment(user_data, item_data, train_data, test_data): Runs an MLflow experiment, logs parameters and metrics, and trains/evaluates the model.
    """
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, hidden_size: int, n_layers: int, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        for i in range(n_layers-1, -1, -1):
            self.fc_layers.append(nn.Linear(input_size, hidden_size * 2 ** i))
            input_size = hidden_size * 2 ** i     
        
        # Activation and regularization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Forward pass of the MLP recommendation model.

        Args:
            user_ids: Tensor of user IDs.
            item_ids: Tensor of item IDs.

        Returns:
            Tensor: Predicted target values (e.g., ratings).
        """
        # Get embeddings
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        
        # Pass through the fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.head(x)
        x = torch.sigmoid(x) * 4 + 1
        return x
    
    def run_experiment(self, user_data: Dataset, item_data: Dataset, train_data: Dataset, test_data: Dataset) -> dict:
        """
        Runs an experiment with MLflow to track model parameters and evaluation metrics.

        Args:
            user_data: User dataset.
            item_data: Item dataset.
            train_data: Training dataset.
            test_data: Testing dataset.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'embedding_dim': self.user_embedding.embedding_dim,
                'hidden_size': self.fc_layers[-1].out_features,
                'n_layers': len(self.fc_layers),
                'hidden_units': [layer.out_features for layer in self.fc_layers],
                'dropout': self.dropout.p,
                'lr': self.lr,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'weight_decay': self.weight_decay
            })

            # Train the model
            self.fit(user_data, item_data, train_data)

            # Evaluate the model and log metrics
            metrics = self.evaluate(test_data)
            mlflow.log_metrics(metrics)

        return metrics
    

class NeuralCFRecSys(NeuralNetRecSys):
    """
    A Neural Collaborative Filtering (NeuralCF) recommendation system combining Generalized Matrix Factorization (GMF)
    and Multi-Layer Perceptron (MLP) for predicting user-item interactions, such as ratings.

    The model merges two approaches: GMF for element-wise multiplication of embeddings (similar to traditional matrix
    factorization) and MLP for learning nonlinear interactions between user and item embeddings.

    Args:
        n_users: Number of unique users in the dataset.
        n_items: Number of unique items in the dataset.
        embedding_dim: Dimensionality of the embedding vectors for users and items.
        hidden_size: Number of units in the hidden layers of the MLP.
        n_layers: Number of fully connected layers in the MLP.
        dropout: Dropout rate for regularization.

    Methods:
        - forward(user_ids, item_ids): Defines the forward pass of the NeuralCF, combining GMF and MLP.
        - run_experiment(user_data, item_data, train_data, test_data): Runs an MLflow experiment, logging parameters and metrics, and training/evaluating the model.
    """
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, hidden_size: int, n_layers: int, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        # Embedding layers for users and items
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        for i in range(n_layers-1, -1, -1):
            self.fc_layers.append(nn.Linear(input_size, hidden_size * 2 ** i))
            input_size = hidden_size * 2 ** i     

        # Activation and regularization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + embedding_dim, 1)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Forward pass of the Neural Collaborative Filtering (NeuralCF) model.

        Args:
            user_ids: Tensor of user IDs.
            item_ids: Tensor of item IDs.

        Returns:
            Tensor: Predicted target values (e.g., ratings).
        """
        # Get embeddings
        user_embedding_mlp = self.user_embedding_mlp(user_ids)
        item_embedding_mlp = self.item_embedding_mlp(item_ids)

        user_embedding_gmf = self.user_embedding_gmf(user_ids)
        item_embedding_gmf = self.item_embedding_gmf(item_ids)
        
        # Concatenate user and item embeddings
        x_mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        
        # Pass through the fully connected layers
        for fc in self.fc_layers:
            x_mlp = fc(x_mlp)
            x_mlp = self.activation(x_mlp)
            x_mlp = self.dropout(x_mlp)

        x_gmf = user_embedding_gmf * item_embedding_gmf

        x = torch.cat([x_mlp, x_gmf], dim=-1)
        
        # Output layer
        x = self.head(x)
        x = torch.sigmoid(x) * 4 + 1
        return x
    
    def run_experiment(self, user_data: Dataset, item_data: Dataset, train_data: Dataset, test_data: Dataset) -> dict:
        """
        Runs an experiment using MLflow to track model parameters and evaluation metrics.

        Args:
            user_data: User dataset.
            item_data: Item dataset.
            train_data: Training dataset.
            test_data: Testing dataset.

        Returns:
            dict: A dictionary of evaluation metrics (e.g., MSE, RMSE, MAE).
        """
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'embedding_dim': self.user_embedding_mlp.embedding_dim,
                'hidden_size': self.fc_layers[-1].out_features,
                'n_layers': len(self.fc_layers),
                'hidden_units': [layer.out_features for layer in self.fc_layers],
                'dropout': self.dropout.p,
                'lr': self.lr,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'weight_decay': self.weight_decay
            })

            # Train the model
            self.fit(user_data, item_data, train_data)

            # Evaluate the model and log metrics
            metrics = self.evaluate(test_data)
            mlflow.log_metrics(metrics)

        return metrics