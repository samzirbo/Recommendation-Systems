# Recommendation System Project

## Overview
This project is a solution to a Data Science assignment focused on building a recommendation system. The goal of the project is to predict user preferences based on past interactions and provide personalized product recommendations. This system utilizes various recommendation models, including collaborative filtering, matrix factorization (SVD), and neural network-based approaches. Additionally, a hyperparameter tuning module is included to optimize model performance.

---

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Model](#model)
- [Hyperparameter Tuner](#hyperparameter-tuner)

---

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/samzirbo/Recommendation-Systems
   cd Recommendation-Systems
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
    or
   ```bash
   conda activate venv
   ```


3. Install the required dependencies:
   ```bash
   pip install poetry
   poetry install
   ```

4. Run the experiments:
   ```bash
   python experiment.py --search_space data/search_space.json --metric RMSE
   ```

This command will train the model using the provided search space and optimize based on the RMSE metric.

---

## Data
### Datasets
The project uses three primary datasets:
- **User Dataset**: Contains user profiles and attributes.
- **Product Dataset**: Includes details about the products or items to recommend.
- **User-Interaction Dataset**: Contains user-item interactions, such as ratings or behavioral data (e.g., clicks, purchases).

The interaction data is split into training and test sets to evaluate the performance of the recommendation models.

### Data Format
- **User Dataset (`user_data.csv`)**: Includes features like `user_id`, `age`, `gender`, etc.
- **Product Dataset (`item_data.csv`)**: Contains product-related features like `item_id`, `category`, `price`, etc.
- **User-Interaction Dataset (`ratings.csv`)**: Includes `user_id`, `item_id`, and `rating`, as well as behavioral data in `behavior.csv`.

---

## Model
### Models Implemented
This project implements multiple recommendation models, each targeting different aspects of recommendation systems:

- **Collaborative Filtering (CollaborativeFilteringRecSys)**: A model that uses user-item interactions to identify similar users or items.
- **Singular Value Decomposition (SVDRecSys)**: Matrix factorization technique that decomposes the user-item interaction matrix to predict missing values.
- **Multi-Layer Perceptron (MLPRecSys)**: A deep learning model that uses user-item interactions as input and learns latent representations of users and items.
- **Neural Collaborative Filtering (NeuralCFRecSys)**: A neural network-based collaborative filtering approach that merges matrix factorization and deep learning for more accurate recommendations.

### Model Evaluation
The models are evaluated on common recommendation metrics, including:
- **RMSE (Root Mean Squared Error)**: Measures how close predicted ratings are to actual ratings.
- **MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual ratings.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual ratings.

Each model can be customized with hyperparameters such as latent factor size, learning rate, and regularization, as shown in the `search_space.json` configuration file.

---

## Hyperparameter Tuner
### Overview
The hyperparameter tuner uses **Hyperopt** to optimize model parameters. It automatically adjusts the search space to find the best hyperparameters for each recommendation model based on the specified metric (e.g., RMSE).

### How It Works
1. **Search Space Definition**: Hyperparameter search space is defined in a JSON file (e.g., `data/search_space.json`). This includes the model class and ranges for each hyperparameter.
   
   Example snippet from `search_space.json`:
   ```json
   {
       "Multi-Layer Perceptron": {
            "model": "MLPRecSys",
            "no_iter": 100,
            "params": {
                "embedding_dim": [10, 50, 100],
                "hidden_size": [8, 16, 32],
                "n_layers": [1, 2, 3],
                "dropout": [0.2, 0.4, 0.6],
                "n_epochs": [5, 10, 20, 50, 100],
                "batch_size": [8, 16, 32, 64],
                "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "weight_decay": [0.000001, 0.00001, 0.0001, 0.001]
            }
        }
   }
   ```

2. **Hyperopt Search**: The tuner uses Hyperopt's `fmin` function to explore the hyperparameter space efficiently.
   
3. **Results**: The experiments, along with hyperparameters and metrics, are logged using **MLflow** for tracking and analysis.

---

## Experiment Tracking and Visualization

The project uses **MLflow** to log and visualize the results of the experiments. You can view the results by running the MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5001
```

This command will start the MLflow server, and you can access the UI by navigating to `http://0.0.0.0:5001` in your browser.

---