from data import UserDataset, ProductDataset, UserInteractionDataset
from models import CollaborativeFilteringRecSys, SVDRecSys, MLPRecSys, NeuralCFRecSys
import json
from pathlib import Path
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import argparse
from data.constants import RecSysConstants
import mlflow
from models.base import NeuralNetRecSys

# Dictionary to map model names to corresponding classes
models = {
    'CollaborativeFilteringRecSys': CollaborativeFilteringRecSys,
    'SVDRecSys': SVDRecSys,
    'MLPRecSys': MLPRecSys,
    'NeuralCFRecSys': NeuralCFRecSys
}


class HyperParameterTuner:
    """
    HyperParameterTuner is responsible for managing the hyperparameter search
    for different recommender system models.

    Args:
        user_file: Path to the file containing user data.
        item_file: Path to the file containing item/product data.
        ratings_file: Path to the file containing user-item rating interactions.
        behavior_file: Path to the file containing user behavioral interactions.
        verbose (bool): Whether to print detailed information about the datasets.
    """
    def __init__(self, user_file: Path, item_file: Path, ratings_file: Path, behavior_file: Path):
        # Load user, item, and interaction datasets
        self.user_data = UserDataset(user_file)
        self.item_data = ProductDataset(item_file)
        self.interaction_data = UserInteractionDataset(ratings_file=ratings_file, behavior_file=behavior_file)

        # Extend user and item datasets with interaction data
        self.user_data.extend(self.interaction_data)
        self.item_data.extend(self.interaction_data)

        # Prepare train/test datasets for interaction data
        self.train_data = UserInteractionDataset(ratings_file=ratings_file, behavior_file=behavior_file, subset='train')
        self.test_data = UserInteractionDataset(ratings_file=ratings_file, behavior_file=behavior_file, subset='test')

    def tune(self, search_space: Path, metric='RMSE', verbose: bool = False):
        """
        Performs hyperparameter tuning using the provided search space.

        Args:
            search_space: Path to the JSON file specifying the hyperparameter search space.
            metric: The evaluation metric to optimize (default is 'RMSE').
            verbose: Whether to print detailed information about the tuning process.
        """
        # Load the hyperparameter search space from a JSON file
        search_space = json.load(open(search_space, 'r'))

        # Iterate over each experiment defined in the search space
        for experiment, setup in search_space.items():
            # Set up MLflow experiment for tracking
            mlflow.set_experiment(experiment)

            # Extract model class from the setup configuration
            model_class = setup.pop('model')
            model_class = models[model_class]

            # Number of iterations for hyperparameter search
            no_iter = setup.pop('no_iter')

            # Define hyperparameter search space
            params = setup.pop('params')
            params = {k: hp.choice(k, v) for k, v in params.items()}

            # Define the objective function to minimize (e.g., RMSE)
            def objective(params):
                # If the model is a neural network-based recommender, add user/item dimensions
                if issubclass(model_class, NeuralNetRecSys):
                    params.update({
                        'n_users': len(self.user_data),
                        'n_items': len(self.item_data),
                    })

                # Instantiate the model with the current parameters
                model = model_class(**params)

                # Run an experiment and return the evaluation metric
                return model.run_experiment(self.user_data, self.item_data, self.train_data, self.test_data)[metric]
            
            trials = Trials()
            best_params = fmin(
                fn=objective,  # The objective function to minimize
                space=params,  # The search space for hyperparameters
                algo=tpe.suggest,  # The algorithm for choosing next parameter set (TPE)
                max_evals=no_iter,  # Maximum number of iterations for search
                trials=trials  # Track experiment trials
            )
            
            # Optionally print the best parameters for the experiment
            if verbose:
                print(f'Best parameters for {experiment}: {best_params}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--search_space', type=str, required=True)
    parser.add_argument('--metric', type=str, default='RMSE')

    parser.add_argument('--user_file', type=str, default=RecSysConstants.USER_FILE)
    parser.add_argument('--item_file', type=str, default=RecSysConstants.PRODUCT_FILE)
    parser.add_argument('--ratings_file', type=str, default=RecSysConstants.RATINGS_FILE)
    parser.add_argument('--behavior_file', type=str, default=RecSysConstants.BEHAVIOR_FILE)

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    tuner = HyperParameterTuner(
        user_file=Path(args.user_file),
        item_file=Path(args.item_file),
        ratings_file=Path(args.ratings_file),
        behavior_file=Path(args.behavior_file),
    )

    tuner.tune(
        search_space=args.search_space,
        metric=args.metric,
        verbose=args.verbose
    )