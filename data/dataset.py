from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from .constants import RecSysConstants


class UserDataset(Dataset):
    """
    A dataset class that processes and manages user data for a recommendation system.

    This class reads user data from a CSV file, computes additional features such as tenure (days since
    joining), and supports extensions with interaction data. 

    Args:
        csv_file: Path to the CSV file containing user data.

    Features:
        - user_id: Unique identifier for the user
        - age: Age of the user
        - gender: Gender of the user
        - location: City of the user
        + tenure: Days since joining
        + purchases: Total number of purchases
        + views: Total number of views
        + view_purchase_ratio: Ratio of purchases to views
        + view_to_purchase_time: Average time from view to purchase
        + avg_rating: Average rating given by the user
    """
    def __init__(self, csv_file: Path):
        # Load user data from CSV
        self.data = pd.read_csv(csv_file)

        # Convert 'join_date' to datetime and compute 'tenure' in days
        self.data['join_date'] = pd.to_datetime(self.data['join_date'])
        self.data['tenure'] = (pd.to_datetime('today') - self.data['join_date']).dt.days

        # Map categorical fields to numerical values using predefined mappings
        self.data['gender'] = self.data['gender'].map(RecSysConstants.GENDER_MAP)
        self.data['location'] = self.data['location'].map(RecSysConstants.CITY_MAP)

        # Drop the 'join_date' column as it is no longer needed
        self.data.drop(columns=['join_date'], inplace=True)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, user_id: int) -> dict:
        return self.data[self.data['user_id'] == user_id].to_dict()
    
    def extend(self, interaction_ds: Dataset):
        """
        Extends the user dataset by merging it with user interaction data.

        This method aggregates interaction data such as purchases, views, ratings, and computes additional
        features like view-to-purchase ratio and average rating. It also maps categorical data like gender 
        and location to numeric values using RecSysConstants.

        Args:
            interaction_ds: A dataset containing user interaction data.
        """
        # Extract interaction data from the provided interaction dataset
        interaction_data = interaction_ds.data

        # Aggregate interaction data by 'user_id' and add the calculated fields to the user data
        self.data['purchases'] = interaction_data.groupby('user_id')['purchases'].sum()
        self.data['views'] = interaction_data.groupby('user_id')['views'].sum()
        self.data['view_purchase_ratio'] = self.data['purchases'] / self.data['views']
        self.data['view_to_purchase_time'] = interaction_data.groupby('user_id')['view_to_purchase_time'].mean()
        self.data['avg_rating'] = interaction_data.groupby('user_id')['rating'].mean()

        # Fill any missing values with 0 to avoid NaNs in the dataset
        self.data.fillna(0, inplace=True)
    
    def save(self, filename: str):
        """
        Saves the processed user dataset to a CSV file.

        Args:
            filename: The name of the file to save the dataset to.
        """
        self.data.to_csv(filename)


class ProductDataset(Dataset):
    """
    A dataset class that processes and manages product data for a recommendation system.

    This class reads product data from a CSV file and supports extensions with interaction data to calculate
    additional product-level features such as total views, purchases, and average ratings.

    Args:
        csv_file: Path to the CSV file containing product data.

    Features:
        - product_id: Unique identifier for the product
        - name: Name of the product
        - category: Category of the product (e.g., Electronics, Clothing)
        - brand: Brand of the product
        + purchases: Total number of purchases for the product
        + views: Total number of views for the product
        + avg_rating: Average user rating of the product
        + avg_view_to_purchase_time: Average time taken from product view to purchase
    """
    def __init__(self, csv_file: Path):
        # Load product data from CSV file
        self.data = pd.read_csv(csv_file)

        # Map categorical fields to numerical values using predefined mappings
        self.data['category'] = self.data['category'].map(RecSysConstants.CATEGORY_MAP)
        self.data['brand'] = self.data['brand'].map(RecSysConstants.BRAND_MAP)


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, product_id: int) -> dict:
        return self.data[self.data['product_id'] == product_id].to_dict()
    
    def extend(self, interaction_ds: Dataset):
        """
        Extends the product dataset by merging it with product interaction data.

        This method aggregates interaction data such as purchases, views, and ratings, and computes additional
        features like average view-to-purchase time and average rating. 

        Args:
            interaction_ds: A dataset containing user interaction data.
        """
        # Extract interaction data from the provided interaction dataset
        interaction_data = interaction_ds.data

        # Aggregate interaction data by 'product_id' and add the calculated fields to the product data
        self.data['purchases'] = interaction_data.groupby('product_id')['purchases'].sum()
        self.data['views'] = interaction_data.groupby('product_id')['views'].sum()
        self.data['avg_rating'] = interaction_data.groupby('product_id')['rating'].mean()
        self.data['avg_view_to_purchase_time'] = interaction_data.groupby('product_id')['view_to_purchase_time'].mean()

        # Fill any missing values with 0 to avoid NaNs in the dataset
        self.data.fillna(0, inplace=True)
    
    def save(self, filename: str):
        """
        Saves the processed product dataset to a CSV file.

        Args:
            filename: The name of the file to save the dataset to.
        """
        self.data.to_csv(filename)
    

class UserInteractionDataset(Dataset):
    """
    A dataset class that processes and manages user interaction data, including product views, purchases,
    and ratings, for a recommendation system.

    This class reads interaction data from behavior and ratings CSV files, computes additional features such
    as the time between viewing and purchasing a product, and supports splitting the data into training and
    testing sets.

    Args:
        behavior_file: Path to the CSV file containing user behavior data (views and purchases).
        ratings_file: Path to the CSV file containing user ratings data.
        test_split: Fraction of the dataset to be used for testing (default: 0.2).
        subset: Can be either 'train' or 'test', to work with a subset of the data after splitting.

    Features:
        - user_id: Unique identifier for the user.
        - product_id: Unique identifier for the product.
        - rating: Rating given by the user to the product.
        + views: Total number of views by the user for the product.
        + purchases: Total number of purchases by the user for the product.
        + view_to_purchase_time: Average time (in hours) between viewing and purchasing the product.
    """
    def __init__(self, behavior_file: Path, ratings_file: Path, test_split: int = 0.2, subset=None):
        self.subset = subset

        # Load user behavior data (views and purchases) from CSV file
        self.behavior = pd.read_csv(behavior_file)

        # Convert 'view_timestamp' and 'purchase_timestamp' to datetime format
        self.behavior['view_timestamp'] = pd.to_datetime(self.behavior['view_timestamp'])
        self.behavior['purchase_timestamp'] = pd.to_datetime(self.behavior['purchase_timestamp'])

        # Calculate the time taken from viewing to purchasing a product in hours
        self.behavior['view_to_purchase_time'] = (self.behavior['purchase_timestamp'] - self.behavior['view_timestamp']).dt.total_seconds() // 3600  # in hours

        # Aggregate the counts of views and purchases as the pair of user_id and product_id is not unique
        self.behavior = self.behavior.groupby(['user_id', 'product_id']).agg({
            'view_timestamp': ['count'],
            'purchase_timestamp': ['count', 'max'],
            'view_to_purchase_time': ['mean'],
        }).reset_index()
        self.behavior.columns = ['_'.join(col).strip() if col[1] else col[0] for col in self.behavior.columns.values]
        self.behavior.rename(columns={
            'view_timestamp_count': 'views',
            'purchase_timestamp_count': 'purchases',
            'purchase_timestamp_max': 'purchase_timestamp',
            'view_to_purchase_time_mean': 'view_to_purchase_time'
        }, inplace=True)

        # Load user ratings data from CSV file
        self.ratings = pd.read_csv(ratings_file)

        # Convert 'timestamp' to datetime format
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'])
        self.ratings.rename(columns={'timestamp': 'rating_timestamp'}, inplace=True)

        # Merge the behavior data with the ratings data
        self.data = pd.merge(self.ratings, self.behavior, on=['user_id', 'product_id'], how='outer')

        # Drop unnecessary columns
        self.data.drop(columns=['rating_timestamp', 'purchase_timestamp'], inplace=True)

        # Fill missing values with 0 to avoid NaNs in the dataset
        self.data[['views', 'purchases', 'view_to_purchase_time']] = self.data[['views', 'purchases', 'view_to_purchase_time']].fillna(0)

        # Split the data into training and testing sets
        if subset:
            self.data = self.data.dropna(subset=['rating'])
            self.train, self.test = train_test_split(self.data, test_size=test_split, shuffle=True, random_state=42, stratify=self.data['rating'])
            
            self.data = self.train if subset == 'train' else self.test
            self.data.reset_index(drop=True, inplace=True)


    def __len__(self) -> int:
        return len(self.data) 
    
    def __getitem__(self, idx) -> dict:
        return self.data.iloc[idx].to_dict()

    def save(self, filename: str):
        """
        Saves the processed user interaction dataset to a CSV file.
        """
        self.data.to_csv(filename, index=False)
