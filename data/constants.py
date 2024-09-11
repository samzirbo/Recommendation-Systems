import numpy as np
from .generate import cities as CITIES, brands as BRANDS


class RecSysConstants:
    """
    Class to store constants and mappings for a recommendation system.
    """

    GENDER_MAP = {
        'M': 0,
        'F': 1
    }

    CITY_MAP = {city: i for i, city in enumerate(CITIES)}

    CATEGORIES = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports', 'Toys', 'Food', 'Automotive', 'Health']
    CATEGORY_MAP = {category: i for i, category in enumerate(CATEGORIES)}

    BRAND_MAP = {brand: i for i, brand in enumerate(BRANDS)}

    USER_FILE = 'data/source/user_data.csv'
    PRODUCT_FILE = 'data/source/product_data.csv'
    BEHAVIOR_FILE = 'data/source/user_behavior_data.csv'
    RATINGS_FILE = 'data/source/user_ratings.csv'
