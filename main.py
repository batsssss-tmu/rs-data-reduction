from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer

from codecarbon import EmissionsTracker

dataset_name = 'amazon_books_60core_train_65_newest_ratings_each_user'

config_params = {
    'dataset': dataset_name,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'TIME_FIELD': 'timestamp',
    'load_col': {
        'inter': [
            'user_id',
            'item_id',
            'rating',
            'timestamp',
        ]
    },
    'reproducibility': True,
    'save_dataset': True,
    'save_dataloaders': True,
    'threshold': {
        'rating': 3,
    },
    'repeatable': True
}
