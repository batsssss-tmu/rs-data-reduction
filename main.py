from evaluation import model_tests, create_result_file
from codecarbon import EmissionsTracker

if __name__ == "__main__":

    """
    Sets RecBole framework configuration and runs model tests.
    Tests are wrapped by CodeCarbon power usage tracking script.
    """

    config = {
        'dataset': 'amazon_books_60core_train_65_newest_ratings_each_user',
        'dataset_alias': 'amz_65_newest',
        'benchmark_filename': ['train', 'valid', 'test'],
        'epochs': 200,
        'embedding_size': 64,
        'n_layers': 2,
        'reg_weight': 1e-05,
        'metrics': [
            'Precision', 'Recall', 'MAP', 'GAUC',
            'MRR', 'NDCG', 'Hit',
            'AveragePopularity', 'ItemCoverage', 'TailPercentage',
            'GiniIndex', 'ShannonEntropy'
        ],
        'top_k': 10,
        'seed': 2020,
        'reproducibility': True,
        'show_progress': True,
        'save_dataset': False,
        'save_dataloaders': False,
        'repeatable': False,
    }

    tracker = EmissionsTracker()
    tracker.start()

    try:

        create_result_file(config['metrics'])
        model_tests(config)

    finally:
        _ = tracker.stop()
