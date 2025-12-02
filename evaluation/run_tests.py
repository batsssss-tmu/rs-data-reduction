import time

from evaluation import record_result

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer
from recbole.utils import utils

models = ['LightGCN']


def model_tests(config_params):
    """
    Trains and tests models in the RecBole framework.
    Results are recorded to a CSV file.

    Args:
        config_params (dict): Recbole config for all tests

    Returns:
        None
    """

    indent = '  '
    print('\n', end='')

    for model_name in models:

        print(indent + 'Model:', model_name)
        print(indent + indent + 'preparing data...')

        run_params = {'model': model_name}

        config = Config(model=model_name, config_dict=config_params)
        dataset = create_dataset(config)

        train_data, valid_data, test_data = data_preparation(config, dataset)

        model_class = utils.get_model(model_name)
        model = model_class(config, train_data.dataset).to(config.device)
        trainer = Trainer(config, model)

        print(indent + indent + 'training...')
        train_result, timings = _run_train(trainer, train_data, valid_data)

        print(indent + indent + 'recording training results...')
        run_params['stage'] = 'train'
        record_result(config_params['metrics'], train_result, run_params, timings)

        print(indent + indent + 'evaluating...')
        evaluation_result, timings = _run_evaluate(trainer, test_data)

        print(indent + indent + 'recording evaluation results...')
        run_params['stage'] = 'eval'
        record_result(config_params['metrics'], evaluation_result, run_params, timings)

    print('\n', end='')
    print(indent + 'All Tests Complete')


def _run_train(trainer, train_data, valid_data):
    """
    Trains a model on the given training data and validates using the given validation data.
    Provides the best score of metric used for early stopping
    and the best run validation results for all metrics.
    Captures the start time and duration of the training.

    Args:
        trainer (Trainer): instance of Trainer
        train_data (Dataset): Training dataset
        valid_data (Dataset): Validation dataset

    Returns:
        tuple (dict, tuple (float, float)): Best validation results and timing of the training.
    """

    train_start_time = time.time()
    best_valid_score, best_valid_result = trainer.fit(train_data=train_data, valid_data=valid_data)
    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    return best_valid_result, (train_start_time, train_duration)


def _run_evaluate(trainer, test_data):
    """
    Evaluates a model on the given test data.
    Provides the results from selected metrics.
    Captures the start time and duration of the test.

    Args:
        trainer (Trainer): instance of Trainer
        test_data (Dataset): Test dataset

    Returns:
        tuple (dict, tuple (float, float)): Test results and timing of the testing.
    """

    eval_start_time = time.time()
    test_result = trainer.evaluate(test_data)
    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time

    return test_result, (eval_start_time, eval_duration)
