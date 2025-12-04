from pathlib import Path
import csv

output_dir = Path('output')


def create_result_file(metrics):
    """
    If not yet exists, creates a CSV file to record evaluation results,
    with headings for the given metrics.

    Args:
        metrics (list): Evaluation metric names

    Returns:
        None
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir, Path('accuracy_results.csv'))

    if not output_path.is_file():

        headers = ['Model', 'Dataset', 'Stage']
        headers.extend(metrics)
        headers.extend(['Start Time', 'Duration (s)'])

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def record_result(metrics, results, run_params, timings):
    """
    Records the results of the current stage of a test run in a CSV file.
    If metrics are not available, records only the start and duration of the run.

    Args:
        metrics (list): Evaluation metric names
        results (dict): Evaluation metric results
        run_params (dict): Parameters for current stage of test run
        timings (tuple): Start time (epoch) and duration (seconds)

    Returns:
        None
    """

    output_path = Path(output_dir, Path('accuracy_results.csv'))

    row = list(run_params.values())

    if results is not None:
        row.extend(list(results.values()))
    else:
        row.extend([None] * len(metrics))

    row.extend(list(map(lambda t: round(t, 2), timings)))

    with open(output_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
