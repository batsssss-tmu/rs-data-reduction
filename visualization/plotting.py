from pathlib import Path
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

output_dir = Path('../output')


def _plot_bar(data, pivot, title):
    """
    Plots a bar graph of the given data.
    Displays it and saves an image file.
    Based on https://github.com/swapUniba/datared-green-recsys

    Args:
        data (DataFrame): results
        pivot (str): test variable
        title (str): title of plot and image file

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=20)
    plt.ylabel("Emissions (g)", fontsize=18)
    plt.xlabel(pivot + 's', fontsize=18)
    plt.xticks(rotation=30, fontsize=15)

    values = data[data.columns[4]].values.tolist()
    x_labels = data[pivot].values.tolist()

    plt.bar(x_labels, values, width=0.9)

    for i, v in enumerate(values):
        plt.text(i, v, str(round(v, 2)), ha='center', va='bottom', fontsize=15)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.tight_layout()

    plt.savefig(Path(output_dir, (title.translate(str.maketrans({' ': '_', '(': '', ')': ''}))).lower() + '.png'))

    plt.show()
    plt.close()


def plot_model_comparison(data):
    """
    Plots a bar graph of power consumption per model.

    Args:
        data (DataFrame): power consumption results grouped by model

    Returns:
        None
    """

    _plot_bar(data, 'Model', 'Model Comparison')


def plot_data_reduction_comparison(data):
    """
    Plots a bar graph of power consumption per data reduction.

    Args:
        data (DataFrame): power consumption results grouped by data reduction.

    Returns:
        None
    """

    _plot_bar(data, 'Dataset', 'Data Reduction Comparison (BPR)')


def plot_accuracy_emissions_tradeoff(power_results, accuracy_results):
    """
    Plots a scatter plot of model accuracy and emissions for each data reduction.
    Based on https://github.com/swapUniba/datared-green-recsys

    Args:
        power_results (DataFrame): power consumption results.
        accuracy_results (DataFrame): results of accuracy metrics.

    Returns:
        None
    """

    metrics_list = ['Precision', 'Recall', 'MAP', 'NDCG', 'AveragePopularity', 'GiniIndex']

    models = {
        'BPR': {
            'colour': '#1f77b4',  # Blue
            'datasets': [
                'amz_100_newest',
                'amz_80_newest',
                'amz_65_newest'
            ]
        },
        'LightGCN': {
            'colour': '#8c564b',  # Brown
            'datasets': [
                'amz_65_newest'
            ]
        }
    }

    marker_list = ['v', '+', 'x']

    compare_trade_offs = defaultdict(dict)

    for metric in metrics_list:
        compare_trade_offs[metric] = defaultdict(dict)

        for model in models:
            compare_trade_offs[metric][model] = []

            for dataset in models[model]['datasets']:

                accuracy_result = accuracy_results.loc[
                    (accuracy_results.Model == model)
                    & (accuracy_results.Dataset == dataset)
                    & (accuracy_results.Stage == 'eval')]
                aix = accuracy_result.index.tolist()[0]

                power_result = power_results.loc[
                    (power_results.Model == model)
                    & (power_results.Dataset == dataset)]
                pix = power_result.index.tolist()[0]

                compare_trade_offs[metric][model].append({
                    'dataset': dataset,
                    'score': accuracy_result.at[aix, metric],
                    'emissions': power_result.at[pix, 'Emissions @30 (kgCO2-eq, Ontario 2024)']
                })

    for metric in compare_trade_offs:

        plt.figure(figsize=(10, 8))
        plt.title(f'Trade-off: {metric}@10 vs Emissions', fontsize=20)
        plt.ylabel('Emissions (g)', fontsize=18)
        plt.xlabel(metric + '@10', fontsize=18)

        min_score = accuracy_results[metric].min()
        max_score = accuracy_results[metric].max()
        score_diff = max_score - min_score
        plt.xlim(min_score - 0.15 * score_diff, max_score + 0.15 * score_diff)

        min_emission = power_results['Emissions @30 (kgCO2-eq, Ontario 2024)'].min()
        max_emission = power_results['Emissions @30 (kgCO2-eq, Ontario 2024)'].max()
        emission_diff = max_emission - min_emission
        plt.ylim(min_emission - 0.15 * emission_diff, max_emission + 0.15 * emission_diff)

        model_ix = 0
        for model in compare_trade_offs[metric]:
            for i in range(len(compare_trade_offs[metric][model]) - 1):
                x1 = compare_trade_offs[metric][model][i]['score']
                y1 = compare_trade_offs[metric][model][i]['emissions']
                x2 = compare_trade_offs[metric][model][i + 1]['score']
                y2 = compare_trade_offs[metric][model][i + 1]['emissions']

                sctr = plt.scatter(x1, y1, color=models[model]['colour'], label=compare_trade_offs[metric][model][i]['dataset'], marker=marker_list[i])
                if model_ix == 0:
                    sctr.set_label(compare_trade_offs[metric][model][i]['dataset'])
                if i == 0:
                    plt.text(x1, y1 + 5, model, fontsize=15)
                plt.plot([x1, x2], [y1, y2], color=models[model]['colour'], linestyle='-', linewidth=1)

            x_last = compare_trade_offs[metric][model][-1]['score']
            y_last = compare_trade_offs[metric][model][-1]['emissions']
            sctr = plt.scatter(x_last, y_last, color=models[model]['colour'], marker=marker_list[-1])
            if model_ix == 0:
                sctr.set_label(compare_trade_offs[metric][model][-1]['dataset'])
            plt.grid(True)
            plt.legend()
            if len(compare_trade_offs[metric][model]) <= 1:
                plt.text(x_last, y_last + 5, model, fontsize=15)
            plt.tight_layout()
            model_ix += 1

        plt.savefig(Path(output_dir, 'accuracy_emissions_tradeoff_' + metric.lower() + '.png'))

    plt.show()
    plt.close()


def main():

    power_results = pd.read_csv(Path(output_dir, 'power_results.csv'))
    power_results = power_results[['experiment_id', 'run_id', 'duration', 'energy_consumed']]
    power_results.rename(columns={
        'experiment_id': 'Model',
        'run_id': 'Dataset',
        'duration': 'Duration',
        'energy_consumed': 'Energy (kWh)'
    }, inplace=True)

    carbon_intensities = {
        'Ontario 2024': 30,
        'Spillo Paper': 267
    }

    for location, ci in carbon_intensities.items():
        power_results['Emissions @' + str(ci) + ' (kgCO2-eq, ' + location + ')'] = power_results.apply(
            lambda row: row.iloc[3] * ci, axis=1)

    # Bar graph of emissions per model
    plot_model_comparison(power_results.head(2))

    # Bar graph of emissions per data reduction
    plot_data_reduction_comparison(power_results.tail(3))

    accuracy_results = pd.read_csv(Path(output_dir, 'accuracy_results.csv'))

    # Scatter plot of accuracy vs emissions
    plot_accuracy_emissions_tradeoff(power_results, accuracy_results)


if __name__ == '__main__':
    main()
