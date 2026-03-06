import os
import pandas as pd

from cpd_evaluator import CPDEvaluator
from entropy_methods import (
    ShannonEntropyCPD, RenyiEntropyCPD, TsallisEntropyCPD, KLDEntropyCPD,
    PermutationEntropyCPD, BubbleEntropyCPD, SlopeEntropyCPD,
    DispersionEntropyCPD, ApproximateEntropyCPD, SampleEntropyCPD
)


def main():
    datasets = {
    "Mean Abrupt": "data/synthetic/mean_abrupt.csv",
    "Mean Smooth": "data/synthetic/mean_smooth.csv",
    "Var Abrupt": "data/synthetic/var_abrupt.csv",
    "Var Smooth": "data/synthetic/var_smooth.csv",
    "AR Abrupt": "data/synthetic/ar_abrupt.csv",
    "AR Smooth": "data/synthetic/ar_smooth.csv",
    "Quasi Abrupt": "data/synthetic/quasi_abrupt.csv",
    "Quasi Smooth": "data/synthetic/quasi_smooth.csv",
    "Cauchy Abrupt": "data/synthetic/cauchy_abrupt.csv",
    "Cauchy Smooth": "data/synthetic/cauchy_smooth.csv"
}


    algorithms = {
        "Shannon": lambda: ShannonEntropyCPD(window_size=75),
        "Renyi (q=2)": lambda: RenyiEntropyCPD(window_size=75, q=2.0),
        "Tsallis (q=2)": lambda: TsallisEntropyCPD(window_size=75, q=2.0),
        "KLD": lambda: KLDEntropyCPD(window_size=75),
        "Permutation": lambda: PermutationEntropyCPD(window_size=75, m=3),
        "Bubble": lambda: BubbleEntropyCPD(window_size=75, m=3),
        "Slope": lambda: SlopeEntropyCPD(window_size=75, m=3),
        "Dispersion": lambda: DispersionEntropyCPD(window_size=75, c=3),
        "Approximate": lambda: ApproximateEntropyCPD(window_size=75, m=2),
        "Sample": lambda: SampleEntropyCPD(window_size=75, m=2)
    }

    evaluator = CPDEvaluator(tolerance_window=100, calibration_period=300, num_thresholds=50)

    all_results = []
    all_details = []

    for ds_name, ds_path in datasets.items():
        if not os.path.exists(ds_path):
            print(f"File {ds_path} not found! Run the generator first..")
            continue

        print(f"\n=== Testing dataset: {ds_name} ===")

        for algo_name, algo_maker in algorithms.items():
            print(f"Running algorithm: {algo_name}...")

            metrics, details = evaluator.evaluate_dataset(ds_path, algo_maker)

            metrics['Dataset'] = ds_name
            metrics['Algorithm'] = algo_name
            all_results.append(metrics)

            for record in details:
                record['Dataset'] = ds_name
                record['Algorithm'] = algo_name
                all_details.append(record)

    df_results = pd.DataFrame(all_results)
    cols = ['Dataset', 'Algorithm', 'AUPRC', 'F1 (3σ)', 'ADD (Delay 3σ)', 'ARL (Time until false)',
            'Time per row (ms)']
    df_results = df_results[cols]

    print("\n" + "=" * 80)
    print("FINAL RESULTS OF THE EXPERIMENTS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    df_results.to_csv("experiment_results.csv", index=False)

    if all_details:
        df_details = pd.DataFrame(all_details)
        cols_details = ['Dataset', 'Algorithm', 'Series_ID', 'True_CP', 'Predicted_CP', 'Delay', 'Status']
        df_details = df_details[cols_details]

        df_details.to_csv("detailed_predictions.csv", index=False)
        print("\nA detailed table for all rows is saved in 'detailed_predictions.csv'")


if __name__ == "__main__":
    main()