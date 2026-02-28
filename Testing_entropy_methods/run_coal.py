import pandas as pd
from cpd_evaluator import CPDEvaluator
from entropy_methods import (
    ShannonEntropyCPD, RenyiEntropyCPD, TsallisEntropyCPD, KLDEntropyCPD,
    PermutationEntropyCPD, BubbleEntropyCPD, SlopeEntropyCPD,
    DispersionEntropyCPD, ApproximateEntropyCPD, SampleEntropyCPD
)


def main():
    ds_path = "data/real/coal_mining.csv"

    algorithms = {
        "Shannon": lambda: ShannonEntropyCPD(window_size=15),
        "Renyi (q=2)": lambda: RenyiEntropyCPD(window_size=15, q=2.0),
        "Tsallis (q=2)": lambda: TsallisEntropyCPD(window_size=15, q=2.0),
        "KLD": lambda: KLDEntropyCPD(window_size=30),
        "Permutation": lambda: PermutationEntropyCPD(window_size=15, m=3),
        "Bubble": lambda: BubbleEntropyCPD(window_size=15, m=3),
        "Slope": lambda: SlopeEntropyCPD(window_size=15, m=3),
        "Dispersion": lambda: DispersionEntropyCPD(window_size=15, c=3),
        "Approximate": lambda: ApproximateEntropyCPD(window_size=15, m=2),
        "Sample": lambda: SampleEntropyCPD(window_size=15, m=2)
    }

    evaluator = CPDEvaluator(tolerance_window=10, calibration_period=30, num_thresholds=50)

    all_metrics = []
    all_details = []

    print("\n=== Real-World Data Testing: Coal Mining Disasters ===")

    for algo_name, algo_maker in algorithms.items():
        metrics, details = evaluator.evaluate_dataset(ds_path, algo_maker)

        metrics['Algorithm'] = algo_name
        all_metrics.append(metrics)

        for record in details:
            record['Algorithm'] = algo_name
            all_details.append(record)

    df_metrics = pd.DataFrame(all_metrics)
    cols_metrics = ['Algorithm', 'AUPRC', 'F1 (3σ)', 'ADD (Delay 3σ)', 'ARL (Time until false)']
    df_metrics = df_metrics[cols_metrics]

    print("\n" + "=" * 70)
    print("AGGREGATED METRICS")
    print("=" * 70)
    print(df_metrics.to_string(index=False))

    df_details = pd.DataFrame(all_details)
    cols_details = ['Algorithm', 'Series_ID', 'True_CP', 'Predicted_CP', 'Delay', 'Status']
    df_details = df_details[cols_details]

    df_details.to_csv("detailed_predictions.csv", index=False)

    print("\n" + "=" * 80)
    print("DETAILED TABLE (1 EXAMPLE FOR EACH OF THE 10 ALGORITHMS))")
    print("=" * 80)

    summary_view = df_details.groupby('Algorithm').first().reset_index()
    print(summary_view[cols_details].to_string(index=False))

    print("\nFull statistics for all 10 methods are stored in 'detailed_predictions.csv'")


if __name__ == "__main__":
    main()