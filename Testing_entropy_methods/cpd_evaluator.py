import pandas as pd
import numpy as np
import time
from scipy.ndimage import median_filter


class CPDEvaluator:
    def __init__(self, tolerance_window: int = 50, calibration_period: int = 200, num_thresholds: int = 50):
        self.tolerance_window = tolerance_window
        self.calibration_period = calibration_period
        self.num_thresholds = num_thresholds

    def evaluate_dataset(self, csv_filepath: str, create_algo_func) -> tuple[dict, list]:
        df = pd.read_csv(csv_filepath)
        y_true = df['changepoint_index'].values
        X = df.drop(columns=['changepoint_index']).values

        n_samples, T = X.shape
        sigma_coefs = np.linspace(0.1, 10.0, self.num_thresholds)

        TP = {k: 0 for k in sigma_coefs}
        FP = {k: 0 for k in sigma_coefs}
        FN = {k: 0 for k in sigma_coefs}

        idx_3sigma = np.argmin(np.abs(sigma_coefs - 3.0))
        k_3sigma = sigma_coefs[idx_3sigma]

        delays_3sigma = []
        fa_times_3sigma = []
        hits_3sigma = 0
        total_algo_time = 0.0
        detailed_records = []

        for i in range(n_samples):
            theta = int(y_true[i])
            series = X[i]
            algo = create_algo_func()

            start_time = time.perf_counter()
            stat_values = np.zeros(T)
            for t, val in enumerate(series):
                stat_values[t] = algo.update(val)
            total_algo_time += (time.perf_counter() - start_time)

            stat_values = median_filter(stat_values, size=5)
            warm_up = getattr(algo, 'window_size', 50)
            baseline = stat_values[warm_up: self.calibration_period]

            if len(baseline) < 10:
                baseline = stat_values[:self.calibration_period]

            mu_base = np.mean(baseline)
            sigma_base = np.std(baseline)
            if sigma_base < 1e-6: sigma_base = 1e-6

            monitoring_area = stat_values[self.calibration_period:]
            pred_cp_3sigma = None
            status_3sigma = "FN (Not found)"

            for k in sigma_coefs:
                deviations = np.abs(monitoring_area - mu_base)
                out_of_bounds = deviations > (k * sigma_base)

                alarms = []
                counter = 0
                for idx, is_out in enumerate(out_of_bounds):
                    if is_out:
                        counter += 1
                    else:
                        counter = 0
                    if counter >= 5:
                        alarms = [idx - 5 + 1]
                        break

                if len(alarms) > 0:
                    tau = alarms[0] + self.calibration_period

                    if tau < theta:
                        FP[k] += 1
                        if np.isclose(k, k_3sigma) and pred_cp_3sigma is None:
                            fa_times_3sigma.append(tau)
                            pred_cp_3sigma = tau
                            status_3sigma = "FP (False alarm)"
                    elif theta <= tau <= theta + self.tolerance_window:
                        if np.isclose(k, k_3sigma) and status_3sigma != "FP (False alarm)":
                            TP[k] += 1
                            delays_3sigma.append(tau - theta)
                            pred_cp_3sigma = tau
                            status_3sigma = "TP (Success)"
                        elif not np.isclose(k, k_3sigma):
                            TP[k] += 1
                    else:
                        if np.isclose(k, k_3sigma) and status_3sigma == "FN (Not found)":
                            FN[k] += 1
                            pred_cp_3sigma = tau
                            status_3sigma = "FN (Too late)"
                        elif not np.isclose(k, k_3sigma):
                            FN[k] += 1
                else:
                    FN[k] += 1

            if pred_cp_3sigma is not None and abs(pred_cp_3sigma - theta) <= 50:
                hits_3sigma += 1

            detailed_records.append({
                "Series_ID": i,
                "True_CP": theta,
                "Predicted_CP": pred_cp_3sigma if pred_cp_3sigma is not None else "Not found",
                "Delay": (pred_cp_3sigma - theta) if status_3sigma == "TP (success)" else "-",
                "Status": status_3sigma
            })

        precisions, recalls = [], []
        for k in sigma_coefs:
            tp, fp, fn = TP[k], FP[k], FN[k]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        r_sorted = np.array(recalls)
        p_sorted = np.array(precisions)
        sort_idx = np.argsort(r_sorted)

        if hasattr(np, 'trapezoid'):
            auprc = np.trapezoid(p_sorted[sort_idx], r_sorted[sort_idx])
        else:
            auprc = np.trapz(p_sorted[sort_idx], r_sorted[sort_idx])

        tp3, fp3, fn3 = TP[k_3sigma], FP[k_3sigma], FN[k_3sigma]
        p3 = tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else 0.0
        r3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0.0
        f1_3sigma = 2 * (p3 * r3) / (p3 + r3) if (p3 + r3) > 0 else 0.0

        metrics = {
            "Hit Rate ±50": round(hits_3sigma / n_samples, 4),
            "AUPRC": round(auprc, 4),
            "F1 (3σ)": round(f1_3sigma, 4),
            "ADD (Delay 3σ)": round(np.mean(delays_3sigma), 2) if delays_3sigma else 0.0,
            "ARL (Time until false)": round(np.mean(fa_times_3sigma), 2) if fa_times_3sigma else 0.0,
            "Time per row (ms)": round((total_algo_time / n_samples) * 1000, 2)
        }
        return metrics, detailed_records