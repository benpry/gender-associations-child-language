"""
This file contains code that runs hypothesis tests using bootstrapped data
"""
import numpy as np


def one_sample_hypothesis_test(samples, bonferroni=1, string=False):
    """
    Run the one-sample hypothesis test, where the null hypothesis is that
    the mean correlation is 0.
    """
    sample_mean = np.mean(samples)

    null_samples = [x - sample_mean for x in samples]

    samples_above_mean = [x for x in null_samples if abs(x) >= abs(sample_mean)]
    p_value = len(samples_above_mean) / len(samples)

    sorted_samples = sorted(samples)
    ci_lower = str(np.round(sorted_samples[int(0.025 * len(sorted_samples))], 2)).lstrip("0").ljust(3, "0")
    ci_upper = str(np.round(sorted_samples[int(0.975 * len(sorted_samples))], 2)).lstrip("0").ljust(3, "0")

    if string:
        return f"{str(np.round(sample_mean, 2)).lstrip('0')} ({str(np.round(min(bonferroni * p_value, 1),5)).lstrip('0')}) [{ci_lower}, {ci_upper}]"
    else:
        return sample_mean, min(bonferroni * p_value, 1), f"[{ci_lower}, {ci_upper}]"


def two_sample_hypothesis_test(x_samples, y_samples, bonferroni=1, string=False):
    """
    Run the two-sample hypothesis test, where the null hypothesis is that the two
    groups (x_samples and y_samples) have the same mean correlation.
    """
    x_samples = list(x_samples)
    y_samples = list(y_samples)

    x_mean = np.mean(x_samples)
    y_mean = np.mean(y_samples)

    mean_stat = x_mean - y_mean
    all_stats = [x_samples[i] - y_samples[i] for i in range(len(x_samples))]
    all_stats_null = [s - mean_stat for s in all_stats]

    samples_above_mean = [x for x in all_stats_null if abs(x) >= abs(mean_stat)]
    p_value = len(samples_above_mean) / len(all_stats_null)

    sorted_stats = sorted(all_stats)
    ci_lower = str(np.round(sorted_stats[int(0.025 * len(sorted_stats))], 2)).lstrip("0").ljust(3, "0")
    ci_upper = str(np.round(sorted_stats[int(0.975 * len(sorted_stats))], 2)).lstrip("0").ljust(3, "0")

    if string:
        return f"{str(np.round(mean_stat, 2)).lstrip('0')} ({str(min(bonferroni * p_value, 1)).lstrip('0')}) [{ci_lower}, {ci_upper}]"
    else:
        return mean_stat, min(bonferroni * p_value, 1), f"[{ci_lower}, {ci_upper}]"
