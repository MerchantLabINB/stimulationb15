import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the two CSV files
new_file = r"C:\Users\samae\Documents\GitHub\stimulationb15\data\ht_24newthresholds\submovement_detailed_summary.csv"
old_file = r"C:\Users\samae\Documents\GitHub\stimulationb15\data\ht_24newcsv\submovement_detailed_summary_old.csv"

new_df = pd.read_csv(new_file)
old_df = pd.read_csv(old_file)

# Print column names to check if they match
print("New file columns:")
print(new_df.columns.tolist())
print("Old file columns:")
print(old_df.columns.tolist())

# Get descriptive statistics for both files
print("\nDescriptive statistics for the NEW file:")
print(new_df.describe(include='all').transpose())
print("\nDescriptive statistics for the OLD file:")
print(old_df.describe(include='all').transpose())

# List of key columns (adjust as needed)
metrics = ["Valor Pico (velocidad)", "Latencia al Inicio (s)", "Duración Total (s)"]

# Compare means and medians for each metric
for col in metrics:
    new_mean = new_df[col].mean()
    old_mean = old_df[col].mean()
    new_median = new_df[col].median()
    old_median = old_df[col].median()
    print(f"\nColumn: {col}")
    print(f"  NEW file: mean = {new_mean:.3f}, median = {new_median:.3f}")
    print(f"  OLD file: mean = {old_mean:.3f}, median = {old_median:.3f}")

    # If the metric is approximately normally distributed, you can do a t-test:
    t_stat, p_value = stats.ttest_ind(new_df[col].dropna(), old_df[col].dropna(), equal_var=False)
    print(f"  T-test: t = {t_stat:.3f}, p = {p_value:.3g}")

    # And a non-parametric test (Mann–Whitney U) if normality is questionable:
    u_stat, p_value_u = stats.mannwhitneyu(new_df[col].dropna(), old_df[col].dropna(), alternative='two-sided')
    print(f"  Mann-Whitney U test: U = {u_stat:.3f}, p = {p_value_u:.3g}")

    # Plot histograms for visual comparison
    plt.figure(figsize=(8, 4))
    sns.histplot(new_df[col].dropna(), color='blue', label='NEW', kde=True, stat="density", bins=30, alpha=0.6)
    sns.histplot(old_df[col].dropna(), color='red', label='OLD', kde=True, stat="density", bins=30, alpha=0.6)
    plt.legend()
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()

    # Boxplot comparison
    plt.figure(figsize=(6, 4))
    data_to_plot = [new_df[col].dropna(), old_df[col].dropna()]
    plt.boxplot(data_to_plot, labels=["NEW", "OLD"])
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()

# You can also save the summary statistics to CSV files for further review:
# new_df.describe(include='all').transpose().to_csv("new_submovement_summary_stats.csv")
# old_df.describe(include='all').transpose().to_csv("old_submovement_summary_stats.csv")
