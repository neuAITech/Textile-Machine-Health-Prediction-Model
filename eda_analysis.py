import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_eda():
    print("Starting EDA for Textile Machine Failure Prediction...")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 2. General Info & Stats
    with open(os.path.join(OUTPUT_DIR, "data_summary.txt"), "w") as f:
        f.write("=== DATASET OVERVIEW ===\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Colum Dtypes:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nMissing Values:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\nDescriptive Statistics:\n")
        f.write(df.describe().to_string())

    # 3. Target Distribution
    print("Generating target distributions...")
    if 'Degradation_Phase' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Degradation_Phase', order=['Healthy', 'Early Wear', 'Moderate Wear', 'Critical'])
        plt.title('Distribution of Machine Degradation Phases')
        plt.savefig(os.path.join(OUTPUT_DIR, "target_degradation_dist.png"))
        plt.close()

    if 'RUL_Hours' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['RUL_Hours'], kde=True, color='teal')
        plt.title('Distribution of Remaining Useful Life (RUL) Hours')
        plt.savefig(os.path.join(OUTPUT_DIR, "target_rul_dist.png"))
        plt.close()

    # 4. Correlation Heatmap
    print("Generating correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(16, 12))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()

    # 5. Key Feature Distributions vs Target
    print("Generating feature vs target plots...")
    key_features = ['Machine_Speed_RPM', 'Temp_C', 'Energy_kWh', 'Mechanical_Friction_Index', 'Wear_Score', 'Efficiency_Index']

    for feat in key_features:
        if feat in df.columns and 'Degradation_Phase' in df.columns:
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df, x='Degradation_Phase', y=feat, order=['Healthy', 'Early Wear', 'Moderate Wear', 'Critical'])
            plt.title(f'{feat} Distribution across Degradation Phases')
            plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{feat}_vs_target.png"))
            plt.close()

    # 6. Categorical Analysis
    print("Generating categorical analysis...")
    cats = ['Machine_Type', 'Section', 'Shift']
    for cat in cats:
        if cat in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=cat)
            plt.title(f'Count of Records by {cat}')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(OUTPUT_DIR, f"count_{cat}.png"))
            plt.close()

    # 7. Time-Series Trends (Daily average)
    if 'Timestamp' in df.columns:
        print("Generating time-series trends...")
        df_sorted = df.sort_values('Timestamp')
        # Only select numeric columns for resampling mean
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_daily = df_sorted.set_index('Timestamp')[numeric_cols].resample('D').mean()

        plt.figure(figsize=(14, 8))
        plt.plot(df_daily.index, df_daily['Temp_C'], label='Avg Temp (C)', color='red')
        plt.plot(df_daily.index, df_daily['Energy_kWh'], label='Avg Energy (kWh)', color='blue')
        plt.title('Daily Average Operational Trends')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "time_series_trends.png"))
        plt.close()

    # 8. Multivariate: Wear vs Friction colored by Phase
    if all(col in df.columns for col in ['Mechanical_Friction_Index', 'Wear_Score', 'Degradation_Phase']):
        print("Generating multivariate scatter plot...")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='Mechanical_Friction_Index', y='Wear_Score', hue='Degradation_Phase',
                        hue_order=['Healthy', 'Early Wear', 'Moderate Wear', 'Critical'], alpha=0.6)
        plt.title('Wear Score vs Friction Index by Degradation Phase')
        plt.savefig(os.path.join(OUTPUT_DIR, "wear_vs_friction_multivariate.png"))
        plt.close()

    print(f"EDA Complete. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    perform_eda()
