"""
Level 1: Data Overview & Route Table Generation
Sysslan IT Solutions - Data Science Internship

Task:
- Inspect dataset shape, data types, and memory usage.
- Generate train-wise route summary table.
- Calculate summary statistics for distance and stops.
- Check for missing values and placeholder formats.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_inspect_structure(df):
    """Prints basic info about columns, types, and shape."""
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"Unique Trains: {df['Train_No'].nunique():,}")
    print(f"Unique Stations: {df['Station_Code'].nunique():,}\n")
    
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(t) for t in df.dtypes],
        "Non-Null Count": [df[c].notnull().sum() for c in df.columns],
        "Null Count": [df[c].isnull().sum() for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Sample": [df[c].iloc[0] for c in df.columns]
    })
    return col_info


def check_missing_and_placeholders(df):
    """
    Checks standard nulls and also searches for railway-specific placeholders 
    like '00:00:00' in time columns or 0 distance at source stations.
    """
    records = []
    for col in df.columns:
        nulls = df[col].isnull().sum()
        zeros = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            zeros = (df[col] == 0).sum()
        else:
            zeros = (df[col].astype(str).str.strip() == "00:00:00").sum()
            
        records.append({
            "Column": col,
            "Null_Count": nulls,
            "Null_Pct": f"{(nulls / len(df)) * 100:.1f}%",
            "Zero_or_00:00:00_Count": zeros
        })
    return pd.DataFrame(records)


def build_train_route_table(df):
    """
    Groups stop records by Train_No to build a route-level table.
    Captures origin, destination, total stops, and total distance.
    """
    # make sure stops are properly ordered
    df_sorted = df.sort_values(by=["Train_No", "SN"])
    
    routes = df_sorted.groupby("Train_No").agg(
        Route_Number=("Route_Number", "first"),
        Source_Station=("Station_Name", "first"),
        Source_Code=("Station_Code", "first"),
        Destination_Station=("Station_Name", "last"),
        Destination_Code=("Station_Code", "last"),
        Total_Stops=("SN", "count"),
        Min_Distance=("Distance", "min"),
        Max_Distance=("Distance", "max"),
        Scheduled_Departure=("Departure_Time", "first"),
        Scheduled_Arrival=("Arrival_time", "last"),
        Coach_1A=("1A", "max"),
        Coach_2A=("2A", "max"),
        Coach_3A=("3A", "max"),
        Coach_SL=("SL", "max")
    ).reset_index()
    
    routes["Total_Distance_KM"] = routes["Max_Distance"] - routes["Min_Distance"]
    routes = routes.drop(columns=["Min_Distance", "Max_Distance"])
    return routes


def compute_basic_stats(df, routes):
    """Computes descriptive statistics for stop and route level distance and stops."""
    stats_dict = {
        "Stop Distance (KM)": df["Distance"].describe(),
        "Route Total Distance (KM)": routes["Total_Distance_KM"].describe(),
        "Route Total Stops": routes["Total_Stops"].describe(),
        "Stop Sequence (SN)": df["SN"].describe()
    }
    stats_df = pd.DataFrame(stats_dict).T
    stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
    return stats_df


def main():
    print("--------------------------------------------------")
    print("Level 1: Dataset Overview & Train Route Table")
    print("--------------------------------------------------")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find dataset at {DATA_PATH}")
        return
        
    print(f"Reading dataset: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.\n")
    
    # 1. Structure
    print("--- 1. Dataset Structure ---")
    col_info = load_and_inspect_structure(df)
    print(col_info.to_string(index=False))
    print()
    
    # 2. Missing values
    print("--- 2. Missing Values & Placeholder Audit ---")
    missing_df = check_missing_and_placeholders(df)
    print(missing_df.to_string(index=False))
    print("\nNote on placeholders:")
    print(" - '00:00:00' in Arrival_time is recorded at origin stations (train hasn't arrived from anywhere).")
    print(" - '00:00:00' in Departure_Time is recorded at terminating stations (journey ends).")
    print(" - Distance 0 is the starting point of each route.\n")
    
    # 3. Route table
    print("--- 3. Generating Train-wise Route Summary Table ---")
    routes = build_train_route_table(df)
    out_table_path = os.path.join(OUTPUT_DIR, "train_wise_route_table.csv")
    routes.to_csv(out_table_path, index=False)
    print(f"Route summary table saved to: {out_table_path}")
    print(f"Total routes compiled: {len(routes):,}\n")
    print("Sample routes (first 5):")
    print(routes[["Train_No", "Source_Station", "Destination_Station", "Total_Stops", "Total_Distance_KM"]].head())
    print()
    
    # 4. Statistics
    print("--- 4. Distance and Stops Statistics ---")
    stats_df = compute_basic_stats(df, routes)
    print(stats_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "IQR"]].round(2))
    print()
    
    # Write summary text
    report_file = os.path.join(OUTPUT_DIR, "level_1_overview_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Level 1: Data Overview & Route Table Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Unique trains: {df['Train_No'].nunique():,}\n")
        f.write(f"Unique stations: {df['Station_Code'].nunique():,}\n\n")
        f.write("Column Schema:\n")
        f.write(col_info.to_string(index=False) + "\n\n")
        f.write("Missing & Placeholders:\n")
        f.write(missing_df.to_string(index=False) + "\n\n")
        f.write("Basic Statistics:\n")
        f.write(stats_df.round(2).to_string() + "\n")
        
    print(f"Summary report written to: {report_file}")
    print("--------------------------------------------------")
    print("Level 1 completed.\n")


if __name__ == "__main__":
    main()
