"""
Level 2: Data Cleaning & Feature Engineering
Sysslan IT Solutions - Data Science Internship

Task:
- Clean whitespace, verify duplicates, and ensure proper sorting.
- Standardize arrival/departure times and calculate station halt minutes.
- Handle midnight rollovers across multi-day journeys to calculate true journey duration.
- Engineer stop-level metrics (cumulative time, leg speed) and route-level metrics.
- Export cleaned datasets for exploration and modeling.
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


def clean_raw_data(df):
    """Trims whitespace and sorts strictly by Train_No and Stop Sequence."""
    # strip text whitespace
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # check for exact duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"Dropping {dupes} duplicate rows...")
        df = df.drop_duplicates()
        
    df = df.sort_values(by=["Train_No", "SN"]).reset_index(drop=True)
    return df


def time_str_to_minutes(series):
    """Converts HH:MM:SS string to minutes from midnight (0 to 1439)."""
    parts = series.astype(str).str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def process_stop_level_features(df):
    """
    Computes leg travel times, halt times, and running cumulative journey duration.
    Accurately handles trains crossing midnight via modulo 1440.
    """
    # parse arrival and departure minutes
    df["arr_min"] = time_str_to_minutes(df["Arrival_time"])
    df["dep_min"] = time_str_to_minutes(df["Departure_Time"])
    
    # flag origin, destination, and intermediate stops
    df["prev_train"] = df["Train_No"].shift(1)
    df["next_train"] = df["Train_No"].shift(-1)
    
    is_origin = (df["Train_No"] != df["prev_train"])
    is_dest = (df["Train_No"] != df["next_train"])
    is_intermediate = (~is_origin) & (~is_dest)
    
    df["Is_Origin"] = is_origin
    df["Is_Destination"] = is_dest
    
    # at start station, arrival equals departure
    df.loc[is_origin, "arr_min"] = df.loc[is_origin, "dep_min"]
    # at terminus station, if departure is recorded as 00:00:00, use arrival time
    df.loc[is_dest & (df["dep_min"] == 0), "dep_min"] = df.loc[is_dest & (df["dep_min"] == 0), "arr_min"]
    
    # halt duration (only intermediate stops have operational halts within the journey)
    df["Halt_Minutes"] = np.where(is_intermediate, (df["dep_min"] - df["arr_min"]) % 1440, 0)
    
    # leg travel time: from previous station departure to current station arrival
    prev_dep = df["dep_min"].shift(1)
    prev_dist = df["Distance"].shift(1)
    
    df["Leg_Minutes"] = np.where(~is_origin, (df["arr_min"] - prev_dep) % 1440, 0)
    df["Leg_Distance_KM"] = np.where(~is_origin, df["Distance"] - prev_dist, 0)
    
    # leg speed
    leg_hours = df["Leg_Minutes"] / 60.0
    df["Leg_Speed_KMH"] = np.where(leg_hours > 0, (df["Leg_Distance_KM"] / leg_hours).round(2), 0.0)
    
    # step duration = leg moving time + station halt time
    df["Step_Duration_Min"] = df["Leg_Minutes"] + df["Halt_Minutes"]
    
    # cumulative elapsed time from origin
    df["Cumulative_Duration_Min"] = df.groupby("Train_No")["Step_Duration_Min"].cumsum()
    df["Cumulative_Duration_Hours"] = (df["Cumulative_Duration_Min"] / 60.0).round(2)
    df["Cumulative_Halt_Min"] = df.groupby("Train_No")["Halt_Minutes"].cumsum()
    df["Cumulative_Distance_KM"] = df["Distance"] - df.groupby("Train_No")["Distance"].transform("first")
    
    # clean up temp columns
    drop_cols = ["prev_train", "next_train", "arr_min", "dep_min", "Step_Duration_Min"]
    df = df.drop(columns=drop_cols)
    return df


def aggregate_route_level_features(stops_df):
    """
    Summarizes stop records into one row per train route with operational features.
    """
    routes = stops_df.groupby("Train_No").agg(
        Route_Number=("Route_Number", "first"),
        Source_Station=("Station_Name", "first"),
        Source_Code=("Station_Code", "first"),
        Destination_Station=("Station_Name", "last"),
        Destination_Code=("Station_Code", "last"),
        Total_Stops=("SN", "count"),
        Total_Distance_KM=("Distance", lambda s: s.iloc[-1] - s.iloc[0]),
        Total_Transit_Minutes=("Leg_Minutes", "sum"),
        Total_Halt_Minutes=("Halt_Minutes", "sum"),
        Journey_Duration_Minutes=("Cumulative_Duration_Min", "last"),
        Journey_Duration_Hours=("Cumulative_Duration_Hours", "last"),
        Coach_1A=("1A", "max"),
        Coach_2A=("2A", "max"),
        Coach_3A=("3A", "max"),
        Coach_SL=("SL", "max")
    ).reset_index()
    
    # commercial average speed (including halts)
    routes["Average_Speed_KMH"] = np.where(
        routes["Journey_Duration_Hours"] > 0,
        (routes["Total_Distance_KM"] / routes["Journey_Duration_Hours"]).round(2),
        0.0
    )
    
    # pure transit speed (excluding halts)
    transit_hrs = routes["Total_Transit_Minutes"] / 60.0
    routes["Transit_Speed_KMH"] = np.where(
        transit_hrs > 0,
        (routes["Total_Distance_KM"] / transit_hrs).round(2),
        0.0
    )
    
    # station density and halt ratio
    routes["Distance_Per_Stop_KM"] = (routes["Total_Distance_KM"] / routes["Total_Stops"]).round(2)
    routes["Halt_Ratio"] = np.where(
        routes["Journey_Duration_Minutes"] > 0,
        (routes["Total_Halt_Minutes"] / routes["Journey_Duration_Minutes"]).round(4),
        0.0
    )
    
    # classify routes by distance
    bins = [-1, 100, 500, 1500, 10000]
    labels = ["Short-Haul (<=100 km)", "Medium-Haul (101-500 km)", "Long-Haul (501-1500 km)", "Super Long-Haul (>1500 km)"]
    routes["Route_Category"] = pd.cut(routes["Total_Distance_KM"], bins=bins, labels=labels)
    
    return routes


def main():
    print("--------------------------------------------------")
    print("Level 2: Data Cleaning & Feature Engineering")
    print("--------------------------------------------------")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
        
    print(f"Loading raw dataset: {DATA_PATH} ...")
    raw_df = pd.read_csv(DATA_PATH)
    
    # 1. Clean
    print("Cleaning whitespace and ordering sequences...")
    cleaned_df = clean_raw_data(raw_df)
    
    # 2. Stop-level features
    print("Calculating halt times, leg travel times, and cumulative journey duration...")
    stops_df = process_stop_level_features(cleaned_df)
    stops_out = os.path.join(OUTPUT_DIR, "cleaned_train_stops.csv")
    stops_df.to_csv(stops_out, index=False)
    print(f"Cleaned stop-level dataset saved to: {stops_out}")
    print(f"Total stops processed: {len(stops_df):,}\n")
    
    # 3. Route-level features
    print("Aggregating route-level summary and computing speeds...")
    routes_df = aggregate_route_level_features(stops_df)
    routes_out = os.path.join(OUTPUT_DIR, "cleaned_train_routes.csv")
    routes_df.to_csv(routes_out, index=False)
    print(f"Cleaned route-level dataset saved to: {routes_out}")
    print(f"Total routes processed: {len(routes_df):,}\n")
    
    print("Sample engineered routes:")
    print(routes_df[["Train_No", "Source_Station", "Destination_Station", "Total_Stops", 
                     "Total_Distance_KM", "Journey_Duration_Hours", "Average_Speed_KMH"]].head())
    print()
    
    print("Summary of key engineered metrics:")
    metrics = ["Total_Distance_KM", "Total_Stops", "Journey_Duration_Hours", "Average_Speed_KMH", "Total_Halt_Minutes"]
    print(routes_df[metrics].describe().round(2))
    print()
    
    print("Route Category Breakdown:")
    print(routes_df["Route_Category"].value_counts())
    print()
    
    # Save text report
    report_file = os.path.join(OUTPUT_DIR, "level_2_cleaning_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Level 2: Data Cleaning & Feature Engineering Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Raw rows: {len(raw_df):,}\n")
        f.write(f"Cleaned stops saved: {stops_out}\n")
        f.write(f"Cleaned routes saved: {routes_out}\n\n")
        f.write("Route Metrics Summary:\n")
        f.write(routes_df[metrics].describe().round(2).to_string() + "\n\n")
        f.write("Route Categories:\n")
        f.write(routes_df["Route_Category"].value_counts().to_string() + "\n")
        
    print(f"Summary written to: {report_file}")
    print("--------------------------------------------------")
    print("Level 2 completed.\n")


if __name__ == "__main__":
    main()
