"""
Level 3: Exploratory Data Analysis (EDA)
Sysslan IT Solutions - Data Science Internship

Task:
- Compare journey durations across different train routes (longest, shortest, fast, slow).
- Find high-traffic and low-traffic stations (busiest junctions and major originating hubs).
- Analyze the relationship between distance, stops, and journey duration.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STOPS_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_stops.csv")
ROUTES_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_routes.csv")


def explore_routes(routes_df):
    """Identifies extremes in duration and speed across routes."""
    # Top 5 longest duration routes
    cols = ["Train_No", "Source_Station", "Destination_Station", "Total_Distance_KM", "Total_Stops", "Journey_Duration_Hours", "Average_Speed_KMH"]
    longest = routes_df.sort_values(by="Journey_Duration_Hours", ascending=False).head(5)[cols]
    
    # Top 5 shortest duration routes
    shortest = routes_df.sort_values(by="Journey_Duration_Hours", ascending=True).head(5)[cols]
    
    # Fastest routes with meaningful distance (at least 50 km)
    meaningful = routes_df[routes_df["Total_Distance_KM"] >= 50]
    fastest = meaningful.sort_values(by="Average_Speed_KMH", ascending=False).head(5)[cols]
    
    # Breakdown by route distance category
    cat_stats = routes_df.groupby("Route_Category", observed=False).agg(
        Route_Count=("Train_No", "count"),
        Mean_Distance=("Total_Distance_KM", "mean"),
        Mean_Duration=("Journey_Duration_Hours", "mean"),
        Median_Duration=("Journey_Duration_Hours", "median"),
        Mean_Speed=("Average_Speed_KMH", "mean")
    ).round(2)
    
    return longest, shortest, fastest, cat_stats


def explore_station_traffic(stops_df):
    """Counts train stops per station to identify major hubs and quiet stops."""
    traffic = stops_df.groupby(["Station_Code", "Station_Name"]).agg(
        Total_Trains=("Train_No", "count"),
        Origin_Trains=("Is_Origin", "sum"),
        Terminus_Trains=("Is_Destination", "sum")
    ).reset_index()
    
    traffic["Intermediate_Pass_Through"] = traffic["Total_Trains"] - (traffic["Origin_Trains"] + traffic["Terminus_Trains"])
    
    busiest = traffic.sort_values(by="Total_Trains", ascending=False).head(10)
    top_origins = traffic.sort_values(by="Origin_Trains", ascending=False).head(10)
    
    # distribution into traffic tiers
    conditions = [
        (traffic["Total_Trains"] >= 100),
        (traffic["Total_Trains"] >= 50) & (traffic["Total_Trains"] < 100),
        (traffic["Total_Trains"] >= 20) & (traffic["Total_Trains"] < 50),
        (traffic["Total_Trains"] >= 5) & (traffic["Total_Trains"] < 20),
        (traffic["Total_Trains"] < 5)
    ]
    labels = ["Major Junction (>=100 trains)", "Busy Station (50-99 trains)", 
              "Medium Station (20-49 trains)", "Minor Station (5-19 trains)", "Halt / Low Traffic (<5 trains)"]
    traffic["Traffic_Tier"] = pd.Categorical(np.select(conditions, labels, default="Other"), categories=labels, ordered=True)
    tier_counts = traffic["Traffic_Tier"].value_counts()
    
    return busiest, top_origins, tier_counts, traffic


def analyze_correlations(routes_df):
    """Calculates correlation and simple regression slope between distance and duration."""
    cols = ["Total_Distance_KM", "Total_Stops", "Total_Halt_Minutes", "Journey_Duration_Hours", "Average_Speed_KMH"]
    corr_matrix = routes_df[cols].corr().round(4)
    
    # linear regression fit for distance vs duration
    res_dist = stats.linregress(routes_df["Total_Distance_KM"], routes_df["Journey_Duration_Hours"])
    res_stops = stats.linregress(routes_df["Total_Stops"], routes_df["Journey_Duration_Hours"])
    
    return corr_matrix, res_dist, res_stops


def main():
    print("--------------------------------------------------")
    print("Level 3: Exploratory Data Analysis (EDA)")
    print("--------------------------------------------------")
    
    if not os.path.exists(ROUTES_PATH) or not os.path.exists(STOPS_PATH):
        print("Error: Cleaned files not found. Please run Level 2 first.")
        return
        
    print("Loading cleaned datasets...")
    routes_df = pd.read_csv(ROUTES_PATH)
    stops_df = pd.read_csv(STOPS_PATH)
    print(f"Loaded {len(routes_df):,} routes and {len(stops_df):,} stops.\n")
    
    # 1. Routes
    print("--- 1. Route Comparison Highlights ---")
    longest, shortest, fastest, cat_stats = explore_routes(routes_df)
    
    print("\nTop 5 Longest Routes by Duration:")
    print(longest.to_string(index=False))
    
    print("\nTop 5 Shortest Routes by Duration:")
    print(shortest.to_string(index=False))
    
    print("\nTop 5 Fastest Routes (>=50 km):")
    print(fastest.to_string(index=False))
    
    print("\nRoute Category Comparison:")
    print(cat_stats)
    print()
    
    # 2. Stations
    print("--- 2. Station Traffic Analysis ---")
    busiest, top_origins, tier_counts, traffic_df = explore_station_traffic(stops_df)
    
    print("\nTop 10 Busiest Stations (Total Trains):")
    print(busiest[["Station_Code", "Station_Name", "Total_Trains", "Origin_Trains", "Terminus_Trains"]].to_string(index=False))
    
    print("\nTop 10 Originating Stations:")
    print(top_origins[["Station_Code", "Station_Name", "Origin_Trains", "Total_Trains"]].to_string(index=False))
    
    print("\nStation Traffic Distribution:")
    for tier, count in tier_counts.items():
        pct = (count / len(traffic_df)) * 100
        print(f"  * {tier:35s}: {count:5,d} ({pct:5.1f}%)")
    print()
    
    # save tables
    traffic_out = os.path.join(OUTPUT_DIR, "eda_station_traffic.csv")
    traffic_df.to_csv(traffic_out, index=False)
    top_routes_out = os.path.join(OUTPUT_DIR, "eda_top_routes.csv")
    longest.to_csv(top_routes_out, index=False)
    
    # 3. Correlations
    print("--- 3. Distance-Duration Relationships ---")
    corr_matrix, res_dist, res_stops = analyze_correlations(routes_df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    print()
    
    print(f"Pearson correlation between Distance and Duration: {corr_matrix.loc['Total_Distance_KM', 'Journey_Duration_Hours']:.4f}")
    print(f"Distance slope: {res_dist.slope * 100:.3f} hours per 100 km (~{res_dist.slope * 60:.2f} mins/km)")
    print(f"Distance R-squared: {res_dist.rvalue**2:.4f}")
    print(f"Stops slope: {res_stops.slope:.3f} hours per stop (~{res_stops.slope * 60:.1f} mins/stop)")
    print()
    
    # Save report
    report_file = os.path.join(OUTPUT_DIR, "level_3_eda_summary.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Level 3: Exploratory Data Analysis (EDA) Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top Longest Routes:\n")
        f.write(longest.to_string(index=False) + "\n\n")
        f.write("Route Category Summary:\n")
        f.write(cat_stats.to_string() + "\n\n")
        f.write("Top 10 Busiest Stations:\n")
        f.write(busiest[["Station_Code", "Station_Name", "Total_Trains", "Origin_Trains", "Terminus_Trains"]].to_string(index=False) + "\n\n")
        f.write("Station Traffic Breakdown:\n")
        f.write(tier_counts.to_string() + "\n\n")
        f.write("Correlation Matrix:\n")
        f.write(corr_matrix.to_string() + "\n\n")
        f.write(f"Distance - Duration Correlation: {corr_matrix.loc['Total_Distance_KM', 'Journey_Duration_Hours']:.4f}\n")
        f.write(f"Linear Fit Slope: {res_dist.slope * 100:.3f} hrs / 100 km, R^2 = {res_dist.rvalue**2:.4f}\n")
        
    print(f"EDA summary report saved to: {report_file}")
    print("--------------------------------------------------")
    print("Level 3 completed.\n")


if __name__ == "__main__":
    main()
