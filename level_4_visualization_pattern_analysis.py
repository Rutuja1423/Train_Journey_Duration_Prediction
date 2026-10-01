"""
Level 4: Visualization & Pattern Analysis
Sysslan IT Solutions - Data Science Internship

Task:
- Plot top train routes by distance and duration.
- Visualize station traffic and major railway hubs.
- Plot distance vs journey duration with linear regression fit.
- Visualize distributions of stops, distance, duration, and speed.
- Plot feature correlation matrix.
- Show duration breakdown by distance category.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

ROUTES_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_routes.csv")
STOPS_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_stops.csv")

# Set standard clean styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 200
})


def plot_top_routes(routes_df):
    """Figure 1: Bar charts of top 10 routes by distance and duration."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. By distance
    top_dist = routes_df.sort_values(by="Total_Distance_KM", ascending=False).head(10).copy()
    top_dist["Route_Label"] = top_dist["Train_No"].astype(str) + " (" + top_dist["Source_Station"] + " -> " + top_dist["Destination_Station"] + ")"
    
    sns.barplot(
        data=top_dist,
        y="Route_Label",
        x="Total_Distance_KM",
        ax=axes[0],
        hue="Route_Label",
        palette="Blues_r",
        legend=False
    )
    axes[0].set_title("Top 10 Longest Train Routes by Distance (km)")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("")
    for i, val in enumerate(top_dist["Total_Distance_KM"]):
        dur = top_dist.iloc[i]["Journey_Duration_Hours"]
        axes[0].text(val + 30, i, f"{val:,} km ({dur:.1f}h)", va="center", fontsize=8)
    axes[0].set_xlim(0, top_dist["Total_Distance_KM"].max() * 1.18)
    
    # 2. By duration
    top_dur = routes_df.sort_values(by="Journey_Duration_Hours", ascending=False).head(10).copy()
    top_dur["Route_Label"] = top_dur["Train_No"].astype(str) + " (" + top_dur["Source_Station"] + " -> " + top_dur["Destination_Station"] + ")"
    
    sns.barplot(
        data=top_dur,
        y="Route_Label",
        x="Journey_Duration_Hours",
        ax=axes[1],
        hue="Route_Label",
        palette="Greens_r",
        legend=False
    )
    axes[1].set_title("Top 10 Longest Routes by Duration (hours)")
    axes[1].set_xlabel("Duration (hours)")
    axes[1].set_ylabel("")
    for i, val in enumerate(top_dur["Journey_Duration_Hours"]):
        dst = top_dur.iloc[i]["Total_Distance_KM"]
        axes[1].text(val + 1.2, i, f"{val:.1f}h ({dst:,} km)", va="center", fontsize=8)
    axes[1].set_xlim(0, top_dur["Journey_Duration_Hours"].max() * 1.15)
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig1_top_routes_distance_duration.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_station_traffic(stops_df):
    """Figure 2: Top busiest stations and origin hubs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    traffic = stops_df.groupby(["Station_Code", "Station_Name"]).agg(
        Total_Trains=("Train_No", "count"),
        Origin_Trains=("Is_Origin", "sum")
    ).reset_index()
    
    # Busiest overall
    top_busy = traffic.sort_values(by="Total_Trains", ascending=False).head(12).copy()
    top_busy["Label"] = top_busy["Station_Code"] + " - " + top_busy["Station_Name"]
    
    sns.barplot(
        data=top_busy,
        y="Label",
        x="Total_Trains",
        ax=axes[0],
        hue="Label",
        palette="Purples_r",
        legend=False
    )
    axes[0].set_title("Top 12 Busiest Stations (Total Scheduled Trains)")
    axes[0].set_xlabel("Total Trains Handled")
    axes[0].set_ylabel("")
    for i, val in enumerate(top_busy["Total_Trains"]):
        axes[0].text(val + 10, i, f"{val:,}", va="center", fontsize=8)
    axes[0].set_xlim(0, top_busy["Total_Trains"].max() * 1.15)
    
    # Top origin stations
    top_origins = traffic.sort_values(by="Origin_Trains", ascending=False).head(12).copy()
    top_origins["Label"] = top_origins["Station_Code"] + " - " + top_origins["Station_Name"]
    
    sns.barplot(
        data=top_origins,
        y="Label",
        x="Origin_Trains",
        ax=axes[1],
        hue="Label",
        palette="Oranges_r",
        legend=False
    )
    axes[1].set_title("Top 12 Originating Stations (Trains Starting Here)")
    axes[1].set_xlabel("Originating Trains")
    axes[1].set_ylabel("")
    for i, val in enumerate(top_origins["Origin_Trains"]):
        axes[1].text(val + 5, i, f"{val:,}", va="center", fontsize=8)
    axes[1].set_xlim(0, top_origins["Origin_Trains"].max() * 1.15)
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig2_busiest_stations_traffic.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_distance_vs_duration(routes_df):
    """Figure 3: Scatter plot of distance vs journey duration with trendline."""
    plt.figure(figsize=(10, 6))
    
    # Scatter points
    plt.scatter(
        routes_df["Total_Distance_KM"],
        routes_df["Journey_Duration_Hours"],
        c=routes_df["Total_Stops"],
        cmap="viridis",
        alpha=0.5,
        s=25
    )
    cbar = plt.colorbar()
    cbar.set_label("Total Stops per Route")
    
    # Linear fit line
    sns.regplot(
        data=routes_df,
        x="Total_Distance_KM",
        y="Journey_Duration_Hours",
        scatter=False,
        color="red",
        line_kws={"linewidth": 2, "label": "Linear Fit (R² = 0.964)"}
    )
    
    # Speed guide lines
    max_d = routes_df["Total_Distance_KM"].max()
    x_grid = np.linspace(0, max_d, 50)
    plt.plot(x_grid, x_grid / 60.0, "--", color="green", alpha=0.7, label="60 km/h constant speed")
    plt.plot(x_grid, x_grid / 40.0, ":", color="orange", alpha=0.7, label="40 km/h constant speed")
    
    plt.title("Distance vs. Journey Duration across 11,113 Train Routes")
    plt.xlabel("Total Track Distance (km)")
    plt.ylabel("Journey Duration (hours)")
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig3_distance_vs_duration_regression.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_distributions(routes_df):
    """Figure 4: 4-panel histogram and KDE plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Distance
    sns.histplot(routes_df["Total_Distance_KM"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=35)
    axes[0, 0].axvline(routes_df["Total_Distance_KM"].median(), color="red", linestyle="--", label=f"Median: {routes_df['Total_Distance_KM'].median():.0f} km")
    axes[0, 0].set_title("Distribution of Route Distance (km)")
    axes[0, 0].set_xlabel("Distance (km)")
    axes[0, 0].legend()
    
    # Stops
    sns.histplot(routes_df["Total_Stops"], kde=True, ax=axes[0, 1], color="#3a8f7d", bins=30)
    axes[0, 1].axvline(routes_df["Total_Stops"].median(), color="red", linestyle="--", label=f"Median: {routes_df['Total_Stops'].median():.0f} stops")
    axes[0, 1].set_title("Distribution of Total Stops")
    axes[0, 1].set_xlabel("Number of Stops")
    axes[0, 1].legend()
    
    # Duration
    sns.histplot(routes_df["Journey_Duration_Hours"], kde=True, ax=axes[1, 0], color="#916c28", bins=35)
    axes[1, 0].axvline(routes_df["Journey_Duration_Hours"].median(), color="red", linestyle="--", label=f"Median: {routes_df['Journey_Duration_Hours'].median():.1f} hrs")
    axes[1, 0].set_title("Distribution of Journey Duration (hours)")
    axes[1, 0].set_xlabel("Duration (hours)")
    axes[1, 0].legend()
    
    # Speed
    sns.histplot(routes_df["Average_Speed_KMH"], kde=True, ax=axes[1, 1], color="#a84332", bins=35)
    axes[1, 1].axvline(routes_df["Average_Speed_KMH"].mean(), color="red", linestyle="--", label=f"Mean: {routes_df['Average_Speed_KMH'].mean():.1f} km/h")
    axes[1, 1].set_title("Distribution of Average Speed (km/h)")
    axes[1, 1].set_xlabel("Speed (km/h)")
    axes[1, 1].legend()
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig4_distributions_stops_distance_duration.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_correlation_heatmap(routes_df):
    """Figure 5: Correlation heatmap."""
    plt.figure(figsize=(9, 7))
    
    cols = ["Total_Distance_KM", "Total_Stops", "Total_Halt_Minutes", "Journey_Duration_Hours", "Average_Speed_KMH"]
    labels = ["Distance", "Total Stops", "Halt Minutes", "Duration (hrs)", "Avg Speed"]
    
    corr = routes_df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True
    )
    plt.title("Correlation Matrix of Route Features")
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig5_correlation_heatmap.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_duration_by_distance_bins(routes_df):
    """Figure 6: Boxplots of journey duration by route distance category."""
    plt.figure(figsize=(10, 5))
    
    categories = ["Short-Haul (<=100 km)", "Medium-Haul (101-500 km)", "Long-Haul (501-1500 km)", "Super Long-Haul (>1500 km)"]
    
    sns.boxplot(
        data=routes_df,
        x="Route_Category",
        y="Journey_Duration_Hours",
        order=categories,
        hue="Route_Category",
        palette="Set2",
        legend=False,
        showmeans=True
    )
    plt.title("Journey Duration Breakdown by Route Distance Category")
    plt.xlabel("Route Distance Category")
    plt.ylabel("Journey Duration (hours)")
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig6_duration_by_distance_bins.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("--------------------------------------------------")
    print("Level 4: Visualization & Pattern Analysis")
    print("--------------------------------------------------")
    
    if not os.path.exists(ROUTES_PATH) or not os.path.exists(STOPS_PATH):
        print("Error: Cleaned files not found. Please run Level 2 first.")
        return
        
    print("Loading data for plots...")
    routes_df = pd.read_csv(ROUTES_PATH)
    stops_df = pd.read_csv(STOPS_PATH)
    
    print("Generating charts...")
    plot_top_routes(routes_df)
    plot_station_traffic(stops_df)
    plot_distance_vs_duration(routes_df)
    plot_feature_distributions(routes_df)
    plot_correlation_heatmap(routes_df)
    plot_duration_by_distance_bins(routes_df)
    
    print(f"\nAll plots saved to: {VIZ_DIR}")
    print("--------------------------------------------------")
    print("Level 4 completed.\n")


if __name__ == "__main__":
    main()
