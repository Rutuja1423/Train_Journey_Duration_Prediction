# Train Route Analysis & Journey Time Prediction
**Sysslan IT Solutions - Data Science Internship Project**

---

## Executive Summary & Project Overview

### Project Title
**Train Route Analysis & Journey Time Prediction System**

### Objective
To analyze real-world railway timetable schedules and develop a data-driven predictive model to estimate train journey durations accurately based on route features (distance and number of stops), facilitating optimal timetable planning, passenger travel planning, and network congestion management.

### Research Question
*How accurately can train journey durations be predicted using route distance and stop counts, and what is the quantitative breakdown between pure moving line speed versus station stop delays across the railway network?*

### Dataset
- **Source**: Timetable Dataset (`Dataset1.csv`)
- **Volume**: **186,074 stop records** spanning **11,113 unique train routes** and **8,147 unique stations**.
- **Attributes**: 12 columns including `Train_No`, `Train_Name`, `SEQ`, `Station_Code`, `Station_Name`, `Arrival_Time`, `Departure_Time`, `Distance`, `Source_Station_Name`, and `Destination_Station_Name`.

---

## Methodology

```
+-----------------------------------+
|  1. Data Collection & Inspection  |  --> 186,074 rows, 11,113 trains, 0 missing values
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  2. Cleaning & Preprocessing      |  --> Modulo time arithmetic for multi-day midnight crossings
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  3. Feature Engineering           |  --> Halt Mins, Leg Speed, Distance/Stop, Route Categories
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  4. Exploratory Data Analysis     |  --> Hub traffic analysis, distance vs duration correlation
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  5. Model Development             |  --> Scikit-Learn Linear Regression (80/20 train/test split)
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  6. Evaluation & System Packaging |  --> MAE 0.98 hrs (58.6 mins), R² 0.9454, TrainJourneyPredictor Class
+-----------------------------------+
```

- **Data Collection and Understanding**:
  - Inspected dataset structure (186,074 rows, 12 features).
  - Confirmed zero missing (`NaN`) values.
  - Identified boundary markers (e.g., `'00:00:00'` arrival at origin stations and departure at terminal stations).

- **Data Cleaning and Preprocessing**:
  - **Multi-day Midnight Handling**: Applied modulo time calculations `(arr_min - prev_dep_min) % 1440` to correctly handle overnight and multi-day train routes.
  - Aggregated stop-level schedule data into route-level summaries (`train_wise_route_table.csv`).

- **Exploratory Data Analysis (EDA)**:
  - Evaluated route extremes (Vivek Express: 4,260 km, 57 stops, 79.7 hrs vs Suburban locals: 2 km, 2 stops, 5 mins).
  - Analyzed station congestion (CSMT Mumbai: 1,027 passing/terminating trains; Kalyan Jn: 828; Thane: 796).
  - Evaluated feature relationships ($r = 0.9820$ between distance and journey duration).

- **Feature Engineering / Selection**:
  - `Halt_Minutes`: Station wait durations.
  - `Leg_Minutes` & `Leg_Distance_KM`: Travel interval metrics between consecutive stations.
  - `Cumulative_Duration_Hours`: Elapsed time from journey origin.
  - `Average_Speed_KMH`: Commercial speed per route (network average: **39.6 km/h**).
  - `Distance_Per_Stop_KM`: Average inter-station distance.
  - `Route_Category`: Distance-based route grouping ($\le 100$ km, 101–500 km, 501–1500 km, $>1500$ km).

- **Model Development**:
  - Trained **Linear Regression** models using `sklearn.linear_model.LinearRegression`.
  - Utilized an **80/20 train/test split** (8,890 training routes, 2,223 testing routes) with `random_state=42`.
  - Validated models using **5-Fold Cross-Validation**.

- **Model Evaluation and Interpretation**:
  - Evaluated performance using **MAE**, **RMSE**, and **$R^2$ Score**.
  - Derived physical model interpretations: effective moving line speed of **57.8 km/h** and average delay of **~6.1 minutes per station stop**.

---

## Key Findings & Actual Results

1. **Strong Linear Distance-Duration Correlation ($r = 0.9820$)**: Route distance is the primary determinant of journey time, with the linear model explaining **94.54% of variance** ($R^2 = 0.9454$, 5-fold CV $R^2 = 0.9820$).
2. **Quantitative Impact of Station Halts (~6.1 mins/stop)**: Each additional intermediate station stop adds approximately **6.1 minutes** to the total travel duration (accounting for braking, platform dwell, and re-acceleration).
3. **Multi-Day Accuracy & Traffic Hubs**: Modulo time preprocessing accurately resolved complex multi-day routes like Vivek Express (4,260 km in 79h 40m actual vs 78h 56m predicted). CSMT Mumbai (1,027 trains) and Kalyan Jn (828 trains) were identified as the busiest station hubs.

---

## Tools & Technologies

$$\text{Python} \,\vert\ ,\text{Pandas} \\vert\, \text{NumPy} \,\vert\, \text{Matplotlib} \,\vert\, \text{Seaborn} \,\vert\, \text{Scikit-Learn} \,\vert\, \text{Joblib}$$

- **Programming Language**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (`LinearRegression`, `train_test_split`, `cross_val_score`)
- **Model Persistence**: Joblib

---

## Outcome & Achievements

- **Predictive System**: Successfully engineered a reusable Python class `TrainJourneyPredictor` that achieves an average absolute prediction error of **0.98 hours (~58.6 minutes)** across 11,113 real-world train routes.
- **Key Learnings**: Mastered handling wrap-around time series/timetable data, building clean data pipelines, conducting feature engineering, validating regression assumptions via residual plots, and creating production-ready Python prediction modules.

---

## Level-by-Level Breakdown & Output Details

### Level 1: Data Overview
- **186,074 rows** and **12 columns**.
- **11,113 unique trains** and **8,147 unique stations**.
- Dataset summary: Average distance per route is **348.7 km** (median: 82 km, max: 4,260 km). Average number of stops per route is **16.7** (median: 15).

### Level 2: Data Cleaning & Feature Engineering
- **Handling Multi-day Journeys**: Leg travel time calculated via modulo arithmetic `(arr_min - prev_dep_min) % 1440` to correctly accumulate elapsed time across multi-day journeys.
- Generated features: `Halt_Minutes`, `Leg_Minutes`, `Cumulative_Duration_Hours`, `Average_Speed_KMH`, `Distance_Per_Stop_KM`, `Route_Category`.

### Level 3: Exploratory Data Analysis (EDA)
- **Extremes**: Longest route is Vivek Express (4,260 km, 57 stops, ~79.7 hrs). Shortest routes are Suburban locals (2 km, 2 stops, 5 mins).
- **Traffic**: CSMT Mumbai (1,027 trains), Kalyan Jn (828), Thane (796), Sealdah (745), Chennai Beach (738), Howrah (699).
- **Correlations**: Distance vs. Duration $r = 0.9820$; Stops vs. Duration $r = 0.5440$.

### Level 4: Visualizations
- Generated 6 visual charts in `visualizations/`:
  1. `fig1_top_routes_distance_duration.png`: Top routes by distance & duration.
  2. `fig2_busiest_stations_traffic.png`: Top busiest stations & origin hubs.
  3. `fig3_distance_vs_duration_regression.png`: Scatter plot with regression line & speed guides.
  4. `fig4_distributions_stops_distance_duration.png`: Distribution histograms.
  5. `fig5_correlation_heatmap.png`: Feature correlation matrix.
  6. `fig6_duration_by_distance_bins.png`: Boxplots across route distance categories.

### Level 5: Model Development & Evaluation
Model comparison on test split (2,223 routes):

| Model | Features | Test MAE | Test RMSE | Test R² | 5-Fold CV R² |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Model 1 | Distance only | 1.12 hrs (67.4 mins) | 2.79 hrs | 0.9383 | 0.9708 |
| **Model 2 (Primary)** | **Distance + Total Stops** | **0.98 hrs (58.6 mins)** | **2.62 hrs** | **0.9454** | **0.9820** |
| Model 3 | Distance + Stops + Halt Mins | 0.71 hrs (42.7 mins) | 1.26 hrs | 0.9874 | 0.9886 |

- **Primary Formula**:
  $$\text{Duration (Hours)} = -0.506 + (0.01729 \times \text{Distance}_{\text{km}}) + (0.1016 \times \text{Stops})$$
- **Physical Interpretation**:
  - Implied line moving speed: $1 / 0.01729 \approx \mathbf{57.8\text{ km/h}}$.
  - Average delay per station stop: $0.1016 \times 60 \approx \mathbf{6.1\text{ minutes}}$.

### Level 6: Final Prediction System
- Packaged into `TrainJourneyPredictor` Python class in `level_6_final_prediction_system.py`.
- Evaluation plots:
  - `fig7_actual_vs_predicted.png`: Scatter plot of actual vs. predicted durations with $\pm 10\%$ corridor.
  - `fig8_residual_analysis.png`: Residual error scatter and normal distribution histogram.
- **Sample Predictions**:
  - *Mumbai to Delhi* (1,386 km, 18 stops) $\rightarrow$ Predicted: **25h 17m** (54.8 km/h).
  - *Howrah to Delhi* (1,447 km, 20 stops) $\rightarrow$ Predicted: **26h 32m** (54.5 km/h).
  - *Chennai to Bengaluru* (359 km, 8 stops) $\rightarrow$ Predicted: **6h 31m** (55.1 km/h).
  - *Vivek Express* (4,260 km, 57 stops) $\rightarrow$ Actual: 79h 40m, Predicted: **78h 56m** (Error: 44.4 mins).

---

## Project Directory Structure

```
final sysslan/
│
├── Dataset1.csv                                  # Input timetable dataset (186,074 rows)
│
├── level_1_data_overview.py                      # Level 1 script
├── level_2_data_cleaning_feature_engineering.py  # Level 2 script
├── level_3_data_exploration.py                   # Level 3 script
├── level_4_visualization_pattern_analysis.py     # Level 4 script
├── level_5_prediction_model_development.py       # Level 5 script
├── level_6_final_prediction_system.py            # Level 6 script
├── run_all_levels.py                             # Master runner script
│
├── outputs/                                      # Data tables and evaluation summaries
│   ├── train_wise_route_table.csv                # Compiled route summary table (11,113 trains)
│   ├── cleaned_train_stops.csv                   # Stop records with halt times & cumulative hours
│   ├── cleaned_train_routes.csv                  # Route records with speeds & categories
│   ├── eda_station_traffic.csv                   # Station traffic counts
│   ├── eda_top_routes.csv                        # Top longest duration routes
│   ├── linear_regression_model.joblib            # Trained Scikit-Learn model artifact
│   ├── level_1_overview_report.txt               # Level 1 text report
│   ├── level_2_cleaning_report.txt               # Level 2 text report
│   ├── level_3_eda_summary.txt                   # Level 3 text report
│   ├── level_5_model_evaluation.txt              # Level 5 model performance summary
│   └── level_6_final_system_report.txt           # Level 6 prediction system summary
│
└── visualizations/                               # Generated visual charts
    ├── fig1_top_routes_distance_duration.png     # Top routes by distance & duration
    ├── fig2_busiest_stations_traffic.png         # Busiest stations & origin hubs
    ├── fig3_distance_vs_duration_regression.png  # Distance vs duration with regression fit
    ├── fig4_distributions_stops_distance_duration.png # Distribution histograms
    ├── fig5_correlation_heatmap.png              # Correlation heatmap
    ├── fig6_duration_by_distance_bins.png        # Boxplot of duration across distance tiers
    ├── fig7_actual_vs_predicted.png              # Actual vs. predicted scatter plot
    └── fig8_residual_analysis.png                # Residual error scatter and distribution
```

---

## How to Run the Project

### 1. Run All Levels
Execute the master runner script:
```bash
python run_all_levels.py
```

### 2. Run Individual Levels
```bash
python level_1_data_overview.py
python level_2_data_cleaning_feature_engineering.py
python level_3_data_exploration.py
python level_4_visualization_pattern_analysis.py
python level_5_prediction_model_development.py
python level_6_final_prediction_system.py
```

### 3. Quick Python Usage Example
```python
from level_6_final_prediction_system import TrainJourneyPredictor

# Initialize predictor
predictor = TrainJourneyPredictor()

# Predict duration by distance and number of stops
result = predictor.predict(distance_km=1386, total_stops=18)
print(result["Formatted"])  # Output: 25h 17m

# Lookup by station names
lookup = predictor.predict_by_stations("KANNIYAKUMAR", "DIBRUGARH")
print(f"Actual: {lookup['Actual_Formatted']}, Predicted: {lookup['Formatted']}")
```
