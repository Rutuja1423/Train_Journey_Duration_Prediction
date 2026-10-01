"""
Level 6: Final Prediction System
Sysslan IT Solutions - Data Science Internship

Task:
- Integrate the trained model into a reusable prediction class.
- Plot Actual vs. Predicted journey durations on the test set.
- Plot residual error analysis.
- Test predictions on major Indian railway corridors.
- Provide station-to-station route lookups.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import joblib
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
MODEL_PATH = os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")
ROUTES_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_routes.csv")


class TrainJourneyPredictor:
    """
    Predicts train journey duration using our trained linear regression model.
    """
    def __init__(self, model_file=MODEL_PATH, routes_file=ROUTES_PATH):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at: {model_file}")
            
        data = joblib.load(model_file)
        self.model = data["model"]
        self.features = data["feature_names"]
        self.intercept = data["intercept"]
        self.coefs = data["coefficients"]
        
        self.routes_df = pd.read_csv(routes_file) if os.path.exists(routes_file) else None

    def predict(self, distance_km, total_stops):
        """Returns predicted duration in hours, minutes, and formatted string."""
        X = np.array([[distance_km, total_stops]])
        pred_hrs = float(self.model.predict(X)[0])
        pred_hrs = max(0.05, pred_hrs)
        
        total_mins = pred_hrs * 60.0
        hrs = int(total_mins // 60)
        mins = int(round(total_mins % 60))
        speed = distance_km / pred_hrs if pred_hrs > 0 else 0.0
        
        return {
            "Distance_KM": distance_km,
            "Total_Stops": total_stops,
            "Predicted_Hours": round(pred_hrs, 2),
            "Predicted_Minutes": round(total_mins, 1),
            "Formatted": f"{hrs}h {mins}m",
            "Avg_Speed_KMH": round(speed, 1)
        }

    def predict_by_stations(self, origin, destination):
        """Looks up existing routes between two stations in our database."""
        if self.routes_df is None:
            return {"error": "Routes database not loaded"}
            
        match = self.routes_df[
            (self.routes_df["Source_Station"].str.contains(origin, case=False, na=False)) &
            (self.routes_df["Destination_Station"].str.contains(destination, case=False, na=False))
        ]
        
        if len(match) == 0:
            return {"error": f"No direct route found for {origin} -> {destination}"}
            
        row = match.iloc[0]
        res = self.predict(float(row["Total_Distance_KM"]), int(row["Total_Stops"]))
        res["Train_No"] = int(row["Train_No"])
        res["Source"] = row["Source_Station"]
        res["Destination"] = row["Destination_Station"]
        res["Actual_Hours"] = float(row["Journey_Duration_Hours"])
        res["Actual_Formatted"] = f"{int(row['Journey_Duration_Minutes'] // 60)}h {int(round(row['Journey_Duration_Minutes'] % 60))}m"
        res["Error_Mins"] = round(abs(res["Predicted_Minutes"] - row["Journey_Duration_Minutes"]), 1)
        return res


def plot_actual_vs_predicted(y_true, y_pred):
    """Figure 7: Actual vs predicted scatter plot with 45-degree identity line."""
    plt.figure(figsize=(9, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", s=25, label="Test Routes")
    
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal (y = x)")
    
    # 10% bounds
    grid = np.linspace(0, max_val, 100)
    plt.plot(grid, grid * 1.10, ":", color="gray", alpha=0.7, label="+/- 10% corridor")
    plt.plot(grid, grid * 0.90, ":", color="gray", alpha=0.7)
    
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    plt.title(f"Actual vs. Predicted Journey Duration (Test Set, N={len(y_true):,})\nMAE: {mae:.2f} hrs ({mae*60:.0f} mins) | R²: {r2:.4f}")
    plt.xlabel("Actual Duration (hours)")
    plt.ylabel("Predicted Duration (hours)")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend()
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig7_actual_vs_predicted.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_residuals(y_true, y_pred):
    """Figure 8: Residual error plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuals = y_true - y_pred
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.45, color="#333333", s=20)
    axes[0].axhline(0, color="red", linestyle="--", lw=1.8)
    axes[0].set_title("Residuals vs. Predicted Duration")
    axes[0].set_xlabel("Predicted Duration (hours)")
    axes[0].set_ylabel("Residual (Actual - Predicted) [hours]")
    
    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2b5c8f", bins=40)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Error Distribution")
    axes[1].set_xlabel("Residual (hours)")
    axes[1].set_xlim(-15, 15)
    
    plt.tight_layout()
    out_file = os.path.join(VIZ_DIR, "fig8_residual_analysis.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("--------------------------------------------------")
    print("Level 6: Final Prediction System & Evaluation")
    print("--------------------------------------------------")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ROUTES_PATH):
        print("Error: Required model or routes file not found.")
        return
        
    predictor = TrainJourneyPredictor()
    print(f"Loaded trained model (Features: {predictor.features})\n")
    
    # 1. Evaluation on test set
    saved_meta = joblib.load(MODEL_PATH)
    test_idx = saved_meta["test_indices"]
    routes = predictor.routes_df
    
    X_test = routes.loc[test_idx, predictor.features].values
    y_test = routes.loc[test_idx, "Journey_Duration_Hours"].values
    y_pred = predictor.model.predict(X_test)
    
    print("Generating evaluation plots...")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    print()
    
    # 2. Live Corridor Predictions
    print("Testing predictions on popular routes:")
    test_cases = [
        ("Mumbai to Delhi", 1386, 18),
        ("Howrah to Delhi (Rajdhani corridor)", 1447, 20),
        ("Chennai to Bengaluru", 359, 8),
        ("Vivek Express (Kanyakumari to Dibrugarh)", 4260, 57),
        ("Mumbai to Pune", 192, 7),
        ("Local shuttle (Karjat to Palasdari)", 2, 2)
    ]
    
    demo_records = []
    for name, d, s in test_cases:
        res = predictor.predict(d, s)
        demo_records.append({
            "Route": name,
            "Distance (km)": f"{d:,}",
            "Stops": s,
            "Predicted Duration": res["Formatted"],
            "Est. Speed (km/h)": res["Avg_Speed_KMH"]
        })
    print(pd.DataFrame(demo_records).to_string(index=False))
    print()
    
    # 3. Route Lookup from dataset
    print("Testing station lookup from historical database:")
    lookups = [
        ("MADGOAN", "CHHATRAPATI"),
        ("AURANGABAD", "VARANASI"),
        ("KANNIYAKUMAR", "DIBRUGARH")
    ]
    lookup_records = []
    for orig, dest in lookups:
        res = predictor.predict_by_stations(orig, dest)
        if "error" not in res:
            lookup_records.append({
                "Train": res["Train_No"],
                "From -> To": f"{res['Source']} -> {res['Destination']}",
                "Distance (km)": f"{res['Distance_KM']:,}",
                "Stops": res["Total_Stops"],
                "Actual": res["Actual_Formatted"],
                "Predicted": res["Formatted"],
                "Error": f"{res['Error_Mins']} mins"
            })
    print(pd.DataFrame(lookup_records).to_string(index=False))
    print()
    
    # Save text report
    report_file = os.path.join(OUTPUT_DIR, "level_6_final_system_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Level 6: Final Prediction System Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Equation: Duration (hours) = {predictor.intercept:.3f} + ({predictor.coefs['Total_Distance_KM']:.5f} * Distance) + ({predictor.coefs['Total_Stops']:.4f} * Stops)\n\n")
        f.write("Sample Corridor Predictions:\n")
        f.write(pd.DataFrame(demo_records).to_string(index=False) + "\n\n")
        f.write("Database Station Lookup Validation:\n")
        f.write(pd.DataFrame(lookup_records).to_string(index=False) + "\n")
        
    print(f"Summary saved to: {report_file}")
    print("--------------------------------------------------")
    print("Level 6 completed.\n")


if __name__ == "__main__":
    main()
