"""
Level 5: Prediction Model Development (Linear Regression)
Sysslan IT Solutions - Data Science Internship

Task:
- Build a Linear Regression model using Scikit-Learn to predict train journey duration.
- Split data into 80% training and 20% testing sets.
- Compare baseline (distance only) vs multi-feature models (distance + stops).
- Evaluate performance using MAE, RMSE, and R^2 score.
- Interpret the physical meaning of the model coefficients.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ROUTES_PATH = os.path.join(OUTPUT_DIR, "cleaned_train_routes.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "linear_regression_model.joblib")


def evaluate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE (Hours)": round(mae, 3),
        "MAE (Mins)": round(mae * 60, 1),
        "RMSE (Hours)": round(rmse, 3),
        "RMSE (Mins)": round(rmse * 60, 1),
        "R2 Score": round(r2, 4)
    }


def train_models(routes_df):
    """Trains baseline and multi-variable linear regression models."""
    y = routes_df["Journey_Duration_Hours"].values
    
    # feature setups to compare
    model_configs = {
        "Model 1 (Distance Only)": ["Total_Distance_KM"],
        "Model 2 (Distance + Stops)": ["Total_Distance_KM", "Total_Stops"],
        "Model 3 (Distance + Stops + Halt Min)": ["Total_Distance_KM", "Total_Stops", "Total_Halt_Minutes"]
    }
    
    # 80/20 train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(routes_df)),
        test_size=0.20,
        random_state=42
    )
    
    results = []
    trained_models = {}
    
    for name, cols in model_configs.items():
        X = routes_df[cols].values
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # fit model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # 5-fold cross validation on training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(lr, X_train, y_train, cv=kf, scoring="r2")
        
        # test set prediction
        y_pred = lr.predict(X_test)
        metrics = evaluate_metrics(y_test, y_pred)
        metrics["Model"] = name
        metrics["Features"] = ", ".join(cols)
        metrics["5-Fold CV R2"] = f"{cv_r2.mean():.4f}"
        
        results.append(metrics)
        trained_models[name] = lr
        
    return results, trained_models, model_configs, (train_idx, test_idx)


def print_coefficient_meaning(model, feature_names):
    """Prints plain-English explanation of linear regression coefficients."""
    intercept = model.intercept_
    coefs = model.coef_
    
    print(f"Intercept: {intercept:.3f} hours ({intercept * 60:.1f} mins)")
    for feat, val in zip(feature_names, coefs):
        if "Distance" in feat:
            speed = 1.0 / val if val > 0 else 0
            print(f"  * {feat}: {val:.5f} hrs/km ({val * 100:.2f} hrs per 100 km / {val * 60:.2f} mins/km)")
            print(f"    -> Implied moving speed: ~{speed:.1f} km/h")
        elif "Stops" in feat:
            print(f"  * {feat}: {val:.4f} hrs/stop ({val * 60:.1f} mins added per intermediate stop)")
            print(f"    -> Represents train slowing down, platform halt time, and accelerating again.")


def main():
    print("--------------------------------------------------")
    print("Level 5: Prediction Model Development")
    print("--------------------------------------------------")
    
    if not os.path.exists(ROUTES_PATH):
        print("Error: Cleaned routes dataset not found. Please run Level 2 first.")
        return
        
    print("Loading route dataset...")
    routes_df = pd.read_csv(ROUTES_PATH)
    print(f"Total routes: {len(routes_df):,}\n")
    
    print("Training models with 80% train / 20% test split...")
    results, models, configs, (train_idx, test_idx) = train_models(routes_df)
    
    res_df = pd.DataFrame(results)
    disp_cols = ["Model", "MAE (Hours)", "MAE (Mins)", "RMSE (Hours)", "R2 Score", "5-Fold CV R2"]
    print("\nModel Performance on Test Set (2,223 routes):")
    print(res_df[disp_cols].to_string(index=False))
    print()
    
    # Primary model is Model 2 (Distance + Stops)
    primary_name = "Model 2 (Distance + Stops)"
    primary_model = models[primary_name]
    primary_features = configs[primary_name]
    
    print("Physical Meaning of Primary Model Coefficients:")
    print_coefficient_meaning(primary_model, primary_features)
    print()
    
    # Save the model
    save_data = {
        "model": primary_model,
        "feature_names": primary_features,
        "intercept": primary_model.intercept_,
        "coefficients": dict(zip(primary_features, primary_model.coef_)),
        "all_models": {k: {"model": m, "features": configs[k]} for k, m in models.items()},
        "train_indices": train_idx,
        "test_indices": test_idx
    }
    joblib.dump(save_data, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # Save report
    report_file = os.path.join(OUTPUT_DIR, "level_5_model_evaluation.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Level 5: Prediction Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train samples: {len(train_idx):,}, Test samples: {len(test_idx):,}\n\n")
        f.write(res_df[disp_cols].to_string(index=False) + "\n\n")
        f.write(f"Primary Formula: Duration (hrs) = {primary_model.intercept_:.3f} + ({primary_model.coef_[0]:.5f} * Distance) + ({primary_model.coef_[1]:.4f} * Stops)\n")
        f.write(f"Implied Moving Speed: ~{1.0 / primary_model.coef_[0]:.1f} km/h\n")
        f.write(f"Average Delay Per Stop: ~{primary_model.coef_[1] * 60:.1f} minutes\n")
        
    print(f"Model evaluation report written to: {report_file}")
    print("--------------------------------------------------")
    print("Level 5 completed.\n")


if __name__ == "__main__":
    main()
