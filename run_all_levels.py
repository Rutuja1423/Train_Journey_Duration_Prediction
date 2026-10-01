"""
Master script to run all levels sequentially (Level 1 to Level 6).
Sysslan IT Solutions - Data Science Internship
"""

import os
import sys
import subprocess
import time

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_CMD = sys.executable

SCRIPTS = [
    ("Level 1: Data Overview", "level_1_data_overview.py"),
    ("Level 2: Data Cleaning & Feature Engineering", "level_2_data_cleaning_feature_engineering.py"),
    ("Level 3: Exploratory Data Analysis", "level_3_data_exploration.py"),
    ("Level 4: Visualization", "level_4_visualization_pattern_analysis.py"),
    ("Level 5: Model Training", "level_5_prediction_model_development.py"),
    ("Level 6: Final Prediction System", "level_6_final_prediction_system.py"),
]

def main():
    print("==================================================")
    print("Train Journey Duration Prediction Pipeline")
    print("Sysslan IT Solutions - Data Science Internship")
    print("==================================================\n")
    
    start_total = time.time()
    
    for label, script in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script)
        print(f">>> Running {label} ({script})...")
        t0 = time.time()
        ret = subprocess.run([PYTHON_CMD, script_path], cwd=BASE_DIR)
        if ret.returncode != 0:
            print(f"Error executing {script}. Terminating.")
            sys.exit(ret.returncode)
        dt = time.time() - t0
        print(f"Finished {label} in {dt:.1f}s.\n")
        
    total_time = time.time() - start_total
    print("==================================================")
    print(f"All 6 levels completed successfully in {total_time:.1f}s.")
    print("==================================================")

if __name__ == "__main__":
    main()
