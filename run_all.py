# run_all.py
# End-to-end pipeline orchestrator: regenerates data, engineering, trains every
# model stage, seeds the database, and prints a final summary.
# Usage:  python run_all.py
import subprocess
import sys
import json
import os


def run_step(label, command):
    print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
    r = subprocess.run(command, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        print(f"[X] FAILED: {label}")
        sys.exit(r.returncode)
    print(f"[+] OK: {label}")


def main():
    # Stage 1-2: generate raw dataset
    run_step("Stage 1-2: Generate dataset", "python generate_data.py")

    # Stage 4: feature engineering
    run_step("Stage 4: Feature engineering", "python feature_engineering.py")

    # Stage 5-9: ML models (Linear, SVR, RF, GB, LightGBM, XGBoost)
    run_step("Stage 5/7/9: Train ML models", "python train_model.py")

    # Stage 6: seasonal baseline
    run_step("Stage 6: Seasonal baseline", "python seasonal_model.py")

    # Stage 8: LSTM deep learning
    run_step("Stage 8: LSTM model", "python deep_learning_model.py")

    # Stage 15: seed database
    run_step("Stage 15: Seed SQLite database", "python database.py")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - MODEL COMPARISON")
    print("=" * 70)
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json") as f:
            m = json.load(f)
        rows = sorted(
            [(k, v["MAE"], v["RMSE"]) for k, v in m["metrics"].items()
             if isinstance(v, dict) and "MAE" in v and isinstance(v["MAE"], (int, float))],
            key=lambda x: x[1],
        )
        print(f"{'Model':<32}{'MAE':>10}{'RMSE':>10}")
        for name, mae, rmse in rows:
            marker = "  <-- BEST" if name == m["best_model"] else ""
            print(f"{name:<32}{mae:>10.3f}{rmse:>10.3f}{marker}")
    print("\nStart the dashboard:  streamlit run app.py")
    print("Start the API:        python api.py")


if __name__ == "__main__":
    main()
