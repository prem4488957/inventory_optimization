# deep_learning_model.py
# Stage 8 - LSTM deep learning forecaster (PyTorch).
# A per-product univariate LSTM that learns from rolling windows of daily sales history,
# then is evaluated on the same time-based test split used by the ML models.
import pandas as pd
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

WINDOW = 30        # look-back days
HIDDEN = 64
LAYERS = 2
EPOCHS = 40
LR = 1e-3
BATCH = 32


class LSTMRegressor(nn.Module):
    def __init__(self, window, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # take last timestep
        return self.fc(out).squeeze(-1)


def make_sequences(values, window):
    xs, ys = [], []
    for i in range(len(values) - window):
        xs.append(values[i:i + window])
        ys.append(values[i + window])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_one_product(series_train, series_test, product_name):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(series_train.reshape(-1, 1)).ravel()
    scaled_test = scaler.transform(series_test.reshape(-1, 1)).ravel()

    X, y = make_sequences(scaled_train, WINDOW)
    X = X.reshape(X.shape[0], WINDOW, 1)

    tensor_x = torch.tensor(X)
    tensor_y = torch.tensor(y)
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=BATCH, shuffle=True)

    model = LSTMRegressor(WINDOW, HIDDEN, LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    # Predict on the test split one day at a time using a sliding window
    model.eval()
    predictions = []
    # start with the last WINDOW scaled values of the training series
    context = list(scaled_train[-WINDOW:])
    with torch.no_grad():
        for i in range(len(scaled_test)):
            x = torch.tensor(np.array(context[-WINDOW:], dtype=np.float32)).reshape(1, WINDOW, 1)
            pred_scaled = model(x).item()
            pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(max(0.0, float(pred)))
            context.append(scaled_test[i])   # feed true history to the context window

    return model, scaler, np.array(predictions)


def train():
    print("[+] Loading engineered data...")
    df = pd.read_csv("featured_sales_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    split_date = "2025-10-01"
    test_df = df[df["date"] >= split_date]

    all_y, all_p = [], []
    saved = {}
    per_product = {}

    for product_id, g in df.groupby("product_id"):
        g = g.sort_values("date").reset_index(drop=True)
        train_series = g.loc[g["date"] < split_date, "units_sold"].astype(float).to_numpy()
        test_series = g.loc[g["date"] >= split_date, "units_sold"].astype(float).to_numpy()

        print(f"[+] Training LSTM for {g['product_name'].iloc[0]} ...")
        model, scaler, preds = train_one_product(train_series, test_series, g["product_name"].iloc[0])

        mae = mean_absolute_error(test_series, preds)
        rmse = root_mean_squared_error(test_series, preds)

        per_product[product_id] = {"MAE": round(float(mae), 3), "RMSE": round(float(rmse), 3), "name": g["product_name"].iloc[0]}
        saved[product_id] = {"model": model, "scaler": scaler, "window": WINDOW}
        all_y.extend(test_series.tolist())
        all_p.extend(preds.tolist())
        print(f"   * MAE={mae:.2f} | RMSE={rmse:.2f}")

    with open("lstm_models.pkl", "wb") as f:
        pickle.dump(saved, f)
    print("[+] Saved LSTM models to 'lstm_models.pkl'")

    overall_mae = mean_absolute_error(all_y, all_p)
    overall_rmse = root_mean_squared_error(all_y, all_p)
    print(f"\n[+] Overall LSTM: MAE={overall_mae:.3f} | RMSE={overall_rmse:.3f}")

    with open("model_metrics.json", "r") as f:
        metrics_report = json.load(f)

    metrics_report["metrics"]["LSTM (PyTorch)"] = {
        "MAE": round(float(overall_mae), 3),
        "RMSE": round(float(overall_rmse), 3),
        "per_product": per_product,
    }

    scalar = {k: v for k, v in metrics_report["metrics"].items()
              if isinstance(v, dict) and "MAE" in v and isinstance(v["MAE"], (int, float))}
    rankings = sorted(scalar.items(), key=lambda kv: kv[1]["MAE"])
    metrics_report["best_model"] = rankings[0][0]

    with open("model_metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print("[+] Updated 'model_metrics.json' with LSTM results.")
    print(f"[+] Best model overall is now: {rankings[0][0]}\n")


if __name__ == "__main__":
    train()
