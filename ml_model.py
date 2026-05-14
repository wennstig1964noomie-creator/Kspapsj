"""Train and serve the gradient-boosted signal model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import MODEL_PATH, PREDICTION_HORIZON_BARS
from features import FEATURE_COLUMNS, build_features, make_training_frame

log = logging.getLogger(__name__)


@dataclass
class SignalModel:
    booster: lgb.Booster
    scaler: StandardScaler
    horizon: int
    auc: float

    def predict_proba(self, bars: pd.DataFrame) -> float:
        feats = build_features(bars).iloc[[-1]][FEATURE_COLUMNS]
        if feats.isna().any(axis=None):
            return float("nan")
        x = self.scaler.transform(feats.values)
        return float(self.booster.predict(x)[0])

    def save(self, path: Path = MODEL_PATH) -> None:
        joblib.dump(
            {"booster": self.booster, "scaler": self.scaler, "horizon": self.horizon, "auc": self.auc},
            path,
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SignalModel | None":
        if not Path(path).exists():
            return None
        data = joblib.load(path)
        return cls(**data)


def train(bars_by_symbol: dict[str, pd.DataFrame], horizon: int = PREDICTION_HORIZON_BARS) -> SignalModel:
    """Train one cross-sectional model across all symbols.

    Uses a time-ordered split so we don't leak future data into validation.
    """
    frames = []
    for sym, bars in bars_by_symbol.items():
        if len(bars) < 200:
            continue
        f = make_training_frame(bars, horizon)
        f["symbol"] = sym
        frames.append(f)
    if not frames:
        raise RuntimeError("No symbols had enough data to train on.")

    full = pd.concat(frames).sort_index()
    split = int(len(full) * 0.8)
    train_df, val_df = full.iloc[:split], full.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    X_val = scaler.transform(val_df[FEATURE_COLUMNS].values)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 200,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    preds = booster.predict(X_val)
    auc = float(roc_auc_score(y_val, preds))
    log.info("Trained signal model. Validation AUC=%.4f", auc)
    return SignalModel(booster=booster, scaler=scaler, horizon=horizon, auc=auc)
