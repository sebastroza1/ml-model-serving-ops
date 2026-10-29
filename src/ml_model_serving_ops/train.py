from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COUNT = 8
SEED = 42


def train(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features, target = make_classification(
        n_samples=5000,
        n_features=FEATURE_COUNT,
        n_informative=6,
        n_redundant=2,
        class_sep=1.2,
        weights=[0.7, 0.3],
        random_state=SEED,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED))
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    metadata = {
        "model_version": "1.0.0",
        "model_type": "standard-scaler + logistic-regression",
        "dataset": "sklearn.make_classification synthetic fixture",
        "synthetic_data": True,
        "training_rows": len(train_x),
        "test_rows": len(test_x),
        "feature_count": FEATURE_COUNT,
        "seed": SEED,
        "test_accuracy": float(accuracy_score(test_y, predictions)),
        "test_f1": float(f1_score(test_y, predictions)),
    }
    joblib.dump(model, output / "model.joblib")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the serving model artifact.")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
