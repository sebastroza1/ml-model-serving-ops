from __future__ import annotations

from fastapi.testclient import TestClient

from ml_model_serving_ops.app import create_app
from ml_model_serving_ops.train import train


def test_training_and_api_contract(tmp_path) -> None:
    metadata = train(tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert metadata["test_accuracy"] > 0.7
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        model = client.get("/model")
        assert model.status_code == 200
        assert model.json()["model_version"] == "1.0.0"
        response = client.post("/predict", json={"features": [0.0] * 8})
        assert response.status_code == 200
        assert response.json()["prediction"] in {0, 1}
        assert 0 <= response.json()["probability"] <= 1
        metrics = client.get("/metrics").text
        assert "model_requests_total" in metrics
        assert "model_inference_seconds" in metrics


def test_validation_and_missing_model(tmp_path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422
        response = client.post("/predict", json={"features": [0.0] * 8})
        assert response.status_code == 503
