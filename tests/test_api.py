from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_predict_valid_request():
    payload = {
        "f0": 0,
        "f1": 0,
        "f2": 0,
        "f3": 0,
        "f4": 0,
        "f5": 0,
        "f6": 0,
        "f7": 0,
        "f8": 0,
        "f9": 0,
        "f10": 0,
        "f11": 0,
        "treatment": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_uplift" in data
    assert "recommendation" in data
    assert "model_name" in data
    assert "model_version" in data


def test_predict_missing_feature():
    payload = {
        "f0": 0,
        "f1": 0,
        "f2": 0,
        "f3": 0,
        "f4": 0,
        "f5": 0,
        "f6": 0,
        "f7": 0,
        "f8": 0,
        "f9": 0,
        "f10": 0,
        "treatment": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_multiple_requests():
    payloads = [
        {
            "f0": 0,
            "f1": 0,
            "f2": 0,
            "f3": 0,
            "f4": 0,
            "f5": 0,
            "f6": 0,
            "f7": 0,
            "f8": 0,
            "f9": 0,
            "f10": 0,
            "f11": 0,
            "treatment": 0,
        },
        {
            "f0": 1,
            "f1": 1,
            "f2": 1,
            "f3": 1,
            "f4": 1,
            "f5": 1,
            "f6": 1,
            "f7": 1,
            "f8": 1,
            "f9": 1,
            "f10": 1,
            "f11": 1,
            "treatment": 1,
        },
    ]

    for payload in payloads:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert isinstance(data["predicted_uplift"], float)