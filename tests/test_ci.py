import pytest
from fastapi.testclient import TestClient

from api.main import app, FEATURES_ORDER


# ---- Fake DB pour éviter PostgreSQL en CI ----
class FakeSession:
    def execute(self, *args, **kwargs):
        return None

    def commit(self):
        return None

    def close(self):
        return None


@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """Remplace SessionLocal par une session fake"""
    monkeypatch.setattr("api.main.SessionLocal", lambda: FakeSession())


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def features_payload():
    return {"features": {f: 0.0 for f in FEATURES_ORDER}}


def test_metadata_endpoint(client):
    r = client.get("/metadata")
    assert r.status_code == 200

    data = r.json()
    assert "features_order" in data
    assert "threshold" in data


def test_predict_endpoint(client, features_payload):
    r = client.post("/predict", json=features_payload)
    assert r.status_code == 200

    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)


def test_predict_bad_payload(client):
    r = client.post("/predict", json={"features": {"Age": 10}})
    assert r.status_code == 422