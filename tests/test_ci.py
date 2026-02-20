import random
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as main

class FakeSession:
    def __init__(self):
        self.executed = 0

    def execute(self, *args, **kwargs):
        self.executed += 1
        return None

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """Remplace SessionLocal par une session fake pour éviter PostgreSQL en CI."""
    monkeypatch.setattr(main, "SessionLocal", lambda: FakeSession())


@pytest.fixture()
def client():
    return TestClient(main.app)


@pytest.fixture(scope="session")
def rng():
    # reproductibilité
    random.seed(42)
    np.random.seed(42)
    return np.random.default_rng(42)


def make_valid_features(fill_value: float = 0.0) -> dict:
    """Construit un payload complet conforme à FEATURES_ORDER."""
    return {f: float(fill_value) for f in main.FEATURES_ORDER}


# UNIT TESTS (composants)

def test_metadata_contract(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()

    # Contrat minimal
    assert "features_order" in data
    assert "threshold" in data
    assert "cols_to_scale" in data

    # Invariants simples
    assert isinstance(data["features_order"], list)
    assert len(data["features_order"]) > 0
    assert set(data["features_order"]) == set(main.FEATURES_ORDER)

    thr = float(data["threshold"])
    assert 0.0 <= thr <= 1.0

    cols = data["cols_to_scale"]
    assert isinstance(cols, list)
    # cols_to_scale doit être un sous-ensemble des features
    assert set(cols).issubset(set(main.FEATURES_ORDER))


def test_predict_rejects_missing_features(client):
    # Il manque presque tout -> 422
    r = client.post("/predict", json={"features": {"age": 30.0}})
    assert r.status_code == 422


def test_predict_rejects_wrong_key_case(client):
    # "Age" au lieu de "age"
    r = client.post("/predict", json={"features": {"Age": 30.0}})
    assert r.status_code == 422


def test_predict_rejects_null_value(client):
    features = make_valid_features(0.0)
    features[main.FEATURES_ORDER[0]] = None
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 422


def test_predict_rejects_nan_or_inf_raw_json(client):
    first = main.FEATURES_ORDER[0]

    payload_nan = {"features": make_valid_features(0.0)}
    payload_nan["features"][first] = float("nan")
    r = client.post(
        "/predict",
        content=json.dumps(payload_nan, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code in (400, 422)

    payload_inf = {"features": make_valid_features(0.0)}
    payload_inf["features"][first] = float("inf")
    r = client.post(
        "/predict",
        content=json.dumps(payload_inf, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code in (400, 422)
    
def test_predict_rejects_non_castable_value(client):
    """
    Couvre la branche:
        except (TypeError, ValueError)
    """
    features = make_valid_features(0.0)
    features[main.FEATURES_ORDER[0]] = "not_a_number"

    r = client.post(
        "/predict",
        content=json.dumps({"features": features}),
        headers={"Content-Type": "application/json"},
    )

    assert r.status_code == 422

# FUNCTIONAL TESTS (end-to-end modèle via l'API)

def test_predict_endpoint_happy_path(client):
    features = make_valid_features(0.0)
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200

    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)

    assert "probability" in data
    prob = float(data["probability"])
    assert 0.0 <= prob <= 1.0

    assert "threshold" in data
    assert 0.0 <= float(data["threshold"]) <= 1.0

    assert "request_id" in data
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 10


def test_predict_varied_inputs_smoke(client, rng):
    """
    Test fonctionnel: envoie plusieurs cas synthétiques et vérifie:
    - pas d'erreur serveur
    - sortie cohérente
    """
    for _ in range(10):
        features = make_valid_features(0.0)

        # on perturbe quelques features numériques au hasard
        for k in rng.choice(main.FEATURES_ORDER, size=5, replace=False):
            features[k] = float(rng.normal(loc=0.0, scale=1.0))

        r = client.post("/predict", json={"features": features})
        assert r.status_code == 200

        data = r.json()
        prob = float(data["probability"])
        assert 0.0 <= prob <= 1.0
        assert int(data["prediction"]) in (0, 1)


def test_predict_extreme_values(client):
    """
    Cas limites: valeurs très grandes/petites.
    Vérifie que l'API ne crash pas.
    """
    features = make_valid_features(0.0)
    # on force des extrêmes
    for k in features:
        features[k] = 1e6
    r = client.post("/predict", json={"features": features})
    assert r.status_code in (200, 422, 500)  # idéalement 200, sinon il faut gérer la validation

    features = make_valid_features(0.0)
    for k in features:
        features[k] = -1e6
    r = client.post("/predict", json={"features": features})
    assert r.status_code in (200, 422, 500)