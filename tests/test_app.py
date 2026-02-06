from pathlib import Path
import importlib.util
import joblib

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"
MODEL_PATH = ROOT / "model" / "classifier_employee.pkl"


def test_import_app():
    """L'app doit être importable (pas d'erreur au chargement du module)."""
    assert APP_PATH.exists(), f"app.py introuvable: {APP_PATH}"

    spec = importlib.util.spec_from_file_location("app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # doit ne pas lever d'exception


def test_model_is_loadable():
    """Le fichier modèle doit exister et être chargeable via joblib."""
    assert MODEL_PATH.exists(), f"Modèle introuvable: {MODEL_PATH}"

    obj = joblib.load(MODEL_PATH)
    assert isinstance(obj, dict), "Le .pkl doit contenir un dict (ex: {'model':..., 'scaler':..., 'seuil':...})."
    assert "model" in obj, "Clé 'model' manquante dans l'objet joblib."
    assert "scaler" in obj, "Clé 'scaler' manquante dans l'objet joblib."
    assert "seuil" in obj, "Clé 'seuil' manquante dans l'objet joblib."
