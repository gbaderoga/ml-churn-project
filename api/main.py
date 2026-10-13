from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
import pandas as pd
import logging

import os


from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, HTTPException
from pathlib import Path

import logging
from pathlib import Path

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "api.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0"
)

model = joblib.load("models/model.pkl")


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP"
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "Random Forest",
        "version": "1.0.0"
    }



@app.post("/predict")
def predict(data: ClientData):

    start = time.perf_counter()

    logger.info(
        "Prediction request: model_version=%s",
        MODEL_VERSION
    )

    try:
        # Construire un DataFrame avec les variables
        values = pd.DataFrame([{
            "age": data.age,
            "anciennete": data.anciennete,
            "revenu": data.revenu,
            "nb_reclamations": data.nb_reclamations,
            "montant_mensuel": data.montant_mensuel,
            "retards_paiement": data.retards_paiement
        }])

        # Respecter l'ordre des colonnes de l'entraînement
        if hasattr(model, "feature_names_in_"):
            values = values.loc[:, model.feature_names_in_]

        # Effectuer la prédiction
        prediction = model.predict(values)[0]

        # Probabilité de la classe CHURN (classe 1)
        churn_index = list(model.classes_).index(1)
        probability = model.predict_proba(values)[0][churn_index]

        # Temps d'exécution en millisecondes
        duration_ms = (time.perf_counter() - start) * 1000

        # Enregistrer le succès dans les logs
        logger.info(
            "Prediction success: model_version=%s, "
            "prediction=%s, probability=%.4f, duration_ms=%.2f",
            MODEL_VERSION,
            "CHURN" if prediction == 1 else "NO_CHURN",
            probability,
            duration_ms
        )

        return {
            "prediction": "CHURN" if prediction == 1 else "NO_CHURN",
            "probability": round(float(probability), 4),
            "model_version": MODEL_VERSION,
            "duration_ms": round(duration_ms, 2)
        }

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "Prediction error: model_version=%s, duration_ms=%.2f",
            MODEL_VERSION,
            duration_ms
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction"
        )


Instrumentator().instrument(app).expose(app)

