from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
import pandas as pd

from prometheus_fastapi_instrumentator import Instrumentator

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

    start = time.time()

    # Construire un DataFrame avec les noms des variables
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

    duration = time.time() - start

    print(
        f"prediction={prediction}, "
        f"probability={probability:.4f}, "
        f"duration={duration:.4f}s"
    )

    return {
        "prediction": "CHURN" if prediction == 1 else "NO_CHURN",
        "probability": round(float(probability), 4),
        "model_version": "1.0.0",
        "duration_ms": round(duration * 1000, 2)
    }


Instrumentator().instrument(app).expose(app)

