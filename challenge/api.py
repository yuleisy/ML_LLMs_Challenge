import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import os
from datetime import datetime

app = FastAPI()

# Cargar el modelo entrenado
model_path = 'model_entrenado.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo {model_path} no existe.")

with open(model_path, 'rb') as f:
    try:
        data = pickle.load(f)
        model = data['model']
        columns = data['columns']
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}")

# Función para validar el formato de fecha
def validate_date(date_str: str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Fecha no válida: {date_str}")

# Clase para recibir los datos de entrada
class FlightData(BaseModel):
    Fecha_I: str = Field(..., description="Formato 'YYYY-MM-DD HH:MM:SS'")
    Vlo_I: str
    Ori_I: str
    Des_I: str
    Emp_I: str
    Fecha_O: str = Field(..., description="Formato 'YYYY-MM-DD HH:MM:SS'")
    Vlo_O: str
    Ori_O: str
    Des_O: str
    Emp_O: str
    DIA: int
    MES: int
    AÑO: int
    DIANOM: str
    TIPOVUELO: str
    OPERA: str
    SIGLAORI: str
    SIGLADES: str

    class Config:
        # Cambiado el nombre de la clave a la nueva nomenclatura de Pydantic v2
        str_strip_whitespace = True  # Eliminar espacios en blanco de las cadenas

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(flight: FlightData) -> dict:
    # Validar fechas
    validate_date(flight.Fecha_I)
    validate_date(flight.Fecha_O)
    
    try:
        # Convierte los datos recibidos a un DataFrame
        df = pd.DataFrame([flight.dict()])

        # Preprocesar datos
        df_preprocessed = model.preprocess(df)

        # Alinear las columnas con las del modelo
        df_preprocessed = df_preprocessed.reindex(columns=columns, fill_value=0)

        # Realizar la predicción
        prediction = model.predict(df_preprocessed)

        return {"prediction": prediction.tolist()}  # Convertir a lista si es necesario
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {e}")
