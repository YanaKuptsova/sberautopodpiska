import json
from typing import Optional

import joblib

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model.pkl')


class Form(BaseModel):
    session_id:str
    client_id:str
    visit_date:str
    visit_time:str
    visit_number: int
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None

class Prediction(BaseModel):
    session_id: str
    Result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/prediction', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'Result': y[0]
    }
