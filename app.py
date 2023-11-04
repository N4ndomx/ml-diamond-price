from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib

app = FastAPI(title = 'Diamond Prediction')

model = load(pathlib.Path('model/model_ps.joblib'))

class InputData(BaseModel):
    carat:int= 0.53
    cut:int =2.0
    color:int =3.2
    clarity:int= 6.0
    depth:int=61.8
    table:int=56.0
    x:int=5.19
    y:int =5.24
    z:int=3.22

class OutputData(BaseModel):
    price:float=0.80318881046519

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)

    return {'price':result}
