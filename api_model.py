from typing import Optional
from fastapi import FastAPI, HTTPException
import api_function as model_helper
from pydantic import BaseModel

class ImageBody(BaseModel):
    base64_image: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"error":"False", "message":"Welcome to the API"}

@app.post("/predict", status_code=200)
def predic_image(data: ImageBody):
    try:    
        prediction = model_helper.predict_image(data.base64_image)
        return {"error":"False", "message":"Prediction successful", "response":prediction.tolist()}
    except:
        raise HTTPException(status_code=500, detail="Internal Server Error")