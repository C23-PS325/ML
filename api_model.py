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

@app.post("/predict-image", status_code=200)
def predic_image(data: ImageBody):
    try:    
        prediction = model_helper.predict_image(data.base64_image)
        prediction = prediction.tolist()
        data_response  = {"angry": prediction[0][0], "fear": prediction[0][1], "happy": prediction[0][2], "sad": prediction[0][3], "surprise": prediction[0][4]}
        return {"error":"False", "message":"Prediction successful", "response":data_response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")