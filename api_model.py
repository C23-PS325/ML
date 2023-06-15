from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi_gcs import FGCSUpload
from api_function import APIHelper
from pydantic import BaseModel
import datetime as dt
import os, json

class ImageBody(BaseModel):
    base64_image: str

app = FastAPI()
credential_path = "/.service-account.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
google_project_id = "communicare-staging"
google_bucket_name = "communicare-proc-dev"

@app.get("/")
def read_root():
    return {"error":"False", "message":"Welcome to the API"}

# @app.post("/predict-image", status_code=200)
# def predic_image(data: ImageBody):
#     try:    
#         prediction = helper.predict_image(data.base64_image)
#         prediction = prediction.tolist()
#         data_response  = {"angry": prediction[0][0], "fear": prediction[0][1], "happy": prediction[0][2], "sad": prediction[0][3], "surprise": prediction[0][4]}
#         return {"error":"False", "message":"Prediction successful", "response":data_response}
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/predict-video", status_code=200)
async def upload_video(file_video : UploadFile = File(...)):
    try:
        curr_timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        file_video.filename = f"{file_video.filename.replace('.mp4','') + curr_timestamp }.mp4"
        upload_res = await FGCSUpload.file(
                project_id=google_project_id, 
                bucket_name=google_bucket_name, 
                file=file_video, 
                file_path='video', 
                maximum_size=100_000_000, #100MB
                allowed_extension= ['mp4'],
                file_name=file_video.filename #optional custom file name
            )
        helper = APIHelper()
        result = helper.predict_video(upload_res['blob_path'])
        
        return {"error":"false", "message":"Prediction success", "response":result}
    except Exception as e:
        print(e.with_traceback())
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    