[![Build and Deploy to Cloud Run](https://github.com/C23-PS325/communicare-ml-engine/actions/workflows/google-cloudrun-deploy.yml/badge.svg?branch=deploy-to-cloudrun)](https://github.com/C23-PS325/communicare-ml-engine/actions/workflows/google-cloudrun-deploy.yml)
# Communicare Machine Learning Repository

## Development Process
1. Making the dataset 
2. Image and sound prepocessing
3. Making the model using CNN
4. Save the model
5. Use the model for make the prediction using REST API
6. Make the REST API using FastAPI

## Installation
1. ### Clone This Repository
2. ### Install All The Requirements using PIP
#### You can install it by using this command
```
pip install [requirement]==[version]
```
| Requirements         | Version       |
| -------------------- | ------------- |
| numpy                | 1.23.5        |
| pandas               | 2.0.1         |
| keras                | 2.12.0        |
| tensorflow           | 2.12.0        |
| opency-python        | 4.7.0.72      |
| librosa              | 0.10.0.post2  |
| moviepy              | 1.03          |
| google-cloud-storage | 2.0.1         |
| fastapi              | 0.95.2        |
| fast-api-gcs         | 0.0.11        |
| pydantic             | 1.10.7        |
| python-multipart     | 0.0.6         |
| uvicorn              | 0.22.0        |
| hypercorn            | 0.14.3        |

2. ### Running The API
#### If you wants to try the API on local environment, you can run this code on terminal
```
uvicorn api_model:app
```
#### After the uvicorn server running, you can test it using this end point ```localhost:8000/predict-video```
#### This code for running the server on Cloud Run
```
hypercorn api_model:app — log-level debug — bind 0.0.0.0:8080
```

3. ### Finally, You can start using the REST API
