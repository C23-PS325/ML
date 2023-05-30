FROM python:3.11.3-slim-buster
WORKDIR /usr/src/app
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD ["uvicorn", "api_model:app", "--host=0.0.0.0", "--port=8080"]
EXPOSE 8080
