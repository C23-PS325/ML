FROM python:3.11.3-alpine3.18
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8080"]
EXPOSE 8080