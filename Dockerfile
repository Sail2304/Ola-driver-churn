FROM python:3.10-alpine
WORKDIR /app
COPY .. ./app

RUN apt-get update -y && apt-get install awscli -y

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]

