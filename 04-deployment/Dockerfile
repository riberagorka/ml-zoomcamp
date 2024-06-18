FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "yellow_tripdata_2023-05.parquet", "./"]


RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

ENTRYPOINT ["python", "starter.py"]
