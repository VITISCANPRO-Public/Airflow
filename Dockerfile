# La version doit correspondre exactement à celle dans ton docker-compose.yml
FROM apache/airflow:3.1.3

# Root mode to download dependancies if needed
USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt