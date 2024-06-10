FROM python:3.9-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8051

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501"]