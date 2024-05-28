# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW.git .

RUN pip3 install -r requirements.txt

COPY app/data/similarity_bert.pkl app/data/similarity_bert.pkl
COPY app/data/similarity_tfidf.pkl app/data/similarity_tfidf.pkl

RUN cd app/ && streamlit run app.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]