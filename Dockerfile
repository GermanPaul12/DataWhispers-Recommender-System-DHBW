FROM python:3.9-slim

WORKDIR /app

# Create a startup script
RUN echo "#!/bin/bash\n\
echo 'Listing contents of /app directory:'\n\
ls -l /app\n\
cd /app/backend && flask run --port=5000 &\n\
streamlit run /app/app.py --server.port=8501 --server.address=localhost" > /app/startup.sh

RUN chmod +x /app/startup.sh

COPY app/requirements.txt .

RUN pip3 install -r requirements.txt

RUN mkdir data && mkdir backend
RUN mkdir data/images && mkdir src 
RUN mkdir src/utils

COPY app/backend/* backend/

COPY app/app.py .

COPY app/data/* data/

COPY app/data/images/* data/images/

COPY app/src/* src/

COPY app/src/utils/* src/utils/

EXPOSE 5000 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/app/startup.sh"]