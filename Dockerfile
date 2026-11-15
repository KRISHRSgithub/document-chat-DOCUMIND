FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Build & run (OpenAI example):
#   docker build -t documind .
#   docker run -p 8501:8501 --env-file .env documind
# Ollama on the host: add  -e OLLAMA_BASE_URL=http://host.docker.internal:11434
