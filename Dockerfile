FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⬇️ Aqui com S maiúsculo
COPY Streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copia TUDO (inclui Dados/ e Modelos/)
COPY . .

# entra na pasta do app (S maiúsculo)
WORKDIR /app/Streamlit

ENV PORT=8501 \
    STREAMLIT_SERVER_PORT=$PORT \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE $PORT

CMD ["streamlit","run","streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]
