# Usa Python 3.10 (estável p/ numpy/xgboost/shap)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libs nativas (xgboost/numba/matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# instala deps a partir do requirements que está em streamlit/
COPY streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# agora copia TODO o repo (inclui Dados/ e Modelos/)
COPY . .

# entra na pasta onde está o app
WORKDIR /app/streamlit

# variáveis básicas do streamlit
ENV PORT=8501 \
    STREAMLIT_SERVER_PORT=$PORT \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE $PORT

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
