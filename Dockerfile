# Usa Python 3.10 (estável para numpy/xgboost/shap)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependências nativas necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia o requirements.txt da pasta streamlit
COPY streamlit/requirements.txt ./requirements.txt

# Instala pacotes Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Entra na pasta streamlit onde está o app
WORKDIR /app/streamlit

# Configura variáveis de ambiente padrão do Streamlit
ENV PORT=8501 \
    STREAMLIT_SERVER_PORT=$PORT \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE $PORT

# Comando de inicialização
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
