FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libs nativas usadas por xgboost/numba/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements está em Streamlit/
COPY Streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copia TUDO (inclui Dados/ e Modelos/)
COPY . .

# deixe o diretório de trabalho na RAIZ (/app)
# assim os caminhos relativos "Dados/..." e "Modelos/..." funcionam
WORKDIR /app

# Config padrão; Render define $PORT em runtime
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8501

# use a porta do Render ($PORT). Como a expansão não acontece no JSON form,
# usamos shell form para garantir a leitura da env var.
CMD sh -c 'streamlit run Streamlit/streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT'
