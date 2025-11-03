FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependências de sistema para xgboost/numba/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala as dependências (arquivo fica em Streamlit/)
COPY Streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o repo (inclui Dados/ e Modelos/)
COPY . .

# Mantém o trabalho na raiz para que caminhos relativos "Dados/..." funcionem
WORKDIR /app

# Configs padrão do Streamlit (opcionais)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8501

# Usa a porta do Render ($PORT) e garante streamlit no PATH via "python -m"
CMD sh -c 'python -m streamlit run Streamlit/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}'
