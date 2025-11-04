# Streamlit/streamlit_app.py
import os
import time
import pathlib
import random
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
from faker import Faker
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA E ESTILO LIMPO (para embutir no Lovable)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Previsão de Próxima Compra", layout="wide")

st.markdown(
    """
<style>
/* esconde menu/headers do Streamlit (para embed ficar com cara nativa) */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1200px; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PARÂMETROS E CAMINHOS
# -----------------------------------------------------------------------------
# Lê o parâmetro ?autostart=1 (default: 0)
autostart = st.query_params.get("autostart", "0") == "1"

# Pastas esperadas DENTRO da imagem/container
BASE = pathlib.Path(__file__).resolve().parent.parent  # .../app
DADOS = BASE / "Dados"
MODELOS = BASE / "Modelos"

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _ls(p: pathlib.Path) -> str:
    """Lista conteúdo da pasta p com tamanho aproximado."""
    try:
        items = []
        for name in os.listdir(p):
            full = p / name
            try:
                size_mb = full.stat().st_size / 1_000_000
                items.append(f"- {name} ({size_mb:.2f} MB)")
            except Exception:
                items.append(f"- {name}")
        return "\n".join(items) if items else "(vazio)"
    except Exception as e:
        return f"[erro listando: {e}]"

# -----------------------------------------------------------------------------
# CARREGAMENTO (CACHEADO)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_modelos():
    """Carrega os modelos XGBoost do diretório /Modelos (cacheado)."""
    t0 = time.time()
    arq_dia = MODELOS / "xgboost_model_dia_exato.json"
    arq_trecho = MODELOS / "xgboost_model_trecho.json"

    if not arq_dia.exists() or not arq_trecho.exists():
        raise FileNotFoundError(
            "Modelo(s) não encontrados.\n"
            f"- {arq_dia} existe? {arq_dia.exists()}\n"
            f"- {arq_trecho} existe? {arq_trecho.exists()}\n\n"
            f"Conteúdo de {MODELOS}:\n{_ls(MODELOS)}"
        )

    m_dia = xgb.Booster()
    m_dia.load_model(str(arq_dia))

    m_trecho = xgb.Booster()
    m_trecho.load_model(str(arq_trecho))

    return m_dia, m_trecho, time.time() - t0


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados():
    """Carrega os .parquet do diretório /Dados (cacheado)."""
    t0 = time.time()

    arq_dataframe = DADOS / "dataframe.parquet"
    arq_dia = DADOS / "cb_previsao_data.parquet"
    arq_trecho = DADOS / "cb_previsao_trecho.parquet"
    arq_classes = DADOS / "classes.parquet"

    faltantes = [
        str(p) for p in [arq_dataframe, arq_dia, arq_trecho, arq_classes] if not p.exists()
    ]
    if faltantes:
        raise FileNotFoundError(
            "Arquivo(s) de dados ausentes:\n"
            + "\n".join(f"- {f}" for f in faltantes)
            + f"\n\nConteúdo de {DADOS}:\n{_ls(DADOS)}"
        )

    df_compras = pd.read_parquet(arq_dataframe, engine="pyarrow")
    features_dia = pd.read_parquet(arq_dia, engine="pyarrow")
    features_trecho = pd.read_parquet(arq_trecho, engine="pyarrow")
    classes = pd.read_parquet(arq_classes, engine="pyarrow")

    return (df_compras, features_dia, features_trecho, classes, time.time() - t0)

# -----------------------------------------------------------------------------
# CABEÇALHO
# -----------------------------------------------------------------------------
st.title("Previsão de Próxima Compra por Cliente")

# -----------------------------------------------------------------------------
# FLUXO: AUTO-START OU BOTÃO "INICIAR"
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False

# Carrega automaticamente quando embed com ?autostart=1
if autostart and not st.session_state.ready:
    with st.spinner("🔄 Carregando modelos e dados..."):
        modelo_dia, modelo_destino, t_m = carregar_modelos()
        (
            df_compras_cliente,
            features_dia,
            features_trecho,
            classes,
            t_d,
        ) = carregar_dados()
    st.session_state.update(
        dict(
            modelo_dia=modelo_dia,
            modelo_destino=modelo_destino,
            df_compras_cliente=df_compras_cliente,
            features_dia=features_dia,
            features_trecho=features_trecho,
            classes=classes,
            ready=True,
        )
    )

# Fallback: botão Iniciar (para acesso direto)
if not st.session_state.ready:
    st.info(
        "Clique em **Iniciar** para carregar modelos e dados (pode levar alguns segundos)."
    )
    if st.button("🚀 Iniciar", type="primary"):
        with st.spinner("🔄 Carregando modelos e dados..."):
            modelo_dia, modelo_destino, t_m = carregar_modelos()
            (
                df_compras_cliente,
                features_dia,
                features_trecho,
                classes,
                t_d,
            ) = carregar_dados()
        st.session_state.update(
            dict(
                modelo_dia=modelo_dia,
                modelo_destino=modelo_destino,
                df_compras_cliente=df_compras_cliente,
                features_dia=features_dia,
                features_trecho=features_trecho,
                classes=classes,
                ready=True,
            )
        )
        st.success(f"✅ Carregado (modelos: {t_m:.2f}s, dados: {t_d:.2f}s)")
    else:
        st.stop()

# -----------------------------------------------------------------------------
# DADOS EM MÃOS (DAQUI PRA BAIXO É SUA LÓGICA ORIGINAL)
# -----------------------------------------------------------------------------
modelo_dia = st.session_state.modelo_dia
modelo_destino = st.session_state.modelo_destino
df_compras_cliente = st.session_state.df_compras_cliente
features_dia = st.session_state.features_dia
features_trecho = st.session_state.features_trecho
classes = st.session_state.classes

# -----------------------------------------------------------------------------
# FAKER (nomes de clientes)
# -----------------------------------------------------------------------------
Faker.seed(42)
fake = Faker("pt_BR")
unique_ids = features_trecho["id_cliente"].unique()
fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

# Seleção de cliente
selected_fake_name = st.selectbox("Selecione o cliente", fake_names, index=0)
id_cliente = name_to_id[selected_fake_name]

# -----------------------------------------------------------------------------
# PREVISÕES
# -----------------------------------------------------------------------------
# Dia
input_dia = features_dia[features_dia["id_cliente"] == id_cliente].drop(
    columns=["id_cliente"]
)
input_dia_dmatrix = xgb.DMatrix(input_dia)
data_prevista = modelo_dia.predict(input_dia_dmatrix)[0]

# Trecho
input_trecho = features_trecho[features_trecho["id_cliente"] == id_cliente].drop(
    columns=["id_cliente"]
)
input_trecho_dmatrix = xgb.DMatrix(input_trecho)
destino_pred = np.argmax(modelo_destino.predict(input_trecho_dmatrix)[0])

# -----------------------------------------------------------------------------
# MAPEAMENTO DE CIDADES FALSAS PARA TRECHOS
# -----------------------------------------------------------------------------
todos_ids = set()
df_compras_cliente["Trechos"] = (
    df_compras_cliente["origem_ida"] + "_" + df_compras_cliente["destino_ida"]
)
for item in df_compras_cliente["Trechos"]:
    origem, destino = item.split("_")
    todos_ids.update([origem, destino])

Faker.seed(42)
def gerar_cidade_fake(id_unico):
    random.seed(hash(id_unico))
    cidade = fake.city()
    return f"{cidade}"

id_para_cidade = {id_: gerar_cidade_fake(id_) for id_ in todos_ids}

def mapear_para_cidades(par):
    origem, destino = par.split("_")
    cidade_origem = id_para_cidade[origem]
    cidade_destino = id_para_cidade[destino]
    return f"{cidade_origem} -> {cidade_destino}"

classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)
cliente_data = df_compras_cliente[df_compras_cliente["id_cliente"] == id_cliente]

# -----------------------------------------------------------------------------
# RESULTADOS (MÉTRICAS)
# -----------------------------------------------------------------------------
data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
st.write(f"📅 **Data provável da próxima compra:** {data_final.strftime('%Y-%m-%d')}")
st.write(
    f"🧭 **Trecho provável da próxima compra:** "
    f"{classes.iloc[destino_pred][['trecho_fake']][0]}"
)

col1, col2, col3 = st.columns(3)
col1.metric("🛒 Quantidade total de compras", int(cliente_data["qtd_total_compras"].iloc[0]))
col2.metric("📊 Intervalo médio (dias)", int(cliente_data["intervalo_medio_dias"].iloc[0]))
col3.metric("💵 Valor médio ticket (R$)", int(cliente_data["vl_medio_compra"].iloc[0]))
st.metric("🏷️ Cluster", str(cliente_data["cluster_name"].iloc[0]))

# -----------------------------------------------------------------------------
# HISTÓRICO DE COMPRAS DO CLIENTE
# -----------------------------------------------------------------------------
st.subheader("🛒 Histórico de compras do cliente")

cliente_data = cliente_data.copy()
cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(mapear_para_cidades)
cliente_data = cliente_data.sort_values("data_compra", ascending=False)
cliente_data["data_compra"] = cliente_data["data_compra"].dt.strftime("%Y-%m-%d")

cliente_data = cliente_data.rename(
    columns={
        "data_compra": "Data",
        "trecho_fake": "Trecho",
        "qnt_passageiros": "Quantidade de Passageiros",
        "vl_total_compra": "Valor do Ticket (R$)",
    }
)

st.dataframe(
    cliente_data[["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"]],
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# SHAP — EXPLICAÇÕES
# -----------------------------------------------------------------------------
st.subheader("🔍 Explicação da previsão da data (impacto das variáveis)")
try:
    explainer_dia = shap.Explainer(modelo_dia)
    shap_values_dia = explainer_dia(input_dia)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values_dia[0], show=False)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Não foi possível gerar SHAP para data: {e}")

st.subheader("🔍 Explicação da previsão do trecho (impacto das variáveis)")
try:
    explainer_trecho = shap.Explainer(modelo_destino)
    shap_values_trecho = explainer_trecho(input_trecho)
    shap_value_classe = shap_values_trecho[0, :, destino_pred]
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_value_classe, show=False)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Não foi possível gerar SHAP para trecho: {e}")
