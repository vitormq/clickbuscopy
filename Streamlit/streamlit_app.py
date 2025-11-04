# Streamlit/streamlit_app.py
import os
import gc
import datetime
import random

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import pyarrow.parquet as pq
from faker import Faker
from matplotlib import pyplot as plt

st.set_page_config(page_title="Previsão de Próxima Compra", layout="wide")
st.title("Previsão de Próxima Compra por Cliente")

# ============
# BASE DE PATH
# ============
# Garante que "Dados/..." e "Modelos/..." sejam resolvidos a partir da pasta do arquivo,
# mesmo que o WORKDIR do Docker seja /app ou /app/Streamlit.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_DIR = os.path.join(BASE_DIR, "Dados")
MODELOS_DIR = os.path.join(BASE_DIR, "Modelos")

def _must_exist(path: str):
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: `{path}`. "
                 f"Confirme se o arquivo está no repositório e no caminho correto.")
        st.stop()

# ===============
# CACHES / LOADERS
# ===============
@st.cache_resource(show_spinner=False)
def carregar_modelos():
    p1 = os.path.join(MODELOS_DIR, "xgboost_model_dia_exato.json")
    p2 = os.path.join(MODELOS_DIR, "xgboost_model_trecho.json")
    _must_exist(p1); _must_exist(p2)
    m1 = xgb.Booster(); m1.load_model(p1)
    m2 = xgb.Booster(); m2.load_model(p2)
    return m1, m2

@st.cache_data(show_spinner=False)
def listar_clientes():
    p = os.path.join(DADOS_DIR, "cb_previsao_trecho.parquet")
    _must_exist(p)
    tbl = pq.read_table(p, columns=["id_cliente"])
    ids = pd.Series(tbl.column("id_cliente").to_numpy()).drop_duplicates().sort_values().tolist()
    return ids

@st.cache_data(show_spinner=False)
def carregar_classes_slim():
    p = os.path.join(DADOS_DIR, "classes.parquet")
    _must_exist(p)
    tbl = pq.read_table(p, columns=["Trechos"])
    return tbl.to_pandas()

@st.cache_data(show_spinner=False)
def carregar_features_cliente(id_cliente: int):
    p_dia = os.path.join(DADOS_DIR, "cb_previsao_data.parquet")
    p_tre = os.path.join(DADOS_DIR, "cb_previsao_trecho.parquet")
    _must_exist(p_dia); _must_exist(p_tre)
    t_dia = pq.read_table(p_dia, filters=[("id_cliente", "=", id_cliente)])
    t_tre = pq.read_table(p_tre, filters=[("id_cliente", "=", id_cliente)])
    f_dia = t_dia.to_pandas(); f_tre = t_tre.to_pandas()
    del t_dia, t_tre; gc.collect()
    return f_dia, f_tre

@st.cache_data(show_spinner=False)
def carregar_compras_cliente(id_cliente: int):
    p = os.path.join(DADOS_DIR, "dataframe.parquet")
    _must_exist(p)
    cols = [
        "id_cliente","origem_ida","destino_ida",
        "qtd_total_compras","intervalo_medio_dias",
        "vl_medio_compra","cluster_name","data_compra"
    ]
    t = pq.read_table(p, columns=cols, filters=[("id_cliente", "=", id_cliente)])
    df = t.to_pandas()
    del t; gc.collect()
    return df

# =========
# UTILIDADES
# =========
def gerar_mapeamento_cidades(df_compras_cliente):
    Faker.seed(42); fake = Faker("pt_BR")
    todos_ids = set()
    pares = (df_compras_cliente["origem_ida"].astype(str) +
             "_" + df_compras_cliente["destino_ida"].astype(str))
    for item in pares:
        origem, destino = item.split("_")
        todos_ids.update([origem, destino])

    def cidade_fake(i):
        random.seed(hash(i)); return fake.city()
    id2city = {i: cidade_fake(i) for i in todos_ids}

    def mapear(par):
        o, d = par.split("_")
        return f"{id2city[o]} -> {id2city[d]}"
    return mapear

def mostrar_historico(df, mapear):
    df = df.copy()
    df["Trechos"] = df["origem_ida"].astype(str) + "_" + df["destino_ida"].astype(str)
    df["trecho_fake"] = df["Trechos"].apply(mapear)
    df = df.sort_values("data_compra", ascending=False).copy()
    df["data_compra"] = pd.to_datetime(df["data_compra"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "data_compra": "Data",
        "trecho_fake": "Trecho",
        "qtd_total_compras": "Quantidade de Passageiros",  # ajuste se necessário
        "vl_medio_compra": "Valor do Ticket (R$)"
    })
    st.subheader("🛒 Histórico de compras do cliente")
    st.dataframe(df[["Data","Trecho","Quantidade de Passageiros","Valor do Ticket (R$)"]],
                 use_container_width=True)

# =========
# INTERFACE
# =========
with st.sidebar:
    st.markdown("### Fluxo")
    st.markdown("1) Selecione um cliente  \n2) Clique em **Iniciar**")

ids = listar_clientes()
if not ids:
    st.error("Nenhum cliente encontrado em `cb_previsao_trecho.parquet`.")
    st.stop()

selected_id = st.selectbox("Selecione o cliente", ids, index=0)
iniciar = st.button("🚀 Iniciar", type="primary")

if not iniciar:
    st.info("Pronto para iniciar. Selecione um cliente e clique em **Iniciar**.")
    st.stop()

with st.status("Carregando modelos e dados...", expanded=True):
    modelo_dia, modelo_destino = carregar_modelos()
    st.write("✔️ Modelos carregados")

    features_dia, features_trecho = carregar_features_cliente(selected_id)
    if features_dia.empty or features_trecho.empty:
        st.error("Não há features para esse cliente. Tente outro.")
        st.stop()
    st.write("✔️ Features do cliente carregadas")

    df_compras = carregar_compras_cliente(selected_id)
    if df_compras.empty:
        st.warning("Cliente sem histórico em `dataframe.parquet`.")
    else:
        st.write("✔️ Histórico do cliente carregado")

    classes = carregar_classes_slim()
    st.write("✔️ Classes carregadas")

# =============
# PREVISÕES
# =============
try:
    X_dia = features_dia[features_dia["id_cliente"] == selected_id].drop(columns=["id_cliente"], errors="ignore")
    dmx_dia = xgb.DMatrix(X_dia)
    data_prevista = modelo_dia.predict(dmx_dia)[0]

    X_tre = features_trecho[features_trecho["id_cliente"] == selected_id].drop(columns=["id_cliente"], errors="ignore")
    dmx_tre = xgb.DMatrix(X_tre)
    probs = modelo_destino.predict(dmx_tre)[0]
    destino_pred = int(np.argmax(probs))

    mapear = gerar_mapeamento_cidades(df_compras if not df_compras.empty else
                                      pd.DataFrame({"origem_ida":[],"destino_ida":[]}))
    classes = classes.copy()
    classes["trecho_fake"] = classes["Trechos"].apply(mapear)

    data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
    st.success(f"📅 **Data provável da próxima compra:** {data_final:%Y-%m-%d}")
    st.info(f"🧭 **Trecho provável:** {classes.iloc[destino_pred]['trecho_fake']}")

    if not df_compras.empty:
        row = df_compras.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("🛒 Quantidade total de compras", int(row["qtd_total_compras"]))
        c2.metric("📊 Intervalo médio (dias)", int(row["intervalo_medio_dias"]))
        c3.metric("💵 Valor médio ticket (R$)", int(row["vl_medio_compra"]))
        st.metric("Cluster", str(row["cluster_name"]))
        mostrar_historico(df_compras, mapear)

except FileNotFoundError as e:
    st.error(f"Arquivo não encontrado: {e}")
    st.stop()
except Exception as e:
    st.error(f"Falha ao gerar previsões: {e}")
    st.stop()
