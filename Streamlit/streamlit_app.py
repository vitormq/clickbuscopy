# Streamlit/streamlit_app.py
import os
import gc
import datetime
import random
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import pyarrow.parquet as pq
from faker import Faker
from matplotlib import pyplot as plt

st.set_page_config(page_title="Previsão de Próxima Compra", layout="wide")
st.title("Previsão de Próxima Compra por Cliente")
st.caption("Clique em **Iniciar** para carregar modelos e dados (pode levar alguns segundos).")

# =========================
# CACHES (otimizados p/ RAM)
# =========================

@st.cache_resource(show_spinner=False)
def carregar_modelos():
    """Carrega e guarda os modelos XGBoost em cache (1x por instância)."""
    modelo_dia = xgb.Booster()
    modelo_destino = xgb.Booster()
    modelo_dia.load_model("Modelos/xgboost_model_dia_exato.json")
    modelo_destino.load_model("Modelos/xgboost_model_trecho.json")
    return modelo_dia, modelo_destino


@st.cache_data(show_spinner=False)
def listar_clientes():
    """Lê apenas a coluna de id_cliente (muito leve) para montar o dropdown."""
    tbl = pq.read_table("Dados/cb_previsao_trecho.parquet", columns=["id_cliente"])
    ids = pd.Series(tbl.column("id_cliente").to_numpy()).drop_duplicates().sort_values().tolist()
    return ids


@st.cache_data(show_spinner=False)
def carregar_classes_slim():
    """Lê somente a coluna Trechos (ordem das classes) e devolve DataFrame enxuto."""
    tbl = pq.read_table("Dados/classes.parquet", columns=["Trechos"])
    return tbl.to_pandas()


@st.cache_data(show_spinner=False)
def carregar_features_cliente(id_cliente: int):
    """Lê SOMENTE as linhas do cliente nos 2 arquivos de features."""
    t_dia = pq.read_table(
        "Dados/cb_previsao_data.parquet",
        filters=[("id_cliente", "=", id_cliente)],
    )
    t_trecho = pq.read_table(
        "Dados/cb_previsao_trecho.parquet",
        filters=[("id_cliente", "=", id_cliente)],
    )
    f_dia = t_dia.to_pandas()
    f_trecho = t_trecho.to_pandas()
    del t_dia, t_trecho
    gc.collect()
    return f_dia, f_trecho


@st.cache_data(show_spinner=False)
def carregar_compras_cliente(id_cliente: int):
    """Lê SOMENTE as compras do cliente, com colunas usadas na UI."""
    cols = [
        "id_cliente", "origem_ida", "destino_ida",
        "qtd_total_compras", "intervalo_medio_dias",
        "vl_medio_compra", "cluster_name", "data_compra"
    ]
    t = pq.read_table("Dados/dataframe.parquet", columns=cols, filters=[("id_cliente", "=", id_cliente)])
    df = t.to_pandas()
    del t
    gc.collect()
    return df


# ==============
# Utilidades UI
# ==============

def gerar_mapeamento_cidades(df_compras_cliente):
    """Gera nomes de cidades fake APENAS para os IDs presentes no cliente."""
    Faker.seed(42)
    fake = Faker("pt_BR")

    todos_ids = set()
    pares = (df_compras_cliente["origem_ida"].astype(str) + "_" + df_compras_cliente["destino_ida"].astype(str))
    for item in pares:
        origem, destino = item.split("_")
        todos_ids.update([origem, destino])

    def gerar_cidade_fake(id_unico):
        random.seed(hash(id_unico))
        return fake.city()

    id_para_cidade = {id_: gerar_cidade_fake(id_) for id_ in todos_ids}

    def mapear_para_cidades(par):
        origem, destino = par.split("_")
        return f"{id_para_cidade[origem]} -> {id_para_cidade[destino]}"

    return mapear_para_cidades


def mostrar_historico(df_compras_cliente, mapear_para_cidades):
    df = df_compras_cliente.copy()
    df["Trechos"] = df["origem_ida"].astype(str) + "_" + df["destino_ida"].astype(str)
    df["trecho_fake"] = df["Trechos"].apply(mapear_para_cidades)
    df = df.sort_values("data_compra", ascending=False).copy()
    df["data_compra"] = pd.to_datetime(df["data_compra"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "data_compra": "Data",
        "trecho_fake": "Trecho",
        "qtd_total_compras": "Quantidade de Passageiros",  # ajuste se essa coluna significar outra coisa
        "vl_medio_compra": "Valor do Ticket (R$)"
    })
    st.subheader("🛒 Histórico de compras do cliente")
    st.dataframe(
        df[["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"]],
        use_container_width=True
    )


# =========
# Interface
# =========

with st.sidebar:
    st.markdown("### Fluxo")
    st.markdown("1) Selecione um cliente  \n2) Clique em **Iniciar**")

ids = listar_clientes()
if not ids:
    st.error("Não foi possível listar clientes (verifique os arquivos em `Dados/`).")
    st.stop()

selected_id = st.selectbox("Selecione o cliente", ids, index=0)

col_a, col_b = st.columns([1, 3])
with col_a:
    iniciar = st.button("🚀 Iniciar", type="primary")

if not iniciar:
    st.info("Clique em **Iniciar** para carregar dados do cliente e gerar previsões.")
    st.stop()

# Carregamento sob demanda
with st.status("Carregando modelos e dados...", expanded=True):
    modelo_dia, modelo_destino = carregar_modelos()
    st.write("✔️ Modelos carregados")

    features_dia, features_trecho = carregar_features_cliente(selected_id)
    st.write("✔️ Features carregadas")

    df_compras_cliente = carregar_compras_cliente(selected_id)
    st.write("✔️ Compras do cliente carregadas")

    classes = carregar_classes_slim()
    st.write("✔️ Classes carregadas")

# =====================
# PREVISÕES (XGBoost)
# =====================
try:
    # Dia exato
    input_dia = features_dia[features_dia["id_cliente"] == selected_id].drop(columns=["id_cliente"], errors="ignore")
    dmx_dia = xgb.DMatrix(input_dia)
    data_prevista = modelo_dia.predict(dmx_dia)[0]

    # Trecho (classe)
    input_trecho = features_trecho[features_trecho["id_cliente"] == selected_id].drop(columns=["id_cliente"], errors="ignore")
    dmx_trecho = xgb.DMatrix(input_trecho)
    y_pred = modelo_destino.predict(dmx_trecho)[0]
    destino_pred = int(np.argmax(y_pred))

    # Mapeamento de trechos "bonitos"
    mapear_para_cidades = gerar_mapeamento_cidades(df_compras_cliente)
    classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)

    cliente_data = df_compras_cliente.iloc[0] if not df_compras_cliente.empty else None

    # Resultados
    data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
    st.success(f"📅 **Data provável da próxima compra:** {data_final.strftime('%Y-%m-%d')}")
    st.info(f"🧭 **Trecho provável:** {classes.iloc[destino_pred]['trecho_fake']}")

    if cliente_data is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("🛒 Quantidade total de compras", int(cliente_data["qtd_total_compras"]))
        c2.metric("📊 Intervalo médio (dias)", int(cliente_data["intervalo_medio_dias"]))
        c3.metric("💵 Valor médio ticket (R$)", int(cliente_data["vl_medio_compra"]))
        st.metric("Cluster", str(cliente_data["cluster_name"]))

    mostrar_historico(df_compras_cliente, mapear_para_cidades)

except Exception as e:
    st.error(f"Falha ao gerar previsões: {e}")
    st.stop()

# =============
# SHAP (opcional)
# =============
with st.expander("🔍 Mostrar explicações SHAP (opcional)"):
    st.caption("Calcular SHAP consome memória e CPU — execute apenas se necessário.")
    if st.button("Calcular SHAP"):
        try:
            st.write("**Impacto das variáveis (data)**")
            expl_dia = xgb.Booster()  # XGBoost nativo não precisa wrapper para SHAP clássico
            explainer_dia = None  # placeholder (para compatibilidade com sua estrutura)
            import shap as shap_lib

            # Para modelo de data
            explainer = shap_lib.Explainer(modelo_dia)
            vals = explainer(input_dia)
            fig1, ax1 = plt.subplots()
            shap_lib.plots.waterfall(vals[0], show=False)
            st.pyplot(fig1)
            plt.close(fig1); del fig1, ax1, vals, explainer; gc.collect()

            # Para modelo de trecho (classe escolhida)
            explainer2 = shap_lib.Explainer(modelo_destino)
            vals2 = explainer2(input_trecho)
            shap_value_classe = vals2[0, :, destino_pred]
            st.write("**Impacto das variáveis (trecho – classe escolhida)**")
            fig2, ax2 = plt.subplots()
            shap_lib.plots.waterfall(shap_value_classe, show=False)
            st.pyplot(fig2)
            plt.close(fig2); del fig2, ax2, shap_value_classe, vals2, explainer2; gc.collect()

        except Exception as e:
            st.warning(f"Não foi possível calcular SHAP agora: {e}")
