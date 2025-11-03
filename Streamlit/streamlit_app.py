import streamlit as st
import pandas as pd
import datetime
import shap
import random
import xgboost as xgb
import numpy as np
from faker import Faker
from matplotlib import pyplot as plt

# Config básica
st.set_page_config(page_title="Previsão de Próxima Compra", layout="wide")
st.title("Previsão de Próxima Compra por Cliente")

# ---------- FUNÇÕES COM CACHE (só executam quando chamadas) ----------
@st.cache_resource
def carregar_modelos():
    modelo_dia = xgb.Booster()
    modelo_destino = xgb.Booster()
    modelo_dia.load_model("Modelos/xgboost_model_dia_exato.json")
    modelo_destino.load_model("Modelos/xgboost_model_trecho.json")
    return modelo_dia, modelo_destino

@st.cache_data(ttl=3600)
def carregar_dados():
    df_compras = pd.read_parquet("Dados/dataframe.parquet", engine="pyarrow")
    features_dia = pd.read_parquet("Dados/cb_previsao_data.parquet", engine="pyarrow")
    features_trecho = pd.read_parquet("Dados/cb_previsao_trecho.parquet", engine="pyarrow")
    classes = pd.read_parquet("Dados/classes.parquet", engine="pyarrow")
    return df_compras, features_dia, features_trecho, classes

# ---------- UI LEVE PRIMEIRO ----------
st.info("Clique em **Iniciar** para carregar modelos e dados (pode levar alguns segundos).")
colA, colB = st.columns([1,3])
with colA:
    iniciar = st.button("🚀 Iniciar", type="primary")

# Evita timeout no Render: só segue se o usuário iniciar (ou se já iniciamos antes)
if not (iniciar or st.session_state.get("iniciado")):
    st.stop()

st.session_state["iniciado"] = True

# ---------- CARREGAMENTO PESADO APÓS INICIAR ----------
with st.spinner("🔄 Carregando modelos e dados..."):
    modelo_dia, modelo_destino = carregar_modelos()
    df_compras_cliente, features_dia, features_trecho, classes = carregar_dados()
st.success("✅ Dados e modelos carregados com sucesso.")

# ---------- FAKE NAMES ----------
Faker.seed(42)
fake = Faker("pt_BR")
unique_ids = features_trecho["id_cliente"].unique()
fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

# ---------- SELEÇÃO DO CLIENTE ----------
selected_fake_name = st.selectbox("Selecione o cliente", fake_names)
id_cliente = name_to_id[selected_fake_name]

# ---------- PREVISÕES ----------
# Dia da compra
input_dia = features_dia[features_dia["id_cliente"] == id_cliente].drop(columns=["id_cliente"])
input_dia_dmatrix = xgb.DMatrix(input_dia)
data_prevista = modelo_dia.predict(input_dia_dmatrix)[0]

# Destino (classe)
input_trecho = features_trecho[features_trecho["id_cliente"] == id_cliente].drop(columns=["id_cliente"])
input_trecho_dmatrix = xgb.DMatrix(input_trecho)
destino_pred = int(np.argmax(modelo_destino.predict(input_trecho_dmatrix)[0]))

# ---------- MAPEAMENTO DE TRECHOS PARA NOMES FAKE ----------
todos_ids = set()
df_compras_cliente["Trechos"] = df_compras_cliente["origem_ida"] + "_" + df_compras_cliente["destino_ida"]
for item in df_compras_cliente["Trechos"]:
    origem, destino = item.split("_")
    todos_ids.update([origem, destino])

def gerar_cidade_fake(id_unico):
    random.seed(hash(id_unico))
    cidade = fake.city()
    return f"{cidade}"

id_para_cidade = {id_: gerar_cidade_fake(id_) for id_ in todos_ids}

def mapear_para_cidades(par):
    origem, destino = par.split("_")
    cidade_origem = id_para_cidade.get(origem, origem)
    cidade_destino = id_para_cidade.get(destino, destino)
    return f"{cidade_origem} -> {cidade_destino}"

classes = classes.copy()
classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)
cliente_data = df_compras_cliente[df_compras_cliente["id_cliente"] == id_cliente]

# ---------- RESULTADOS ----------
data_final = datetime.date.today() + datetime.timedelta(days=int(data_prevista))
st.write(f"📅 Data provável da próxima compra: **{data_final.strftime('%Y-%m-%d')}**")
st.write(f"✈️ Trecho provável da próxima compra: **{classes.iloc[destino_pred][['trecho_fake']][0]}**")

col1, col2, col3 = st.columns(3)
col1.metric("🛒 Quantidade total de compras", int(cliente_data["qtd_total_compras"].iloc[0]))
col2.metric("📊 Intervalo médio (dias)", int(cliente_data["intervalo_medio_dias"].iloc[0]))
col3.metric("📈 Valor médio ticket (R$)", int(cliente_data["vl_medio_compra"].iloc[0]))
st.metric("Cluster:", str(cliente_data["cluster_name"].iloc[0]))

# ---------- HISTÓRICO ----------
st.subheader("🛒 Histórico de compras do cliente")
cliente_data = cliente_data.copy()
cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(mapear_para_cidades)
cliente_data = cliente_data.sort_values("data_compra", ascending=False)
cliente_data["data_compra"] = pd.to_datetime(cliente_data["data_compra"]).dt.strftime("%Y-%m-%d")
cliente_data = cliente_data.rename(columns={
    "data_compra": "Data",
    "trecho_fake": "Trecho",
    "qnt_passageiros": "Quantidade de Passageiros",
    "vl_total_compra": "Valor do Ticket (R$)",
})
st.dataframe(
    cliente_data[["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"]],
    use_container_width=True,
)

# ---------- SHAP (explicabilidade) ----------
st.subheader("🔍 Explicação da previsão da data (impacto das variáveis)")
explainer_dia = shap.Explainer(modelo_dia)
shap_values_dia = explainer_dia(input_dia)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values_dia[0], show=False)
st.pyplot(fig)

st.subheader("🔍 Explicação da previsão do trecho (impacto das variáveis)")
explainer_dest = shap.Explainer(modelo_destino)
shap_values_dest = explainer_dest(input_trecho)
shap_value_classe = shap_values_dest[0, :, destino_pred]
fig, ax = plt.subplots()
shap.plots.waterfall(shap_value_classe, show=False)
st.pyplot(fig)
