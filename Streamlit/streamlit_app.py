# -*- coding: utf-8 -*-
"""
Previsão de Próxima Compra por Cliente
- Auto-start (sem botão)
- Cache de dados e modelos (menos RAM / mais rápido)
- Caminhos robustos (roda local, Render, Streamlit Cloud)
- Tratamento de erros (arquivos ausentes, SHAP sem memória)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker
import xgboost as xgb

# SHAP é pesado; importamos sob try para poder degradar se faltar memória
try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# -----------------------------------------------------------------------------
# Config de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de Próxima Compra",
    page_icon="🧭",
    layout="wide",
)

st.title("Previsão de Próxima Compra por Cliente")

# -----------------------------------------------------------------------------
# Caminhos (sempre relativos a este arquivo)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dados"
MODELOS_DIR = BASE_DIR / "Modelos"

# Arquivos esperados
ARQ_DATAFRAME = DATA_DIR / "dataframe.parquet"
ARQ_DIA = DATA_DIR / "cb_previsao_data.parquet"
ARQ_TRECHO = DATA_DIR / "cb_previsao_trecho.parquet"
ARQ_CLASSES = DATA_DIR / "classes.parquet"
ARQ_MODEL_DIA = MODELOS_DIR / "xgboost_model_dia_exato.json"
ARQ_MODEL_TRECHO = MODELOS_DIR / "xgboost_model_trecho.json"

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _check_exists(p: Path, label: str) -> None:
    if not p.exists():
        st.error(
            f"Arquivo não encontrado: `{p}`.\n\n"
            f"→ Confirme se ele está no repositório no caminho **{label}**.",
            icon="🚫",
        )
        st.stop()


# -----------------------------------------------------------------------------
# Carregamento de dados e modelos (com cache)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega dataframes parquet com cache."""
    for p, label in [
        (ARQ_DATAFRAME, "Streamlit/Dados/dataframe.parquet"),
        (ARQ_DIA, "Streamlit/Dados/cb_previsao_data.parquet"),
        (ARQ_TRECHO, "Streamlit/Dados/cb_previsao_trecho.parquet"),
        (ARQ_CLASSES, "Streamlit/Dados/classes.parquet"),
    ]:
        _check_exists(p, label)

    df_compras = pd.read_parquet(ARQ_DATAFRAME, engine="pyarrow")
    features_dia = pd.read_parquet(ARQ_DIA, engine="pyarrow")
    features_trecho = pd.read_parquet(ARQ_TRECHO, engine="pyarrow")
    classes = pd.read_parquet(ARQ_CLASSES, engine="pyarrow")
    return df_compras, features_dia, features_trecho, classes


@st.cache_resource(show_spinner=True)
def carregar_modelos() -> tuple[xgb.Booster, xgb.Booster]:
    """Carrega modelos XGBoost com cache de recurso (mantém 1 cópia na RAM)."""
    for p, label in [
        (ARQ_MODEL_DIA, "Streamlit/Modelos/xgboost_model_dia_exato.json"),
        (ARQ_MODEL_TRECHO, "Streamlit/Modelos/xgboost_model_trecho.json"),
    ]:
        _check_exists(p, label)

    model_dia = xgb.Booster()
    model_trecho = xgb.Booster()
    model_dia.load_model(str(ARQ_MODEL_DIA))
    model_trecho.load_model(str(ARQ_MODEL_TRECHO))
    return model_dia, model_trecho


# -----------------------------------------------------------------------------
# Carrega tudo (autostart)
# -----------------------------------------------------------------------------
with st.spinner("🔄 Carregando dados e modelos..."):
    df_compras, features_dia, features_trecho, classes = carregar_dados()
    model_dia, model_trecho = carregar_modelos()

st.success("✅ Dados e modelos carregados.")

# -----------------------------------------------------------------------------
# Nomes fake (determinísticos)
# -----------------------------------------------------------------------------
Faker.seed(42)
fake = Faker("pt_BR")

if "id_cliente" not in features_trecho.columns:
    st.error("Coluna `id_cliente` não encontrada em `cb_previsao_trecho.parquet`.")
    st.stop()

unique_ids = features_trecho["id_cliente"].unique().tolist()
fake_names = [fake.name() for _ in unique_ids]
id_to_name = dict(zip(unique_ids, fake_names))
name_to_id = dict(zip(fake_names, unique_ids))

# -----------------------------------------------------------------------------
# Seleção do cliente
# -----------------------------------------------------------------------------
selected_fake_name = st.selectbox("Selecione o cliente", sorted(fake_names))
id_cliente = name_to_id[selected_fake_name]

# -----------------------------------------------------------------------------
# Previsões
# -----------------------------------------------------------------------------
# 1) Dia
input_dia = features_dia.loc[features_dia["id_cliente"] == id_cliente]
if input_dia.empty:
    st.warning("Não há features de data para esse cliente.", icon="⚠️")
    st.stop()

input_dia = input_dia.drop(columns=["id_cliente"], errors="ignore")
pred_dia = model_dia.predict(xgb.DMatrix(input_dia))[0]
data_prevista = dt.date.today() + dt.timedelta(days=int(pred_dia))

# 2) Trecho
input_trecho = features_trecho.loc[features_trecho["id_cliente"] == id_cliente]
if input_trecho.empty:
    st.warning("Não há features de trecho para esse cliente.", icon="⚠️")
    st.stop()

input_trecho_nid = input_trecho.drop(columns=["id_cliente"], errors="ignore")
probs = model_trecho.predict(xgb.DMatrix(input_trecho_nid))[0]
destino_pred_idx = int(np.argmax(probs))

# -----------------------------------------------------------------------------
# Monta nomes fake de cidades para os trechos
# -----------------------------------------------------------------------------
df_compras = df_compras.copy()
if "origem_ida" not in df_compras.columns or "destino_ida" not in df_compras.columns:
    st.error("Colunas `origem_ida` e/ou `destino_ida` não existem em `dataframe.parquet`.")
    st.stop()

df_compras["Trechos"] = df_compras["origem_ida"].astype(str) + "_" + df_compras["destino_ida"].astype(str)

todos_ids = set()
for item in df_compras["Trechos"]:
    try:
        origem, destino = item.split("_")
        todos_ids.update([origem, destino])
    except Exception:
        continue

def gerar_cidade_fake(id_unico: str) -> str:
    random.seed(hash(id_unico))
    return fake.city()

id_para_cidade = {i: gerar_cidade_fake(str(i)) for i in todos_ids}

def mapear_para_cidades(par: str) -> str:
    try:
        origem, destino = par.split("_")
        return f"{id_para_cidade.get(origem, origem)} -> {id_para_cidade.get(destino, destino)}"
    except Exception:
        return par

classes = classes.copy()
if "Trechos" not in classes.columns:
    st.error("Coluna `Trechos` não existe em `classes.parquet`.")
    st.stop()

classes["trecho_fake"] = classes["Trechos"].apply(mapear_para_cidades)

# -----------------------------------------------------------------------------
# Saída principal
# -----------------------------------------------------------------------------
st.markdown(
    f"📅 **Data provável da próxima compra:** `{data_prevista.strftime('%Y-%m-%d')}`"
)
try:
    trecho_prev = classes.iloc[destino_pred_idx]["trecho_fake"]
except Exception:
    trecho_prev = "—"

st.markdown(
    f"🧭 **Trecho provável da próxima compra:** `{trecho_prev}`"
)

# Métricas do cliente
cliente_data = df_compras.loc[df_compras["id_cliente"] == id_cliente].copy()
if cliente_data.empty:
    st.info("Não há histórico de compras para esse cliente.")
else:
    col1, col2, col3 = st.columns(3)
    # Os campos podem não existir em alguns dumps → usamos get com fallback
    def _get_first(col, default=0):
        try:
            return int(pd.to_numeric(cliente_data[col]).iloc[0])
        except Exception:
            return default

    col1.metric("🛒 Quantidade total de compras", _get_first("qtd_total_compras"))
    col2.metric("📊 Intervalo médio (dias)", _get_first("intervalo_medio_dias"))
    col3.metric("💳 Valor médio ticket (R$)", _get_first("vl_medio_compra"))

    st.metric("Cluster", str(cliente_data.get("cluster_name", pd.Series(["—"])).iloc[0]))

# -----------------------------------------------------------------------------
# Histórico
# -----------------------------------------------------------------------------
st.subheader("🛒 Histórico de compras do cliente")
if not cliente_data.empty:
    cliente_data["trecho_fake"] = cliente_data["Trechos"].apply(mapear_para_cidades)
    # datas como string
    if "data_compra" in cliente_data.columns:
        try:
            cliente_data["data_compra"] = pd.to_datetime(cliente_data["data_compra"]).dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    rename_cols = {
        "data_compra": "Data",
        "trecho_fake": "Trecho",
        "qnt_passageiros": "Quantidade de Passageiros",
        "vl_total_compra": "Valor do Ticket (R$)",
    }
    cliente_view = cliente_data.rename(columns=rename_cols)
    cols = [c for c in ["Data", "Trecho", "Quantidade de Passageiros", "Valor do Ticket (R$)"] if c in cliente_view.columns]
    cliente_view = cliente_view.sort_values(by="Data", ascending=False, ignore_index=True, errors="ignore")
    st.dataframe(cliente_view[cols], use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SHAP (opcional; desliga se faltar memória)
# -----------------------------------------------------------------------------
def _plot_shap_for_regression(model: xgb.Booster, X: pd.DataFrame, title: str):
    if not _HAS_SHAP:
        st.info("SHAP não disponível neste ambiente.", icon="ℹ️")
        return
    try:
        import shap
        from matplotlib import pyplot as plt

        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        st.subheader(title)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"Não foi possível renderizar SHAP ({e}).", icon="⚠️")

def _plot_shap_for_multiclass(model: xgb.Booster, X: pd.DataFrame, class_idx: int, title: str):
    if not _HAS_SHAP:
        st.info("SHAP não disponível neste ambiente.", icon="ℹ️")
        return
    try:
        import shap
        from matplotlib import pyplot as plt

        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_value_classe = shap_values[0, :, class_idx]
        st.subheader(title)
        fig = plt.figure()
        shap.plots.waterfall(shap_value_classe, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"Não foi possível renderizar SHAP ({e}).", icon="⚠️")

# Para ambientes com pouca RAM, as chamadas abaixo podem ser comentadas.
_plot_shap_for_regression(model_dia, input_dia, "🔍 Explicação da previsão da **data** (SHAP)")
_plot_shap_for_multiclass(model_trecho, input_trecho_nid, destino_pred_idx, "🔍 Explicação da previsão do **trecho** (SHAP)")
