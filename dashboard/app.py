"""Etapa 9 — Panel de ventas, actividad y alertas.
Ejecutar después del pipeline:  streamlit run dashboard/app.py"""
import json, os
import pandas as pd
import streamlit as st

RUTA = os.environ.get("RESULTADOS", "outputs/resultados.json")
st.set_page_config(page_title="Pipeline e-commerce", layout="wide")
st.title("Pipeline de data streams · comercio electrónico")
if not os.path.exists(RUTA):
    st.warning(f"No se encontró {RUTA}. Ejecuta primero: python -m src.pipeline ...")
    st.stop()
r = json.load(open(RUTA, encoding="utf-8"))

c = st.columns(6)
c[0].metric("Eventos ingestados", f"{r['eventos_ingestados']:,}")
c[1].metric("Inválidos descartados", f"{r['preprocesamiento']['invalidos']:,}")
c[2].metric("Duplicados eliminados", f"{r['preprocesamiento']['duplicados_eliminados']:,}")
c[3].metric("Alertas", f"{r['alertas']['total']:,}")
c[4].metric("Productos agotados", f"{r['almacen']['productos_agotados']:,}")
c[5].metric("Aciertos caché LRU", f"{r['almacen']['lru_tasa_aciertos']:.0%}")

izq, der = st.columns(2)
with izq:
    st.subheader("Top-10 más vendidos")
    st.dataframe(pd.DataFrame(r["top10_mas_vendidos"], columns=["product_id", "ventas"]), use_container_width=True)
    st.subheader("Eventos atendidos por prioridad")
    niveles = {"0": "alerta", "1": "express", "2": "estándar", "3": "programada", "4": "carrito/búsqueda", "5": "vista"}
    st.bar_chart(pd.Series({niveles[k]: v for k, v in r["atendidos_por_nivel_heap"].items()}))
with der:
    st.subheader("Alertas verificadas por motivo")
    st.bar_chart(pd.Series(r["alertas"]["por_motivo"]))
    st.caption(f"Precisión frente al fraude inyectado: {r['alertas']['precision_fraude_vs_inyectado']}")
    st.dataframe(pd.DataFrame(r["alertas"]["ejemplos"]), use_container_width=True)

st.subheader("Búsquedas más frecuentes (alimentan el autocompletado)")
st.dataframe(pd.DataFrame(r["busquedas"]["top10"], columns=["búsqueda", "veces"]), use_container_width=True)

st.subheader("Refresco de índices por micro-lote")
st.dataframe(pd.DataFrame(r["refrescos_micro_lote"]), use_container_width=True)

st.subheader("Grafo de co-compra")
g = r["grafo"]
st.write(f"Fuente: {g['fuente']} · {g['nodos']:,} nodos · {g['aristas']:,} aristas · Spearman PageRank vs grado = {g['spearman_pagerank_vs_grado']}")
st.dataframe(pd.DataFrame(g["comunidades"]).T, use_container_width=True)

st.subheader("Traza de un evento de punta a punta")
st.table(pd.DataFrame(r["traza_evento"]))
if "recomendacion_ejemplo" in r:
    st.subheader("Recomendaciones Top-K del producto seguido")
    st.json({k: v for k, v in r["recomendacion_ejemplo"].items() if k != "producto"})
st.caption("Tiempos por etapa (s): " + json.dumps(r["tiempos_seg"], ensure_ascii=False))
