"""Reúne los resultados y los ordena según las tablas del informe (sección 7) y la diapositiva 14.

Lee:
  experiments/resultados/benchmark.csv        (experiments/benchmark.py)
  experiments/resultados/plataforma.json
  experiments/resultados/comparacion_*.csv    (Paso 5 del notebook de Colab, datos reales)
  outputs/resultados.json                     (src/pipeline.py)
Escribe:
  outputs/resumen_para_informe.md

Uso:  python experiments/resumen_informe.py
"""
import json, os
import pandas as pd

RAIZ = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(RAIZ, "experiments", "resultados")
OUT = os.path.join(RAIZ, "outputs")


def leer_csv(nombre):
    p = os.path.join(RES, nombre)
    return pd.read_csv(p) if os.path.exists(p) else None


def ms(x):
    return "—" if pd.isna(x) else f"{x * 1000:,.1f} ms"


def mb(x):
    return "—" if pd.isna(x) else f"{x / 1e6:,.2f} MB"


def val(df, est, op, n, col):
    s = df[(df.estructura == est) & (df.operacion == op) & (df.n == n)]
    return s[col].iloc[0] if len(s) and col in s and not pd.isna(s[col].iloc[0]) else float("nan")


def md_tabla(cab, filas):
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    out += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(out)


def n_txt(n):
    return {1_000: "1K", 10_000: "10K", 100_000: "100K", 1_000_000: "1M", 10_000_000: "10M"}.get(n, f"{n:,}")


def main():
    L = ["# Resultados para completar el informe y la presentación", ""]
    b = leer_csv("benchmark.csv")
    plat = os.path.join(RES, "plataforma.json")
    if os.path.exists(plat):
        p = json.load(open(plat))
        L += [f"**Plataforma (sección 6):** {p['plataforma']} · procesador: {p['procesador'] or 's/d'} · Python {p['python']} · semilla {p['semilla']}", ""]

    if b is not None:
        tam = sorted(b.n.unique())
        # ---------------- 7.1
        L += ["## Tabla 7.1 · Filtrado probabilístico frente a estructuras exactas", ""]
        filas = []
        for n in tam:
            filas.append([n_txt(n),
                          f"{ms(val(b, 'bloom', 'construir', n, 'media_s'))} / {mb(val(b, 'bloom', 'construir', n, 'memoria_pico_bytes'))}",
                          f"{ms(val(b, 'set_exacto', 'construir', n, 'media_s'))} / {mb(val(b, 'set_exacto', 'construir', n, 'memoria_pico_bytes'))}",
                          f"{val(b, 'bloom', 'construir', n, 'tasa_fp'):.2%}" if not pd.isna(val(b, 'bloom', 'construir', n, 'tasa_fp')) else "—",
                          f"{val(b, 'count_min', 'insertar', n, 'error_medio'):.3f}" if not pd.isna(val(b, 'count_min', 'insertar', n, 'error_medio')) else "—",
                          mb(val(b, 'dict_exacto', 'insertar', n, 'memoria_pico_bytes'))])
        L += [md_tabla(["n", "Bloom: tiempo / memoria", "Set exacto: tiempo / memoria", "Tasa FP (teórica 1%)", "CMS: error promedio", "Dict exacto: memoria"], filas), ""]
        # ---------------- 7.2
        L += ["## Tabla 7.2 · MinHash+LSH y HNSW frente a métodos exactos", ""]
        filas = []
        for n in [n for n in tam if n <= 100_000]:
            ev = b[(b.estructura == 'minhash_lsh') & (b.operacion == 'pares_similares') & (b.n == n)]
            nota = f" ({ev.evaluacion.iloc[0]})" if len(ev) and 'evaluacion' in ev else ""
            ef = "knn10_x100_ef50"
            filas.append([n_txt(n),
                          ms(val(b, 'minhash_lsh', 'construir', n, 'media_s')),
                          ms(val(b, 'jaccard_bruta', 'pares_similares', n, 'media_s')) + nota,
                          f"{val(b, 'minhash_lsh', 'pares_similares', n, 'precision_candidatos'):.3f} / {val(b, 'minhash_lsh', 'pares_similares', n, 'recall'):.3f}",
                          f"{ms(val(b, 'hnsw', ef, n, 'media_s'))} / {val(b, 'hnsw', ef, n, 'recall_at_10'):.3f}",
                          ms(val(b, 'knn_exacto', 'knn10_x100', n, 'media_s'))])
        L += [md_tabla(["n", "LSH: tiempo", "Fuerza bruta Jaccard: tiempo", "LSH: precisión / recall", "HNSW (ef=50): tiempo / recall@10", "KNN exacto: tiempo"], filas),
              "", "Tiempos de HNSW y KNN exacto: 100 consultas de 10 vecinos. KD-Tree en 32 dimensiones (5.3):", ""]
        filas = [[n_txt(n), ms(val(b, 'kdtree_32d', 'knn10_x100', n, 'media_s')), f"{val(b, 'kdtree_32d', 'knn10_x100', n, 'recall_at_10'):.3f}",
                  ms(val(b, 'hnsw', 'knn10_x100_ef10', n, 'media_s')), ms(val(b, 'hnsw', 'knn10_x100_ef100', n, 'media_s'))] for n in tam if n <= 100_000]
        L += [md_tabla(["n", "KD-Tree 32d: tiempo", "KD-Tree recall@10", "HNSW ef=10", "HNSW ef=100"], filas), ""]
        # ---------------- 7.5
        L += ["## 7.5 · Escalabilidad", "", "Insertar la imagen `experiments/resultados/escalabilidad_loglog.png`. Pendientes medidas:", ""]
        filas = []
        for est, op in [("bloom", "construir"), ("count_min", "insertar"), ("heap", "insertar+extraer"), ("bplustree", "insertar"),
                        ("lista_ordenada", "insertar"), ("kdtree", "construir"), ("minhash_lsh", "construir"), ("hnsw", "construir")]:
            s = b[(b.estructura == est) & (b.operacion == op)].sort_values("n")
            if len(s) >= 2:
                import numpy as np
                pend = np.polyfit(np.log10(s.n), np.log10(s.media_s), 1)[0]
                filas.append([f"{est} ({op})", f"{pend:.2f}"])
        L += [md_tabla(["Estructura", "Pendiente log-log"], filas), ""]
    else:
        L += ["*(Falta experiments/resultados/benchmark.csv: ejecutar experiments/benchmark.py)*", ""]

    # ---------------- comparaciones con datos reales (Colab, paso 5)
    comp = {k: leer_csv(f"comparacion_{k}.csv") for k in ("bloom", "countmin", "lsh", "ann")}
    if any(v is not None for v in comp.values()):
        L += ["## Comparación de configuraciones con datos reales (Colab, paso 5)", ""]
        for k, df in comp.items():
            if df is not None:
                try:
                    tabla = df.to_markdown(index=False, floatfmt=".4g")
                except ImportError:          # sin el paquete tabulate
                    tabla = md_tabla(list(df.columns), df.round(4).astype(str).values.tolist())
                L += [f"**{k}**", "", tabla, ""]

    # ---------------- pipeline
    pr = os.path.join(OUT, "resultados.json")
    if os.path.exists(pr):
        r = json.load(open(pr, encoding="utf-8"))
        g, a, pre, alm = r["grafo"], r["alertas"], r["preprocesamiento"], r["almacen"]
        com = g.get("comunidades", {})
        L += ["## Tabla 7.3 · Grafo de co-compra", ""]
        if com:
            filas = [[m, com.get("louvain", {}).get(k, "—"), com.get("label_propagation", {}).get(k, "—")]
                     for m, k in [("Modularidad Q", "modularidad"), ("Tiempo (s)", "segundos"), ("Número de comunidades", "n_comunidades"),
                                  ("Tamaño mediano / máximo", None), ("Estabilidad (NMI promedio, 10 semillas)", "estabilidad_nmi")]]
            filas[3] = ["Tamaño mediano / máximo",
                        f"{com['louvain']['tam_mediana']} / {com['louvain']['tam_max']}",
                        f"{com['label_propagation']['tam_mediana']} / {com['label_propagation']['tam_max']}"]
            L += [md_tabla(["Métrica", "Louvain", "Label Propagation"], filas), ""]
        L += [f"PageRank frente a grado: Spearman = **{g['spearman_pagerank_vs_grado']}**, productos en común en los Top-10 = **{g.get('top10_comunes', '—')}**. "
              f"Grafo: {g['nodos']:,} nodos, {g['aristas']:,} aristas, fuente: {g['fuente']}.", ""]
        L += ["## Tabla 7.4 · Pipeline de punta a punta", ""]
        refrescos = r.get("refrescos_micro_lote", [])
        recs = [f.get("recomendacion_co_compra_producto_seguido") for f in refrescos if f.get("recomendacion_co_compra_producto_seguido")]
        cambios = sum(1 for x, y in zip(recs, recs[1:]) if x != y)
        filas = [
            ["Eventos procesados", f"{r['eventos_ingestados']:,}"],
            ["Registros inválidos descartados (por motivo) / duplicados / tardíos",
             f"{pre['invalidos']:,} ({', '.join(f'{k}: {v:,}' for k, v in pre['invalidos_por_motivo'].items())}) / {pre['duplicados_eliminados']:,} / {pre['tardios_procesados']:,}"],
            ["Alertas por motivo: lista negra · intensidad · prueba de tarjetas · pago rechazado",
             " · ".join(f"{a['por_motivo'].get(k, 0):,}" for k in ("lista_negra", "intensidad", "card_testing", "pago_rechazado"))],
            ["Precisión de las alertas de fraude frente al fraude inyectado", a["precision_fraude_vs_inyectado"]],
            ["Falsos positivos del Bloom Filter descartados en la verificación", f"{r['filtrado'].get('bloom_falsos_positivos', 0):,} de {r['filtrado'].get('bloom_positivos', 0):,} positivos"],
            ["Tasa de aciertos de la caché LRU / productos agotados", f"{alm['lru_tasa_aciertos']:.1%} / {alm['productos_agotados']:,}"],
            ["Densidad de la matriz usuario-producto", f"{alm['densidad_matriz']:.6f} ({alm['densidad_matriz']:.4%})"],
            ["Refrescos por micro-lote y cambio de la recomendación del producto seguido",
             f"{len(refrescos)} refrescos; la recomendación cambió en {cambios} de {max(0, len(recs) - 1)} transiciones"],
        ]
        L += [md_tabla(["Métrica", "Resultado"], filas), ""]
        L += ["## 7.6 · Traza de un evento", ""] + [f"- **{p['etapa']}:** {p['detalle']}" for p in r.get("traza_evento", [])] + [""]
        L += ["## Diapositiva 14 · columna Resultado", ""]
        filas = []
        if b is not None:
            nmax = max(b[b.estructura == 'bloom'].n)
            fp = val(b, 'bloom', 'construir', nmax, 'tasa_fp')
            filas.append(["Bloom Filter", f"FP = {fp:.2%} a {n_txt(nmax)} → {'cumple' if fp <= 0.015 else 'no cumple'}"])
            cms = b[(b.estructura == 'count_min') & (b.n == nmax)]
            if len(cms):
                ok = cms.error_max.iloc[0] <= cms.cota_eps_N.iloc[0]
                filas.append(["Count-Min Sketch", f"error máx. {int(cms.error_max.iloc[0])} ≤ ε·N = {cms.cota_eps_N.iloc[0]:.0f} → {'cumple' if ok else 'no cumple'}"])
            nl = max(b[(b.estructura == 'minhash_lsh') & (b.operacion == 'pares_similares')].n)
            rec = val(b, 'minhash_lsh', 'pares_similares', nl, 'recall')
            filas.append(["MinHash + LSH", f"recall = {rec:.2f} a {n_txt(nl)} → {'cumple' if rec >= 0.8 else 'no cumple'}"])
            nh = max(b[b.estructura == 'hnsw'].n)
            rh = val(b, 'hnsw', 'knn10_x100_ef50', nh, 'recall_at_10')
            filas.append(["HNSW", f"recall@10 = {rh:.2f} (ef=50) a {n_txt(nh)} → {'cumple' if rh >= 0.9 else 'no cumple'}"])
        sp = g["spearman_pagerank_vs_grado"]
        filas.append(["PageRank", f"Spearman = {sp} · {g.get('top10_comunes', '—')}/10 en común → {'aporta un orden distinto' if sp is not None and sp < 1 else 'igual al grado'}"])
        if com:
            lv, lp = com["louvain"], com["label_propagation"]
            filas.append(["Louvain", f"Q = {lv['modularidad']} vs {lp['modularidad']} · NMI {lv['estabilidad_nmi']} vs {lp['estabilidad_nmi']} → "
                          + ("cumple" if lv['modularidad'] >= lp['modularidad'] else "no cumple")])
        L += [md_tabla(["Técnica", "Resultado"], filas), ""]
    else:
        L += ["*(Falta outputs/resultados.json: ejecutar src/pipeline.py)*", ""]

    os.makedirs(OUT, exist_ok=True)
    destino = os.path.join(OUT, "resumen_para_informe.md")
    open(destino, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\nGuardado en {destino}")


if __name__ == "__main__":
    main()
