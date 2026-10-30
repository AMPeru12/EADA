"""Experimentos de escalabilidad y comparación con alternativas exactas.

Para cada tamaño n (1K, 10K, 100K, 1M, 10M) mide tiempo (media y desviación estándar de
varias repeticiones) y memoria pico (tracemalloc), y para las estructuras aproximadas la
calidad frente a la alternativa exacta:
  Bloom Filter  vs set        → tiempo, memoria, tasa de falsos positivos
  Count-Min     vs Counter    → tiempo, memoria, error de frecuencia
  Heap, B+Tree  vs lista ordenada (bisect), KD-Tree vs fuerza bruta
  MinHash+LSH   vs Jaccard por fuerza bruta → tiempo, precisión y recall de pares
  HNSW          vs KNN exacto → tiempo, recall@10, memoria

Uso:
  python experiments/benchmark.py --max 1000000 --reps 100
Repeticiones: 100 hasta 10K; para n mayores se reducen (ver `reps_para`) porque una sola
construcción tarda segundos o minutos. Se reporta el número usado en cada fila.
"""
import argparse, bisect, json, os, platform, random, sys, time, tracemalloc
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.estructuras.bloom import BloomFilter
from src.estructuras.countmin import CountMinSketch
from src.estructuras.heap_prioridad import ColaPrioridad
from src.estructuras.bplustree import BPlusTree
from src.estructuras.kdtree import KDTree, knn_fuerza_bruta
from src.estructuras.minhash_lsh import MinHash, LSH, jaccard
from src.estructuras.hnsw import HNSW, knn_coseno_exacto, recall_at_k

SEED = 42
FILAS = []


def reps_para(n, reps):
    if n <= 10_000: return reps
    if n <= 100_000: return max(3, reps // 10)
    if n <= 1_000_000: return 3
    return 1


def medir(fn, reps):
    t = []
    for _ in range(reps):
        a = time.perf_counter(); fn(); t.append(time.perf_counter() - a)
    return float(np.mean(t)), float(np.std(t))


def memoria_pico(fn):
    tracemalloc.start(); tracemalloc.reset_peak()
    obj = fn()
    _, pico = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return pico, obj


def fila(estructura, operacion, n, reps, media, std, mem=None, **extra):
    r = {"estructura": estructura, "operacion": operacion, "n": n, "reps": reps,
         "media_s": media, "std_s": std, "memoria_pico_bytes": mem, **extra}
    FILAS.append(r)
    print(f"{estructura:14s} {operacion:22s} n={n:>10,} reps={reps:>3} {media*1e3:10.2f} ms ± {std*1e3:7.2f}"
          + (f"  mem={mem/1e6:8.2f} MB" if mem else "") + ("  " + json.dumps(extra) if extra else ""))


def bloom_vs_set(n, reps):
    claves = [f"user:{i}" for i in range(n)]
    ausentes = [f"x:{i}" for i in range(min(n, 100_000))]
    def bloom():
        b = BloomFilter(n, 0.01)
        for c in claves: b.agregar(c)
        return b
    def conj():
        return set(claves)
    r = reps_para(n, reps)
    mb, b = memoria_pico(bloom); ms, s = memoria_pico(conj)
    fp = sum(c in b for c in ausentes) / len(ausentes)
    fila("bloom", "construir", n, r, *medir(bloom, r), mb, tasa_fp=round(fp, 4), fp_teorica=round(b.tasa_fp_teorica(), 4))
    fila("set_exacto", "construir", n, r, *medir(conj, r), ms, tasa_fp=0.0)
    q = claves[: min(n, 10_000)]
    fila("bloom", "consultar_10k", n, r, *medir(lambda: [c in b for c in q], r))
    fila("set_exacto", "consultar_10k", n, r, *medir(lambda: [c in s for c in q], r))


def cms_vs_dict(n, reps):
    rng = np.random.default_rng(SEED)
    datos = (rng.zipf(1.3, n) % max(10, n // 10)).tolist()
    def cms():
        c = CountMinSketch(0.001, 0.01)
        for x in datos: c.agregar(x)
        return c
    def exacto():
        return Counter(datos)
    r = reps_para(n, reps)
    mc, c = memoria_pico(cms); me, e = memoria_pico(exacto)
    claves = list(e.keys())[:10_000]
    err = [c.estimar(k) - e[k] for k in claves]
    fila("count_min", "insertar", n, r, *medir(cms, r), mc, error_medio=round(float(np.mean(err)), 3),
         error_max=int(max(err)), cota_eps_N=round(0.001 * n, 1))
    fila("dict_exacto", "insertar", n, r, *medir(exacto, r), me, error_medio=0)


def heap(n, reps):
    rng = random.Random(SEED)
    evs = [{"event_type": rng.choice(["view", "cart", "purchase"]), "order_type": "estandar", "ts": i} for i in range(n)]
    def run():
        h = ColaPrioridad()
        for e in evs: h.insertar(e)
        while len(h): h.extraer()
    r = reps_para(n, reps)
    fila("heap", "insertar+extraer", n, r, *medir(run, r), memoria_pico(run)[0])


def bplus_vs_lista(n, reps):
    rng = random.Random(SEED); claves = rng.sample(range(n * 10), n)
    def arbol():
        t = BPlusTree(64)
        for k in claves: t.insertar(k, k)
        return t
    def lista():
        l = []
        for k in claves: bisect.insort(l, k)
        return l
    r = reps_para(n, reps)
    mt, t = memoria_pico(arbol)
    fila("bplustree", "insertar", n, r, *medir(arbol, r), mt, altura=t.altura())
    if n <= 1_000_000:  # insort es O(n) por inserción: O(n²) total
        ml, l = memoria_pico(lista)
        fila("lista_ordenada", "insertar", n, r, *medir(lista, r), ml)
    q = rng.sample(claves, min(n, 1000))
    fila("bplustree", "buscar_1000", n, r, *medir(lambda: [t.buscar(k) for k in q], r))
    fila("bplustree", "rango_1%", n, r, *medir(lambda: t.rango(0, n // 10), r))


def kd_vs_bruta(n, reps, dim=4):
    rng = np.random.default_rng(SEED); X = rng.random((n, dim)); Q = rng.random((100, dim))
    r = reps_para(n, reps)
    mk, t = memoria_pico(lambda: KDTree(X))
    fila("kdtree", "construir", n, r, *medir(lambda: KDTree(X), r), mk)
    fila("kdtree", "knn10_x100", n, r, *medir(lambda: [t.knn(q, 10) for q in Q], r))
    fila("knn_bruta", "knn10_x100", n, r, *medir(lambda: [knn_fuerza_bruta(X, q, 10) for q in Q], r))


def lsh_vs_bruta(n, reps, max_bruta=3000, n_muestra=300):
    """Misma configuración que el pipeline: 256 permutaciones, 128 bandas × 2 filas.
    Pares similares: Jaccard ≥ 0.125, el mismo criterio del paso 5.3 de Colab.
    Hasta max_bruta conjuntos se comparan todos los pares (O(n²)). Para n mayor, la precisión y
    el recall se estiman con n_muestra conjuntos consultados contra todos los demás (O(n_muestra·n))."""
    rng = np.random.default_rng(SEED)
    base = [set(rng.integers(0, 50_000, rng.integers(5, 40)).tolist()) for _ in range(max(1, n // 5))]
    conjuntos = []
    for i in range(n):  # grupos de 5 conjuntos parecidos (variaciones de una base)
        b = base[i // 5]
        conjuntos.append({x for x in b if rng.random() > 0.2} | set(rng.integers(0, 50_000, 3).tolist()))
    mh = MinHash(256)
    def construir():
        l = LSH(256, 128)
        for i, s_ in enumerate(conjuntos): l.insertar(i, mh.firma(s_))
        return l
    r = max(1, reps_para(n, reps) // 10)
    ml, l = memoria_pico(construir)
    fila("minhash_lsh", "construir", n, r, *medir(construir, r), ml)
    umbral = 0.125
    consultas = range(n) if n <= max_bruta else rng.choice(n, n_muestra, replace=False).tolist()
    evaluacion = "todos los pares" if n <= max_bruta else f"muestra de {n_muestra} consultas"
    t = time.perf_counter()
    reales = {(min(i, j), max(i, j)) for i in consultas for j in range(n) if j != i and jaccard(conjuntos[i], conjuntos[j]) >= umbral}
    tb = time.perf_counter() - t
    t = time.perf_counter()
    cand = {(min(i, j), max(i, j)) for i in consultas for j in l.candidatos(l.firmas[i], excluir=i)}
    encontrados = {p for p in cand if jaccard(conjuntos[p[0]], conjuntos[p[1]]) >= umbral}
    tl = time.perf_counter() - t
    prec = len(encontrados) / len(cand) if cand else 1.0
    rec = len(encontrados & reales) / len(reales) if reales else 1.0
    fila("jaccard_bruta", "pares_similares", n, 1, tb, 0.0, pares=len(reales), evaluacion=evaluacion, umbral_jaccard=umbral)
    fila("minhash_lsh", "pares_similares", n, 1, tl, 0.0, precision_candidatos=round(prec, 3), recall=round(rec, 3),
         evaluacion=evaluacion, umbral_jaccard=umbral)


def hnsw_vs_bruta(n, reps, dim=32):
    rng = np.random.default_rng(SEED)
    centros = rng.standard_normal((max(1, n // 100), dim))
    X = (centros[rng.integers(0, len(centros), n)] + 0.3 * rng.standard_normal((n, dim))).astype(np.float32)
    Q = X[rng.choice(n, 100, replace=False)]
    def construir():
        h = HNSW(dim, M=16, ef_construction=100)
        for v in X: h.insertar(v)
        return h
    mh, h = memoria_pico(construir)
    fila("hnsw", "construir", n, 1, *medir(construir, 1), mh)
    exactos = [knn_coseno_exacto(X, q, 10) for q in Q]
    r = reps_para(n, reps)
    fila("knn_exacto", "knn10_x100", n, r, *medir(lambda: [knn_coseno_exacto(X, q, 10) for q in Q], r), X.nbytes)
    # KD-Tree sobre vectores normalizados: la distancia euclidiana ordena igual que el coseno
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    mk, kd = memoria_pico(lambda: KDTree(Xn))
    rec_kd = float(np.mean([recall_at_k(kd.knn(q, 10), e) for q, e in zip(Qn, exactos)]))
    fila("kdtree_32d", "knn10_x100", n, r, *medir(lambda: [kd.knn(q, 10) for q in Qn], r), mk, recall_at_10=round(rec_kd, 3))
    for ef in (10, 50, 100):
        rec = float(np.mean([recall_at_k(h.buscar(q, 10, ef), e) for q, e in zip(Q, exactos)]))
        fila("hnsw", f"knn10_x100_ef{ef}", n, r, *medir(lambda: [h.buscar(q, 10, ef) for q in Q], r), recall_at_10=round(rec, 3))


def graficar(csv_path, out_png):
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    series = [("bloom", "construir"), ("set_exacto", "construir"), ("count_min", "insertar"), ("dict_exacto", "insertar"),
              ("heap", "insertar+extraer"), ("bplustree", "insertar"), ("lista_ordenada", "insertar"),
              ("kdtree", "construir"), ("minhash_lsh", "construir"), ("hnsw", "construir")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for est, op in series:
        s = df[(df.estructura == est) & (df.operacion == op)].sort_values("n")
        if len(s) >= 2:
            pend = np.polyfit(np.log10(s.n), np.log10(s.media_s), 1)[0]
            ax.loglog(s.n, s.media_s, marker="o", label=f"{est} ({op}) pendiente={pend:.2f}")
    ax.set_xlabel("n (elementos)"); ax.set_ylabel("tiempo (s)"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Escalabilidad log-log (pendiente 1 ≈ O(n), 2 ≈ O(n²))"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1_000_000, help="tamaño máximo (10000000 para 10M)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--max-hnsw", type=int, default=100_000, help="HNSW en Python puro: limitar por tiempo")
    ap.add_argument("--max-lsh", type=int, default=100_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "resultados"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tamanos = [n for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000) if n <= a.max]
    info = {"plataforma": platform.platform(), "procesador": platform.processor(), "python": sys.version.split()[0],
            "semilla": SEED, "tamanos": tamanos, "reps_base": a.reps}
    print(json.dumps(info, ensure_ascii=False))
    for n in tamanos:
        bloom_vs_set(n, a.reps); cms_vs_dict(n, a.reps); heap(n, a.reps); bplus_vs_lista(n, a.reps); kd_vs_bruta(n, a.reps)
        if n <= a.max_lsh: lsh_vs_bruta(n, a.reps)
        if n <= a.max_hnsw: hnsw_vs_bruta(n, a.reps)
    import pandas as pd
    csv = os.path.join(a.out, "benchmark.csv")
    pd.DataFrame(FILAS).to_csv(csv, index=False)
    with open(os.path.join(a.out, "plataforma.json"), "w") as f: json.dump(info, f, indent=2)
    graficar(csv, os.path.join(a.out, "escalabilidad_loglog.png"))
    print(f"\nResultados en {csv}")


if __name__ == "__main__":
    main()
