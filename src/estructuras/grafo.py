"""Etapa 8 — Grafo de co-compra, PageRank / Personalized PageRank y comunidades.

PageRank (Brin y Page, 1998): PR = (1−d)·v + d·Pᵀ·PR, con P la matriz de transición
ponderada; iteración de potencias O(iteraciones·(N+E)). Con v uniforme es PageRank y con
v concentrado en semillas es Personalized PageRank.
Comunidades: Louvain (Blondel et al., 2008) frente a Label Propagation (Raghavan et al.,
2007), ambas de networkx; modularidad, tiempo y estabilidad (NMI entre semillas)."""
from collections import Counter, defaultdict
from itertools import combinations
import math
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx


def construir_grafo(ordenes, max_productos_orden=50):
    """ordenes: dict id_orden -> conjunto de product_id. Peso = n.º de órdenes compartidas.
    (Si una orden tiene más productos que el límite se ignora: evita O(k²) por órdenes atípicas.)"""
    pesos = Counter()
    for prods in ordenes.values():
        if 2 <= len(prods) <= max_productos_orden:
            for u, v in combinations(sorted(prods), 2):
                pesos[(u, v)] += 1
    return pesos


def a_matriz(pesos):
    nodos = sorted({u for e in pesos for u in e})
    pos = {n: i for i, n in enumerate(nodos)}
    filas, cols, vals = [], [], []
    for (u, v), w in pesos.items():
        filas += [pos[u], pos[v]]; cols += [pos[v], pos[u]]; vals += [w, w]
    A = sp.csr_matrix((vals, (filas, cols)), shape=(len(nodos), len(nodos)), dtype=np.float64)
    return A, nodos


def pagerank(A, d=0.85, tol=1e-6, max_iter=100, personalizacion=None):
    n = A.shape[0]
    fuerza = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, fuerza, out=np.zeros_like(fuerza), where=fuerza > 0)
    PT = (sp.diags(inv) @ A).T.tocsr()          # transpuesta de la matriz de transición
    v = np.full(n, 1.0 / n) if personalizacion is None else personalizacion / personalizacion.sum()
    pr = v.copy()
    for it in range(1, max_iter + 1):
        colgantes = pr[fuerza == 0].sum()
        nuevo = d * (PT @ pr + colgantes * v) + (1 - d) * v
        if np.abs(nuevo - pr).sum() < tol:
            return nuevo, it
        pr = nuevo
    return pr, max_iter


def ranking_grado(A):
    return np.asarray(A.sum(axis=1)).ravel()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def a_networkx(pesos):
    G = nx.Graph()
    G.add_weighted_edges_from((u, v, w) for (u, v), w in pesos.items())
    return G


def comunidades(G, metodo="louvain", seed=42):
    t = time.perf_counter()
    if metodo == "louvain":
        com = nx.community.louvain_communities(G, weight="weight", seed=seed)
    elif metodo == "label_propagation":
        com = list(nx.community.asyn_lpa_communities(G, weight="weight", seed=seed))
    else:
        raise ValueError(metodo)
    seg = time.perf_counter() - t
    tam = sorted((len(c) for c in com), reverse=True)
    return {
        "comunidades": com,
        "modularidad": nx.community.modularity(G, com, weight="weight"),
        "segundos": seg,
        "n_comunidades": len(com),
        "tam_max": tam[0] if tam else 0,
        "tam_mediana": float(np.median(tam)) if tam else 0,
    }


def _etiquetas(com):
    return {n: i for i, c in enumerate(com) for n in c}


def nmi(com_a, com_b):
    """Información mutua normalizada entre dos particiones (1 = idénticas)."""
    la, lb = _etiquetas(com_a), _etiquetas(com_b)
    nodos = list(la.keys() & lb.keys())
    n = len(nodos)
    if n == 0:
        return 0.0
    ca = Counter(la[x] for x in nodos); cb = Counter(lb[x] for x in nodos)
    cab = Counter((la[x], lb[x]) for x in nodos)
    mi = sum(c / n * math.log(c * n / (ca[i] * cb[j])) for (i, j), c in cab.items())
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    return mi / math.sqrt(ha * hb) if ha > 0 and hb > 0 else 1.0


def estabilidad(G, metodo, semillas=range(10)):
    parts = [comunidades(G, metodo, s)["comunidades"] for s in semillas]
    vals = [nmi(parts[i], parts[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))]
    return float(np.mean(vals)) if vals else 1.0
