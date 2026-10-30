"""Pruebas de correctitud de cada estructura frente a su alternativa exacta.
Ejecutar: python -m pytest tests -q   (o: python tests/test_estructuras.py)"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.estructuras.cola import ColaIngesta
from src.estructuras.heap_prioridad import ColaPrioridad
from src.estructuras.lru import CacheLRU
from src.estructuras.bplustree import BPlusTree
from src.estructuras.kdtree import KDTree, knn_fuerza_bruta
from src.estructuras.bloom import BloomFilter
from src.estructuras.countmin import CountMinSketch, CountMinVentana
from src.estructuras.minhash_lsh import MinHash, LSH, jaccard, jaccard_estimado
from src.estructuras.hnsw import HNSW, knn_coseno_exacto, recall_at_k
from src.estructuras.grafo import construir_grafo, a_matriz, pagerank, a_networkx, comunidades, nmi
from src.estructuras.trie import Trie


def test_cola_fifo():
    q = ColaIngesta(capacidad=3)
    assert all(q.encolar(i) for i in range(3)) and not q.encolar(3)
    assert q.desencolar_lote(5) == [0, 1, 2]


def test_heap_orden():
    h = ColaPrioridad()
    evs = [{"event_type": "view", "ts": 1}, {"event_type": "purchase", "order_type": "programada", "ts": 2},
           {"event_type": "purchase", "order_type": "express", "ts": 3}, {"event_type": "cart", "ts": 0, "alerta": True}]
    for e in evs:
        h.insertar(e)
    assert [h.extraer()["prioridad"] for _ in range(4)] == [0, 1, 3, 5]


def test_lru():
    c = CacheLRU(2); c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1


def test_bplustree_vs_ordenado():
    random.seed(1); t = BPlusTree(orden=8)
    claves = random.sample(range(100_000), 5_000)
    for k in claves:
        t.insertar(k, k * 2)
    assert all(t.buscar(k) == k * 2 for k in claves[:500]) and t.buscar(-1) is None
    esperado = sorted(k for k in claves if 1000 <= k <= 5000)
    assert [k for k, _ in t.rango(1000, 5000)] == esperado


def test_bplustree_eliminar():
    t = BPlusTree(orden=4)
    for k in range(200):
        t.insertar(k, k)
    for k in range(0, 200, 2):
        assert t.eliminar(k)
    assert t.buscar(10) is None and t.buscar(11) == 11
    assert [k for k, _ in t.rango(0, 20)] == list(range(1, 21, 2)) and len(t) == 100


def test_kdtree_exacto():
    rng = np.random.default_rng(0); X = rng.random((3000, 4))
    t = KDTree(X)
    for q in rng.random((20, 4)):
        assert [i for _, i in t.knn(q, 5)] == [i for _, i in knn_fuerza_bruta(X, q, 5)]


def test_bloom_sin_falsos_negativos_y_fp_acotada():
    b = BloomFilter(10_000, 0.01)
    for i in range(10_000):
        b.agregar(f"u{i}")
    assert all(f"u{i}" in b for i in range(10_000))
    fp = sum(f"x{i}" in b for i in range(20_000)) / 20_000
    assert fp < 0.02


def test_countmin_no_subestima():
    cms = CountMinSketch(0.01, 0.01); rng = np.random.default_rng(0)
    datos = rng.zipf(1.5, 20_000) % 1000; real = {}
    for x in datos:
        cms.agregar(int(x)); real[int(x)] = real.get(int(x), 0) + 1
    assert all(cms.estimar(k) >= v for k, v in real.items())
    assert max(cms.estimar(k) - v for k, v in real.items()) <= 0.01 * len(datos) * 3


def test_countmin_ventana_expira():
    w = CountMinVentana(ventana_seg=600, bloque_seg=60)
    for t in range(10):
        w.agregar("u1", t)
    assert w.agregar("u1", 30) == 11
    assert w.agregar("u1", 2000) == 1   # la ventana anterior ya venció


def test_minhash_lsh():
    mh = MinHash(128); a = set(range(0, 100)); b = set(range(20, 120)); c = set(range(1000, 1100))
    fa, fb, fc = mh.firma(a), mh.firma(b), mh.firma(c)
    assert abs(jaccard_estimado(fa, fb) - jaccard(a, b)) < 0.15
    lsh = LSH(128, 32); lsh.insertar("a", fa); lsh.insertar("b", fb); lsh.insertar("c", fc)
    assert "b" in lsh.candidatos(fa) and "c" not in lsh.candidatos(fa)


def test_lsh_actualizar():
    mh = MinHash(128); lsh = LSH(128, 32)
    a = set(range(100)); lsh.insertar("x", mh.firma(set(range(1000, 1100))))
    lsh.insertar("a", mh.firma(a))
    assert "x" not in lsh.candidatos(mh.firma(a), excluir="a")
    lsh.actualizar("x", mh.firma(set(range(5, 100))))
    assert "x" in lsh.candidatos(mh.firma(a), excluir="a")


def test_preprocesador_invalidos_duplicados_tardios():
    from src.pipeline import Preprocesador
    base = {"user_session": "s1", "product_id": 1, "event_type": "view", "price": 10.0,
            "category_code": "A.B", "brand": "X"}
    lote1 = [dict(base, ts=100), dict(base, ts=100), dict(base, ts=101, price=-5), dict(base, ts=102, event_type="vew")]
    pre = Preprocesador()
    assert len(pre.procesar(lote1)) == 1 and pre.duplicados == 1 and pre.invalidos == 2
    out = pre.procesar([dict(base, ts=100), dict(base, ts=50, product_id=2)])
    assert pre.duplicados == 2 and pre.tardios == 1 and len(out) == 1


def test_hnsw_recall():
    rng = np.random.default_rng(0); X = rng.standard_normal((2000, 16)).astype(np.float32)
    h = HNSW(16, M=12, ef_construction=100)
    for v in X:
        h.insertar(v)
    rec = np.mean([recall_at_k(h.buscar(q, 10, ef=100), knn_coseno_exacto(X, q, 10)) for q in X[:50]])
    assert rec > 0.9, rec


def test_pagerank_y_comunidades():
    ordenes = {1: {1, 2, 3}, 2: {1, 2}, 3: {4, 5, 6}, 4: {5, 6}, 5: {3, 4}}
    pesos = construir_grafo(ordenes); A, nodos = a_matriz(pesos)
    pr, _ = pagerank(A)
    assert abs(pr.sum() - 1) < 1e-6
    G = a_networkx(pesos); r = comunidades(G, "louvain")
    assert r["n_comunidades"] == 2 and r["modularidad"] > 0.2
    assert nmi(r["comunidades"], r["comunidades"]) == 1.0


def test_trie():
    t = Trie(); t.insertar("samsung", 50); t.insertar("sony", 80); t.insertar("apple", 99)
    assert t.autocompletar("s", 2) == ["sony", "samsung"]


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f(); print("OK", nombre)
