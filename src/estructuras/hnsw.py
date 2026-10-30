"""Etapa 7 — HNSW: Hierarchical Navigable Small World (Malkov y Yashunin, 2020).

Grafo de proximidad en capas: las superiores tienen pocos nodos y enlaces largos; la capa
0 contiene todos los puntos. La búsqueda desciende de forma voraz y refina en la capa 0
con una lista de candidatos de tamaño ef.
  inserción: ~O(log n)   búsqueda: ~O(log n) empírico   memoria: O(n·M)
Distancia: coseno (1 − <u, v>) sobre vectores normalizados."""
import heapq
import math
import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=200, seed=42, capacidad=1024):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.efc = ef_construction
        self.mL = 1 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((capacidad, dim), dtype=np.float32)
        self.n = 0
        self.grafo = []          # grafo[nivel] = {nodo: [vecinos]}
        self.entrada = None
        self.nivel_max = -1

    @staticmethod
    def _normalizar(v):
        v = np.asarray(v, dtype=np.float32)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    def _dist(self, q, ids):
        return 1.0 - self.X[ids] @ q

    def _buscar_capa(self, q, entradas, ef, nivel):
        visitados = set(entradas)
        d = self._dist(q, entradas)
        cand = [(float(di), i) for di, i in zip(d, entradas)]
        heapq.heapify(cand)
        res = [(-dist, i) for dist, i in cand]
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        vecinos_nivel = self.grafo[nivel]
        while cand:
            dc, c = heapq.heappop(cand)
            if dc > -res[0][0] and len(res) >= ef:
                break
            nuevos = [v for v in vecinos_nivel.get(c, ()) if v not in visitados]
            if not nuevos:
                continue
            visitados.update(nuevos)
            for dv, v in zip(self._dist(q, nuevos), nuevos):
                dv = float(dv)
                if len(res) < ef or dv < -res[0][0]:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(res, (-dv, v))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, i) for d, i in res)

    def insertar(self, v):
        v = self._normalizar(v)
        if self.n == len(self.X):
            self.X = np.vstack([self.X, np.zeros_like(self.X)])
        idx = self.n
        self.X[idx] = v
        self.n += 1
        nivel = int(-math.log(self.rng.random() + 1e-12) * self.mL)
        while len(self.grafo) <= nivel:
            self.grafo.append({})
        if self.entrada is None:
            for l in range(nivel + 1):
                self.grafo[l][idx] = []
            self.entrada, self.nivel_max = idx, nivel
            return idx
        ep = [self.entrada]
        for l in range(self.nivel_max, nivel, -1):
            ep = [self._buscar_capa(v, ep, 1, l)[0][1]]
        for l in range(min(nivel, self.nivel_max), -1, -1):
            W = self._buscar_capa(v, ep, self.efc, l)
            vecinos = [i for _, i in W[:self.M]]
            self.grafo[l][idx] = list(vecinos)
            mmax = self.M0 if l == 0 else self.M
            for nb in vecinos:
                lst = self.grafo[l][nb]
                lst.append(idx)
                if len(lst) > mmax:  # poda: conserva los mmax más cercanos a nb
                    d = self._dist(self.X[nb], lst)
                    self.grafo[l][nb] = [lst[j] for j in np.argsort(d)[:mmax]]
            ep = [i for _, i in W]
        for l in range(self.nivel_max + 1, nivel + 1):
            self.grafo[l][idx] = []
        if nivel > self.nivel_max:
            self.entrada, self.nivel_max = idx, nivel
        return idx

    def buscar(self, q, k=10, ef=50):
        """Devuelve [(distancia, índice)] de los k vecinos aproximados."""
        q = self._normalizar(q)
        ep = [self.entrada]
        for l in range(self.nivel_max, 0, -1):
            ep = [self._buscar_capa(q, ep, 1, l)[0][1]]
        return self._buscar_capa(q, ep, max(ef, k), 0)[:k]

    def memoria_bytes(self):
        aristas = sum(len(v) for capa in self.grafo for v in capa.values())
        return self.X[:self.n].nbytes + aristas * 8


def knn_coseno_exacto(X, q, k=10):
    """Línea base por fuerza bruta: O(n·d)."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q = q / max(np.linalg.norm(q), 1e-12)
    d = 1.0 - Xn @ q
    idx = np.argpartition(d, min(k, len(d) - 1))[:k]
    return sorted((float(d[i]), int(i)) for i in idx)


def recall_at_k(aprox, exacto):
    a = {i for _, i in aprox}
    e = {i for _, i in exacto}
    return len(a & e) / len(e) if e else 1.0
