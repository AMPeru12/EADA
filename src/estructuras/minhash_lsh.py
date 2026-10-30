"""Etapa 7 — MinHash (Broder, 1997) + LSH por bandas (Indyk y Motwani, 1998).

P[min h(A) = min h(B)] = Jaccard(A, B). Con k funciones se obtiene una firma de k valores
(error ~ 1/√k). LSH divide la firma en b bandas de r filas: dos conjuntos con similitud s
son candidatos con probabilidad 1 − (1 − s^r)^b; umbral ≈ (1/b)^(1/r).
  firma: O(k·|S|)     consulta: sublineal (solo se comparan los candidatos)"""
from collections import defaultdict
import numpy as np

_P = (1 << 31) - 1


class MinHash:
    def __init__(self, num_perm=128, seed=42):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _P, num_perm, dtype=np.int64)[:, None]
        self.b = rng.integers(0, _P, num_perm, dtype=np.int64)[:, None]

    def firma(self, elementos):
        """elementos: iterable de enteros no negativos."""
        S = np.fromiter(elementos, dtype=np.int64) % _P
        return ((self.a * S[None, :] + self.b) % _P).min(axis=1)


def jaccard_estimado(f1, f2):
    return float(np.mean(f1 == f2))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


class LSH:
    def __init__(self, num_perm=128, bandas=32):
        if num_perm % bandas:
            raise ValueError("num_perm debe ser múltiplo de bandas")
        self.b, self.r = bandas, num_perm // bandas
        self.tablas = [defaultdict(list) for _ in range(bandas)]
        self.firmas = {}

    def umbral(self):
        return (1 / self.b) ** (1 / self.r)

    def _claves(self, firma):
        return [firma[i * self.r:(i + 1) * self.r].tobytes() for i in range(self.b)]

    def insertar(self, id_, firma):
        self.firmas[id_] = firma
        for t, c in zip(self.tablas, self._claves(firma)):
            t[c].append(id_)

    def eliminar(self, id_):
        firma = self.firmas.pop(id_, None)
        if firma is None:
            return
        for t, c in zip(self.tablas, self._claves(firma)):
            cubeta = t.get(c)
            if cubeta is not None:
                cubeta.remove(id_)
                if not cubeta:
                    del t[c]

    def actualizar(self, id_, firma):
        """Reemplaza la firma de un elemento cuyo conjunto cambió (refresco por micro-lote)."""
        self.eliminar(id_)
        self.insertar(id_, firma)

    def candidatos(self, firma, excluir=None):
        out = set()
        for t, c in zip(self.tablas, self._claves(firma)):
            out.update(t.get(c, ()))
        out.discard(excluir)
        return out

    def similares(self, id_, k=10):
        """Top-k candidatos ordenados por Jaccard estimado con la firma."""
        f = self.firmas[id_]
        cands = self.candidatos(f, excluir=id_)
        return sorted(((jaccard_estimado(f, self.firmas[c]), c) for c in cands), reverse=True)[:k]
