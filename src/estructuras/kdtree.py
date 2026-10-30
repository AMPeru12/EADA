"""Etapa 5 — KD-Tree y KNN exacto.

Árbol binario que divide el espacio alternando dimensiones por la mediana.
  construcción: O(n log n)     consulta KNN: ~O(log n) en baja dimensión
En alta dimensión la poda deja de funcionar y la consulta se acerca a O(n)
(maldición de la dimensionalidad); por eso solo se usa con atributos de baja dimensión."""
import heapq
import numpy as np


class KDTree:
    def __init__(self, puntos, hoja_max=16):
        self.X = np.asarray(puntos, dtype=np.float64)
        self.hoja_max = hoja_max
        self.d = self.X.shape[1]
        self.nodos_visitados = 0
        self.raiz = self._construir(np.arange(len(self.X)), 0)

    def _construir(self, idx, prof):
        if len(idx) <= self.hoja_max:
            return ("hoja", idx)
        eje = prof % self.d
        orden = idx[np.argsort(self.X[idx, eje], kind="stable")]
        m = len(orden) // 2
        return ("nodo", eje, self.X[orden[m], eje],
                self._construir(orden[:m], prof + 1), self._construir(orden[m:], prof + 1))

    def knn(self, q, k=5):
        """Devuelve [(distancia, índice)] de los k vecinos más cercanos (exacto)."""
        q = np.asarray(q, dtype=np.float64)
        mejores = []  # max-heap con distancias negadas
        self._buscar(self.raiz, q, k, mejores)
        return sorted((-d, i) for d, i in mejores)

    def _buscar(self, nodo, q, k, mejores):
        self.nodos_visitados += 1
        if nodo[0] == "hoja":
            idx = nodo[1]
            dist = np.sqrt(((self.X[idx] - q) ** 2).sum(axis=1))
            for d, i in zip(dist, idx):
                if len(mejores) < k:
                    heapq.heappush(mejores, (-d, int(i)))
                elif d < -mejores[0][0]:
                    heapq.heapreplace(mejores, (-d, int(i)))
            return
        _, eje, corte, izq, der = nodo
        diff = q[eje] - corte
        cerca, lejos = (izq, der) if diff < 0 else (der, izq)
        self._buscar(cerca, q, k, mejores)
        if len(mejores) < k or abs(diff) < -mejores[0][0]:
            self._buscar(lejos, q, k, mejores)


def knn_fuerza_bruta(X, q, k=5):
    """Línea base exacta O(n·d)."""
    dist = np.sqrt(((np.asarray(X) - q) ** 2).sum(axis=1))
    idx = np.argpartition(dist, min(k, len(dist) - 1))[:k]
    return sorted((float(dist[i]), int(i)) for i in idx)
