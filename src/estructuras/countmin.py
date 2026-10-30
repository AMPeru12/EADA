"""Etapa 6 — Count-Min Sketch (Cormode y Muthukrishnan, 2005) y versión por ventana temporal.

Matriz de d filas × w contadores. Actualizar suma en una celda por fila; estimar toma el
mínimo. Nunca subestima; con w = ⌈e/ε⌉ y d = ⌈ln(1/δ)⌉ sobreestima a lo más ε·N con
probabilidad 1 − δ.   actualizar / estimar: O(d)    memoria: O(w·d)"""
import math
import numpy as np
from .hashing import hash64

_P = (1 << 61) - 1  # primo de Mersenne


class CountMinSketch:
    def __init__(self, epsilon=0.001, delta=0.01, seed=42):
        self.w = int(math.ceil(math.e / epsilon))
        self.d = int(math.ceil(math.log(1 / delta)))
        rng = np.random.default_rng(seed)
        self.a = [int(x) for x in rng.integers(1, _P, self.d, dtype=np.int64)]
        self.b = [int(x) for x in rng.integers(0, _P, self.d, dtype=np.int64)]
        self.tabla = np.zeros((self.d, self.w), dtype=np.int32)
        self.total = 0

    def indices(self, clave):
        x = hash64(clave)
        return [((a * x + b) % _P) % self.w for a, b in zip(self.a, self.b)]

    def agregar(self, clave, c=1, idx=None):
        idx = self.indices(clave) if idx is None else idx
        for fila, col in enumerate(idx):
            self.tabla[fila, col] += c
        self.total += c

    def estimar(self, clave, idx=None):
        idx = self.indices(clave) if idx is None else idx
        return int(min(self.tabla[fila, col] for fila, col in enumerate(idx)))

    def memoria_bytes(self):
        return self.tabla.nbytes


class CountMinVentana:
    """Ventana deslizante de `ventana_seg` segundos formada por sketches de `bloque_seg` segundos.
    La estimación de un evento suma los bloques de su propia ventana (según su timestamp).
    Se conservan además `tolerancia_seg` segundos de bloques para aceptar eventos que llegan
    tarde o que el heap atiende fuera de orden; lo más antiguo se descarta."""

    def __init__(self, ventana_seg=600, bloque_seg=60, epsilon=0.001, delta=0.01, seed=42, tolerancia_seg=7200):
        self.bloque_seg = bloque_seg
        self.n_bloques = ventana_seg // bloque_seg
        self.n_retener = self.n_bloques + tolerancia_seg // bloque_seg
        self.eps, self.delta, self.seed = epsilon, delta, seed
        self.bloques = {}  # id_bloque -> CountMinSketch
        self.bloque_max = None
        self.tardios_descartados = 0
        self._plantilla = CountMinSketch(epsilon, delta, seed)  # mismas funciones hash en todos los bloques

    def _bloque(self, ts):
        b = int(ts // self.bloque_seg)
        if self.bloque_max is not None and b <= self.bloque_max - self.n_retener:
            return None                      # llegó después de la tolerancia de retraso
        if b not in self.bloques:
            cms = CountMinSketch(self.eps, self.delta, self.seed)
            self.bloques[b] = cms
            self.bloque_max = b if self.bloque_max is None else max(self.bloque_max, b)
            for viejo in [k for k in self.bloques if k <= self.bloque_max - self.n_retener]:
                del self.bloques[viejo]
        return b

    def agregar(self, clave, ts):
        """Registra la ocurrencia y devuelve la estimación en la ventana que termina en ts."""
        b = self._bloque(ts)
        if b is None:
            self.tardios_descartados += 1
            return 0
        idx = self._plantilla.indices(clave)
        self.bloques[b].agregar(clave, idx=idx)
        return sum(cms.estimar(clave, idx=idx) for k, cms in self.bloques.items() if b - self.n_bloques < k <= b)

    def memoria_bytes(self):
        return sum(c.memoria_bytes() for c in self.bloques.values())
