"""Etapa 6 — Bloom Filter (Bloom, 1970).

Arreglo de m bits y k funciones hash (doble hashing: h_i = h1 + i·h2 mod m).
  insertar / consultar: O(k)          memoria: m bits, independiente del número de consultas
  falso positivo ≈ (1 − e^(−kn/m))^k   sin falsos negativos
Dimensionamiento óptimo para n elementos y tasa p:
  m = −n·ln p / (ln 2)²     k = (m/n)·ln 2"""
import math
from .hashing import hash128


class BloomFilter:
    def __init__(self, n_esperado, tasa_fp=0.01):
        self.m = max(8, int(math.ceil(-n_esperado * math.log(tasa_fp) / math.log(2) ** 2)))
        self.k = max(1, int(round(self.m / n_esperado * math.log(2))))
        self.bits = bytearray((self.m + 7) // 8)
        self.n = 0

    def _posiciones(self, clave):
        h1, h2 = hash128(clave)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def agregar(self, clave):
        for p in self._posiciones(clave):
            self.bits[p >> 3] |= 1 << (p & 7)
        self.n += 1

    def __contains__(self, clave):
        """True = posiblemente presente (verificar si la decisión es crítica). False = ausente seguro."""
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._posiciones(clave))

    def tasa_fp_teorica(self):
        return (1 - math.exp(-self.k * self.n / self.m)) ** self.k

    def memoria_bytes(self):
        return len(self.bits)
