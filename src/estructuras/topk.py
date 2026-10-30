"""Etapa 9 — Top-K con heap: los K elementos más frecuentes en O(n log K)."""
import heapq
from collections import Counter


class TopK:
    def __init__(self):
        self.c = Counter()

    def agregar(self, clave, peso=1):
        self.c[clave] += peso

    def top(self, k=10):
        return heapq.nlargest(k, self.c.items(), key=lambda x: x[1])
