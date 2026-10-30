"""Etapa 0 — Cola FIFO acotada para el stream de entrada.

Operaciones O(1): encolar al final y desencolar del inicio (collections.deque es una
lista doblemente enlazada de bloques). La capacidad máxima regula la presión cuando el
consumidor es más lento que el productor (backpressure)."""
from collections import deque


class ColaIngesta:
    def __init__(self, capacidad=100_000):
        self.capacidad = capacidad
        self._q = deque()
        self.encolados = 0
        self.rechazados = 0

    def encolar(self, evento):
        """O(1). Devuelve False si la cola está llena (el productor debe esperar)."""
        if len(self._q) >= self.capacidad:
            self.rechazados += 1
            return False
        self._q.append(evento)
        self.encolados += 1
        return True

    def desencolar_lote(self, tam):
        """Extrae hasta `tam` eventos en orden de llegada. O(tam)."""
        n = min(tam, len(self._q))
        return [self._q.popleft() for _ in range(n)]

    def __len__(self):
        return len(self._q)
