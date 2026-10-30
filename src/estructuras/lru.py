"""Etapa 3 — Caché LRU (Least Recently Used).

Tabla hash + lista doblemente enlazada (OrderedDict): get y put en O(1). Al superar la
capacidad se expulsa el elemento usado hace más tiempo."""
from collections import OrderedDict


class CacheLRU:
    def __init__(self, capacidad=10_000):
        self.capacidad = capacidad
        self._d = OrderedDict()
        self.aciertos = 0
        self.fallos = 0

    def get(self, clave, cargar=None):
        if clave in self._d:
            self._d.move_to_end(clave)
            self.aciertos += 1
            return self._d[clave]
        self.fallos += 1
        if cargar is None:
            return None
        valor = cargar(clave)
        self.put(clave, valor)
        return valor

    def put(self, clave, valor):
        self._d[clave] = valor
        self._d.move_to_end(clave)
        if len(self._d) > self.capacidad:
            self._d.popitem(last=False)

    @property
    def tasa_aciertos(self):
        total = self.aciertos + self.fallos
        return self.aciertos / total if total else 0.0
