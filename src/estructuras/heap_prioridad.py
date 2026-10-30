"""Etapa 2 — Cola con prioridades basada en un min-heap binario (heapq).

Clave: (nivel, timestamp, secuencia). Un nivel menor se atiende antes; dentro del mismo
nivel se respeta el orden temporal y la secuencia rompe empates (FIFO estable).
insertar y extraer: O(log n); consultar el mínimo: O(1)."""
import heapq
from itertools import count

NIVELES = {
    "alerta": 0,       # alertas de pago o fraude verificadas
    "express": 1,
    "estandar": 2,
    "programada": 3,
    "cart": 4,
    "search": 4,       # la búsqueda alimenta el autocompletado: más urgente que una vista
    "view": 5,
}


def nivel_evento(ev):
    if ev.get("alerta"):
        return NIVELES["alerta"]
    if ev["event_type"] == "purchase":
        return NIVELES.get(ev.get("order_type") or "estandar", 2)
    return NIVELES.get(ev["event_type"], 5)


class ColaPrioridad:
    def __init__(self):
        self._h = []
        self._seq = count()

    def insertar(self, ev, nivel=None):
        nivel = nivel_evento(ev) if nivel is None else nivel
        heapq.heappush(self._h, (nivel, ev["ts"], next(self._seq), ev))

    def extraer(self):
        nivel, _, _, ev = heapq.heappop(self._h)
        ev["prioridad"] = nivel
        return ev

    def __len__(self):
        return len(self._h)
