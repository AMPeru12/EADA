"""Etapa 9 — Trie para autocompletado ponderado por popularidad.
insertar: O(L)     autocompletar: O(L + tamaño del subárbol)"""
import heapq


class _NodoTrie:
    __slots__ = ("hijos", "peso", "palabra")

    def __init__(self):
        self.hijos = {}
        self.peso = 0
        self.palabra = None


class Trie:
    def __init__(self):
        self.raiz = _NodoTrie()
        self.n = 0

    def insertar(self, palabra, peso=1):
        nodo = self.raiz
        for ch in palabra.lower():
            nodo = nodo.hijos.setdefault(ch, _NodoTrie())
        if nodo.palabra is None:
            self.n += 1
        nodo.palabra = palabra
        nodo.peso += peso

    def autocompletar(self, prefijo, k=5):
        nodo = self.raiz
        for ch in prefijo.lower():
            nodo = nodo.hijos.get(ch)
            if nodo is None:
                return []
        out, pila = [], [nodo]
        while pila:
            x = pila.pop()
            if x.palabra is not None:
                out.append((x.peso, x.palabra))
            pila.extend(x.hijos.values())
        return [p for _, p in heapq.nlargest(k, out)]
