"""Etapa 4 — B+Tree para un índice de disco.

Cada nodo representa una página de disco con hasta `orden` claves. Los datos viven solo
en las hojas, que están enlazadas para recorrer rangos. Se cuenta cada nodo visitado
como una lectura de página.

  búsqueda puntual: O(log_m n)     consulta por rango: O(log_m n + k)
  inserción:        O(log_m n) con divisiones de nodo amortizadas"""
from bisect import bisect_left, bisect_right


class _Nodo:
    __slots__ = ("hoja", "claves", "hijos", "valores", "sig")

    def __init__(self, hoja):
        self.hoja = hoja
        self.claves = []
        self.hijos = []      # nodos internos
        self.valores = []    # hojas
        self.sig = None      # hoja siguiente


class BPlusTree:
    def __init__(self, orden=64):
        if orden < 3:
            raise ValueError("orden >= 3")
        self.orden = orden
        self.raiz = _Nodo(hoja=True)
        self.n = 0
        self.lecturas_pagina = 0

    # ---------------- búsqueda ----------------
    def _hoja_para(self, clave):
        nodo = self.raiz
        self.lecturas_pagina += 1
        while not nodo.hoja:
            nodo = nodo.hijos[bisect_right(nodo.claves, clave)]
            self.lecturas_pagina += 1
        return nodo

    def buscar(self, clave):
        hoja = self._hoja_para(clave)
        i = bisect_left(hoja.claves, clave)
        if i < len(hoja.claves) and hoja.claves[i] == clave:
            return hoja.valores[i]
        return None

    def rango(self, desde, hasta):
        """Devuelve los (clave, valor) con desde <= clave <= hasta, en orden."""
        hoja = self._hoja_para(desde)
        i = bisect_left(hoja.claves, desde)
        out = []
        while hoja is not None:
            while i < len(hoja.claves):
                if hoja.claves[i] > hasta:
                    return out
                out.append((hoja.claves[i], hoja.valores[i]))
                i += 1
            hoja, i = hoja.sig, 0
            if hoja is not None:
                self.lecturas_pagina += 1
        return out

    # ---------------- inserción ----------------
    def insertar(self, clave, valor):
        division = self._insertar(self.raiz, clave, valor)
        if division is not None:
            separador, nuevo = division
            raiz = _Nodo(hoja=False)
            raiz.claves = [separador]
            raiz.hijos = [self.raiz, nuevo]
            self.raiz = raiz

    def _insertar(self, nodo, clave, valor):
        if nodo.hoja:
            i = bisect_left(nodo.claves, clave)
            if i < len(nodo.claves) and nodo.claves[i] == clave:
                nodo.valores[i] = valor          # actualización
                return None
            nodo.claves.insert(i, clave)
            nodo.valores.insert(i, valor)
            self.n += 1
            if len(nodo.claves) > self.orden:
                return self._dividir_hoja(nodo)
            return None
        i = bisect_right(nodo.claves, clave)
        division = self._insertar(nodo.hijos[i], clave, valor)
        if division is None:
            return None
        separador, nuevo = division
        nodo.claves.insert(i, separador)
        nodo.hijos.insert(i + 1, nuevo)
        if len(nodo.claves) > self.orden:
            return self._dividir_interno(nodo)
        return None

    def _dividir_hoja(self, hoja):
        mitad = len(hoja.claves) // 2
        nueva = _Nodo(hoja=True)
        nueva.claves, hoja.claves = hoja.claves[mitad:], hoja.claves[:mitad]
        nueva.valores, hoja.valores = hoja.valores[mitad:], hoja.valores[:mitad]
        nueva.sig, hoja.sig = hoja.sig, nueva
        return nueva.claves[0], nueva

    def _dividir_interno(self, nodo):
        mitad = len(nodo.claves) // 2
        separador = nodo.claves[mitad]
        nuevo = _Nodo(hoja=False)
        nuevo.claves, nodo.claves = nodo.claves[mitad + 1:], nodo.claves[:mitad]
        nuevo.hijos, nodo.hijos = nodo.hijos[mitad + 1:], nodo.hijos[:mitad + 1]
        return separador, nuevo

    def eliminar(self, clave):
        """Eliminación perezosa: borra la clave de su hoja sin redistribuir ni fusionar nodos.
        Mantiene la correctitud de búsquedas y rangos; las hojas pueden quedar con menos
        claves que el mínimo (se compactarían en una reconstrucción periódica). O(log_m n)."""
        hoja = self._hoja_para(clave)
        i = bisect_left(hoja.claves, clave)
        if i < len(hoja.claves) and hoja.claves[i] == clave:
            del hoja.claves[i]
            del hoja.valores[i]
            self.n -= 1
            return True
        return False

    def altura(self):
        h, nodo = 1, self.raiz
        while not nodo.hoja:
            nodo, h = nodo.hijos[0], h + 1
        return h

    def __len__(self):
        return self.n
