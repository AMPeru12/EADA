"""Pipeline de 10 etapas para el escenario B (comercio electrónico).

Cada evento del stream atraviesa, en orden de llegada:
  0 ingesta (cola FIFO) → 1 preprocesamiento (validar, normalizar, deduplicar, detectar tardíos)
  → 2 heap de prioridad → 6 filtrado probabilístico (Bloom Filter + Count-Min Sketch)
  → 3 almacenamiento (tabla hash, LRU, stock, matriz usuario-producto, órdenes)
  → 4 B+Tree (se actualiza evento a evento) y 9 Trie de búsquedas (evento a evento).
Cada `refresco_eventos` eventos (micro-lote) se actualizan las etapas costosas:
  5 KD-Tree, 7 MinHash+LSH (solo productos modificados) y SVD+HNSW, 8 grafo y PageRank.
Al cerrar el stream se calculan además las comunidades y su estabilidad.

Uso:
  python -m src.pipeline --stream data/stream_10M.parquet --blacklist data/blacklist.csv \
                         --stock data/catalogo_stock.csv --n 200000
"""
import argparse, json, math, os, time
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .estructuras.cola import ColaIngesta
from .estructuras.heap_prioridad import ColaPrioridad
from .estructuras.lru import CacheLRU
from .estructuras.bplustree import BPlusTree
from .estructuras.kdtree import KDTree
from .estructuras.bloom import BloomFilter
from .estructuras.countmin import CountMinVentana
from .estructuras.minhash_lsh import MinHash, LSH
from .estructuras.hnsw import HNSW
from .estructuras import grafo as gr
from .estructuras.trie import Trie
from .estructuras.topk import TopK

TIPOS_PRODUCTO = {"view", "cart", "purchase"}
PESO_INTERACCION = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
MOTIVOS_FRAUDE = {"lista_negra", "intensidad", "card_testing"}


# ============================================================ utilidades
def cargar_stream(ruta, n=None):
    """Lee el stream (ya ordenado por llegada). `ts` = momento en que ocurrió el evento."""
    df = pd.read_parquet(ruta) if ruta.endswith(".parquet") else pd.read_pickle(ruta)
    if n:
        df = df.head(n)
    df = df.copy()
    df["ts"] = df["event_time"].astype("int64") // 1_000_000_000
    df["event_type"] = df["event_type"].astype(str)
    return df


def eventos(df):
    cols = list(df.columns)
    for fila in df.itertuples(index=False, name=None):
        yield {c: (None if (v is pd.NA or (isinstance(v, float) and math.isnan(v) and c != "price")) else v)
               for c, v in zip(cols, fila)}


def _texto(v):
    return v.strip().lower() if isinstance(v, str) and v.strip() else None


# ============================================================ etapa 1
class Preprocesador:
    """Valida, normaliza y deduplica por lotes. El lote se ordena (Timsort, O(b log b)),
    se eliminan duplicados consecutivos y se usa búsqueda binaria (O(log b)) contra las
    claves del lote anterior. Los eventos con ts menor que la marca de agua (máximo ts ya
    procesado) se cuentan como tardíos y se procesan igual (tolerancia de retraso)."""

    def __init__(self):
        self.claves_previas = []
        self.marca_agua = None
        self.invalidos = 0
        self.invalidos_por_motivo = Counter()
        self.duplicados = 0
        self.tardios = 0

    @staticmethod
    def clave(ev):
        return (ev["user_session"], ev["product_id"], ev["event_type"], ev["ts"])

    def _motivo_invalido(self, ev):
        if not isinstance(ev.get("user_session"), str) or not ev["user_session"]:
            return "sesion_vacia"
        if ev["event_type"] == "search":
            return None if _texto(ev.get("query")) else "busqueda_vacia"
        if ev["event_type"] not in TIPOS_PRODUCTO:
            return "tipo_desconocido"
        if ev.get("price") is None or not (ev["price"] > 0):
            return "precio_invalido"
        if not ev.get("product_id"):
            return "producto_vacio"
        return None

    @staticmethod
    def _normalizar(ev):
        if ev["event_type"] == "search":
            ev["query"] = _texto(ev["query"])
            return ev
        ev["category_code"] = _texto(ev.get("category_code")) or "sin_categoria"
        ev["brand"] = _texto(ev.get("brand")) or "sin_marca"
        ev["price"] = round(float(ev["price"]), 2)
        return ev

    def procesar(self, lote):
        validos = []
        for ev in lote:
            motivo = self._motivo_invalido(ev)
            if motivo:
                self.invalidos += 1
                self.invalidos_por_motivo[motivo] += 1
            else:
                validos.append(self._normalizar(ev))
        validos.sort(key=self.clave)
        limpio, ultima = [], None
        for ev in validos:
            k = self.clave(ev)
            if k == ultima:
                self.duplicados += 1
                continue
            i = bisect_left(self.claves_previas, k)
            if i < len(self.claves_previas) and self.claves_previas[i] == k:
                self.duplicados += 1
                continue
            ultima = k
            limpio.append(ev)
            if self.marca_agua is not None and ev["ts"] < self.marca_agua:
                self.tardios += 1
        self.claves_previas = [self.clave(e) for e in limpio]
        if limpio:
            mx = max(e["ts"] for e in limpio)
            self.marca_agua = mx if self.marca_agua is None else max(self.marca_agua, mx)
        return limpio


# ============================================================ etapa 6
class FiltroFraude:
    """Bloom Filter sobre entidades señaladas + Count-Min Sketch por ventana deslizante.
    Ninguna operación se bloquea solo por la estructura probabilística: los positivos se
    verifican contra la fuente exacta (lista negra / registro de eventos recientes)."""

    def __init__(self, entidades, ventana_seg=600, umbral_cart=18, umbral_compra=6, umbral_rechazos=3, tasa_fp=0.01):
        self.exactas = set(entidades)                      # fuente exacta (en producción: BD)
        self.bloom = BloomFilter(max(1, len(self.exactas)), tasa_fp)
        for e in self.exactas:
            self.bloom.agregar(e)
        self.tolerancia = 7200                             # tolerancia a eventos tardíos o fuera de orden
        self.cms = CountMinVentana(ventana_seg=ventana_seg, bloque_seg=60, tolerancia_seg=self.tolerancia)
        self.ventana = ventana_seg
        self._ts_max = 0
        self.umbral = {"cart": umbral_cart, "purchase": umbral_compra, "rechazo": umbral_rechazos}
        self.recientes = defaultdict(deque)
        self.ultima_alerta = {}
        self.m = Counter()

    @staticmethod
    def entidades_evento(ev):
        return [f"user:{ev['user_id']}", f"payment_token:{ev['payment_token']}", f"device:{ev['device_id']}"]

    def _intensidad(self, clave, ts, umbral):
        """Count-Min Sketch estima; si supera el umbral se verifica con el conteo exacto."""
        est = self.cms.agregar(clave, ts)
        dq = self.recientes[clave]               # registro exacto (en producción: consulta a la BD)
        dq.append(ts)
        self._ts_max = max(self._ts_max, ts)
        while dq and dq[0] <= self._ts_max - self.ventana - self.tolerancia:
            dq.popleft()
        if est < umbral:
            return None, est
        self.m["cms_sobre_umbral"] += 1
        exacto = sum(1 for x in dq if ts - self.ventana < x <= ts)
        if exacto < umbral:
            self.m["cms_falsas_alertas"] += 1
            return None, est
        if ts - self.ultima_alerta.get(clave, -10 ** 12) >= self.ventana:
            self.ultima_alerta[clave] = ts
            return "alertar", est
        return "retener", est                               # ya alertado en esta ventana

    def evaluar(self, ev):
        """Devuelve (aprobado: bool, alerta: dict | None)."""
        self.m["evaluados"] += 1
        # 1) Lista negra
        for ent in self.entidades_evento(ev):
            if ent in self.bloom:
                self.m["bloom_positivos"] += 1
                if ent in self.exactas:
                    self.m["bloqueados_lista_negra"] += 1
                    return False, self._alerta(ev, "lista_negra", ent, None)
                self.m["bloom_falsos_positivos"] += 1
        tipo = ev["event_type"]
        # 2) Pago rechazado (alerta de pago) y repetición de rechazos (prueba de tarjetas)
        if tipo == "purchase" and ev.get("payment_status") == "rechazado":
            self.m["pagos_rechazados"] += 1
            decision, est = self._intensidad(f"{ev['payment_token']}|rechazo", ev["ts"], self.umbral["rechazo"])
            if decision == "alertar":
                self.m["alertas_card_testing"] += 1
                return False, self._alerta(ev, "card_testing", ev["payment_token"], est)
            return False, self._alerta(ev, "pago_rechazado", ev["payment_token"], None)
        # 3) Intensidad de carritos y compras por usuario
        if tipo in ("cart", "purchase"):
            decision, est = self._intensidad(f"{ev['user_id']}|{tipo}", ev["ts"], self.umbral[tipo])
            if decision == "alertar":
                self.m["alertas_intensidad"] += 1
                return False, self._alerta(ev, "intensidad", f"{ev['user_id']}|{tipo}", est)
            if decision == "retener":
                return False, None
        return True, None

    @staticmethod
    def _alerta(ev, motivo, clave, estimado):
        return {"alerta": True, "motivo": motivo, "clave": str(clave), "estimado": estimado,
                "user_id": int(ev["user_id"]), "event_type": ev["event_type"], "product_id": int(ev["product_id"]),
                "ts": ev["ts"], "es_fraude_real": bool(ev.get("is_fraud", False))}


# ============================================================ etapa 3 (+ 4 y 9 incrementales)
class Almacen:
    """Tabla hash de productos con stock, caché LRU, B+Tree incremental (categoría, precio),
    interacciones para la matriz CSR, sesiones por producto (MinHash) y órdenes (grafo)."""

    def __init__(self, stock_inicial=None, capacidad_lru=10_000):
        self.productos = {}
        self.stock_inicial = stock_inicial or {}
        self.cache = CacheLRU(capacidad_lru)
        self.bpt = BPlusTree(orden=64)
        self.usuarios, self.sesiones = {}, {}
        self.u, self.p, self.w = array("q"), array("q"), array("d")
        self.sesiones_producto = defaultdict(set)
        self.modificados = set()                 # productos con sesiones nuevas desde el último refresco
        self.ordenes = defaultdict(set)
        self.carritos = defaultdict(set)
        self.ventas = TopK()
        self.terminos = Counter()                # popularidad de categorías y marcas (para el Trie)
        self.ventas_sin_stock = 0
        self.cambios_precio = 0

    def registrar(self, ev):
        pid = int(ev["product_id"])
        tipo = ev["event_type"]
        if tipo == "view":
            self.producto(pid)                   # una vista = consulta de la ficha (pasa por la caché LRU)
        prod = self.productos.get(pid)
        if prod is None:
            prod = {"product_id": pid, "category_id": int(ev["category_id"]), "category_code": ev["category_code"],
                    "brand": ev["brand"], "price": ev["price"], "stock": int(self.stock_inicial.get(pid, 0)),
                    "view": 0, "cart": 0, "purchase": 0}
            self.productos[pid] = prod
            self.bpt.insertar((prod["category_id"], prod["price"], pid), pid)
        elif prod["price"] != ev["price"]:       # cambio de precio: se reindexa en el B+Tree
            self.bpt.eliminar((prod["category_id"], prod["price"], pid))
            prod["price"] = ev["price"]
            self.bpt.insertar((prod["category_id"], prod["price"], pid), pid)
            self.cambios_precio += 1
        prod[tipo] += 1
        if tipo == "view":
            self.terminos[prod["category_code"]] += 1
            self.terminos[prod["brand"]] += 1
        uid = self.usuarios.setdefault(int(ev["user_id"]), len(self.usuarios))
        sid = self.sesiones.setdefault(ev["user_session"], len(self.sesiones))
        self.u.append(uid); self.p.append(pid); self.w.append(PESO_INTERACCION[tipo])
        if tipo in ("view", "cart"):
            if sid not in self.sesiones_producto[pid]:
                self.sesiones_producto[pid].add(sid)
                self.modificados.add(pid)
        if tipo == "cart":
            self.carritos[ev["user_session"]].add(pid)
        elif tipo == "purchase":
            self.ordenes[ev["user_session"]].add(pid)
            self.ventas.agregar(pid)
            if prod["stock"] > 0:
                prod["stock"] -= 1
            else:
                self.ventas_sin_stock += 1

    def producto(self, pid):
        """Consulta puntual con caché LRU (el diccionario simula el acceso a disco/BD)."""
        return self.cache.get(pid, cargar=self.productos.get)

    def matriz_usuario_producto(self):
        pids = np.frombuffer(self.p, dtype=np.int64)
        prod_ids, cols = np.unique(pids, return_inverse=True)
        M = sp.coo_matrix((np.frombuffer(self.w, dtype=np.float64), (np.frombuffer(self.u, dtype=np.int64), cols)),
                          shape=(len(self.usuarios), len(prod_ids))).tocsr()
        return M, prod_ids


# ============================================================ pipeline
class Pipeline:
    def __init__(self, entidades_lista_negra, stock_inicial=None, cfg=None):
        self.cfg = {"tam_lote": 10_000, "capacidad_cola": 50_000, "capacidad_lru": 10_000,
                    "umbral_cart": 18, "umbral_compra": 6, "umbral_rechazos": 3, "ventana_seg": 600,
                    "refresco_eventos": 1_000_000, "min_sesiones_lsh": 5, "num_perm": 256, "bandas_lsh": 128,
                    "max_hnsw": 5_000, "dim_svd": 32, "semillas_estabilidad": 10, **(cfg or {})}
        c = self.cfg
        self.cola = ColaIngesta(c["capacidad_cola"])
        self.pre = Preprocesador()
        self.heap = ColaPrioridad()
        self.filtro = FiltroFraude(entidades_lista_negra, c["ventana_seg"], c["umbral_cart"], c["umbral_compra"],
                                   c["umbral_rechazos"])
        self.almacen = Almacen(stock_inicial, c["capacidad_lru"])
        self.trie = Trie()
        self.busquedas = TopK()
        self.minhash = MinHash(c["num_perm"])
        self.lsh = LSH(c["num_perm"], c["bandas_lsh"])   # 128 bandas × 2 filas: umbral ≈ 0.088 (elegido en el paso 5.3)
        self.alertas = []
        self.atendidos_por_nivel = Counter()
        self.tiempos = Counter()
        self.refrescos = []
        self.traza, self._seguido = [], None
        self.ingestados = 0
        self._desde_refresco = 0
        self.indices_listos = False

    # ---------------- etapas por evento: 0, 1, 2, 6, 3, 4 y 9 ----------------
    def procesar_stream(self, df):
        t = time.perf_counter()
        for ev in eventos(df):
            if self._seguido is None and ev["event_type"] == "purchase" and ev.get("order_type") == "express" \
                    and ev.get("payment_status") == "aprobado" and not ev.get("is_fraud") and not ev.get("synthetic"):
                self._seguido = (ev["user_session"], ev["product_id"], ev["ts"])
                self._t("0 Ingesta", f"compra express del producto {ev['product_id']} entra a la cola FIFO")
            self.ingestados += 1
            self._desde_refresco += 1
            if not self.cola.encolar(ev):
                self._procesar_lote(self.cola.desencolar_lote(self.cfg["tam_lote"]))
                self.cola.encolar(ev)
            if len(self.cola) >= self.cfg["tam_lote"]:
                self._procesar_lote(self.cola.desencolar_lote(self.cfg["tam_lote"]))
                if self._desde_refresco >= self.cfg["refresco_eventos"]:
                    self.tiempos["etapas por evento"] += time.perf_counter() - t
                    self.refrescar(final=False)
                    t = time.perf_counter()
        if len(self.cola):
            self._procesar_lote(self.cola.desencolar_lote(len(self.cola)))
        self.tiempos["etapas por evento"] += time.perf_counter() - t
        self.refrescar(final=True)

    def _es_seguido(self, ev):
        return self._seguido is not None and (ev["user_session"], ev["product_id"], ev["ts"]) == self._seguido

    def _procesar_lote(self, lote):
        limpio = self.pre.procesar(lote)
        for ev in limpio:
            if self._es_seguido(ev):
                self._t("1 Preprocesamiento", "pasa la validación, la normalización y la deduplicación")
            self.heap.insertar(ev)
        while len(self.heap):
            ev = self.heap.extraer()
            self.atendidos_por_nivel[ev["prioridad"]] += 1
            if ev.get("alerta"):                      # alerta verificada: se atiende con prioridad 0
                self.alertas.append(ev)
                continue
            seguido = self._es_seguido(ev)
            if seguido:
                self._t("2 Heap", f"se atiende con prioridad {ev['prioridad']} (express), antes que estándar y vistas")
            aprobado, alerta = self.filtro.evaluar(ev)
            if alerta is not None:
                self.heap.insertar(alerta, nivel=0)
            if not aprobado:
                continue
            if ev["event_type"] == "search":          # etapa 9 incremental: autocompletado con búsquedas reales
                self.trie.insertar(ev["query"], 1)
                self.busquedas.agregar(ev["query"])
                continue
            self.almacen.registrar(ev)
            if seguido:
                p = self.almacen.productos[int(ev["product_id"])]
                self._t("6 Filtrado", "Bloom Filter: entidades no señaladas · pago aprobado · Count-Min Sketch bajo el umbral → aprobado")
                self._t("3 Almacenamiento", f"actualiza ficha, stock (quedan {p['stock']}), matriz usuario-producto y órdenes")
                self._t("4 B+Tree", f"producto indexado por (categoría, precio = {p['price']}) para consultas por rango")

    def _t(self, etapa, detalle):
        self.traza.append({"etapa": etapa, "detalle": detalle})

    # ---------------- micro-lote: etapas 5, 7, 8 (y comunidades al final) ----------------
    def refrescar(self, final=False):
        a, c = self.almacen, self.cfg
        prods = a.productos
        if not prods:
            return
        tiempos = {}

        t = time.perf_counter()                                   # 5) KD-Tree sobre atributos
        self.kd_ids = np.array([pid for pid, p in prods.items() if p["view"] > 0])
        if len(self.kd_ids):
            F = np.array([[math.log1p(prods[i]["price"]), math.log1p(prods[i]["view"]),
                           prods[i]["purchase"] / prods[i]["view"], prods[i]["cart"] / prods[i]["view"]] for i in self.kd_ids])
            self.kd_F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
            self.kd = KDTree(self.kd_F)
        self.kd_pos = {int(pid): i for i, pid in enumerate(self.kd_ids)}
        tiempos["5 KD-Tree"] = time.perf_counter() - t

        t = time.perf_counter()                                   # 7a) MinHash+LSH incremental
        actualizados = 0
        for pid in a.modificados:
            ses = a.sesiones_producto[pid]
            if len(ses) >= c["min_sesiones_lsh"]:
                self.lsh.actualizar(pid, self.minhash.firma(ses))
                actualizados += 1
        a.modificados.clear()
        tiempos["7 MinHash+LSH"] = time.perf_counter() - t

        t = time.perf_counter()                                   # 7b) SVD + HNSW (reconstrucción)
        M, prod_ids = a.matriz_usuario_producto()
        self.densidad = M.nnz / (M.shape[0] * M.shape[1])
        k = max(2, min(c["dim_svd"], min(M.shape) - 1))
        _, s, Vt = svds(M.astype(np.float64), k=k)
        emb = (Vt.T * s).astype(np.float32)
        top = np.argsort(-np.asarray(M.sum(axis=0)).ravel())[:c["max_hnsw"]]
        self.hnsw_ids, self.emb = prod_ids[top], emb[top]
        self.hnsw = HNSW(k, M=16, ef_construction=100)
        for v in self.emb:
            self.hnsw.insertar(v)
        self.hnsw_pos = {int(pid): i for i, pid in enumerate(self.hnsw_ids)}
        tiempos["7 SVD+HNSW"] = time.perf_counter() - t

        t = time.perf_counter()                                   # 8) grafo de co-compra y PageRank
        pesos = gr.construir_grafo(a.ordenes)
        self.grafo_fuente = "compras"
        if len(pesos) < 1000:                                     # grafo disperso: se suman carritos
            for e, w in gr.construir_grafo(a.carritos).items():
                pesos[e] += 0.5 * w
            self.grafo_fuente = "compras + carritos (peso 0.5)"
        self.pesos = pesos
        self.A, self.nodos = gr.a_matriz(pesos)
        self.nodo_pos = {n: i for i, n in enumerate(self.nodos)}
        self.pr, self.pr_iter = gr.pagerank(self.A) if len(self.nodos) else (np.array([]), 0)
        tiempos["8 Grafo y PageRank"] = time.perf_counter() - t

        t = time.perf_counter()                                   # 9) términos del catálogo al Trie
        for termino, n in a.terminos.items():
            if termino not in ("sin_categoria", "sin_marca"):
                self.trie.insertar(termino, n)
        a.terminos.clear()
        tiempos["9 Trie (catálogo)"] = time.perf_counter() - t

        if final:                                                 # 8) comunidades y estabilidad
            t = time.perf_counter()
            G = gr.a_networkx(pesos)
            self.com = {m: gr.comunidades(G, m) for m in ("louvain", "label_propagation")} if len(G) else {}
            sem = range(c["semillas_estabilidad"])
            self.estab = {m: gr.estabilidad(G, m, sem) for m in self.com} if len(G) else {}
            tiempos["8 Comunidades"] = time.perf_counter() - t

        for k_, v in tiempos.items():
            self.tiempos[k_] += v
        self.indices_listos = True
        foto = {"eventos": self.ingestados, "final": final, "productos": len(prods),
                "lsh_actualizados": actualizados, "lsh_indexados": len(self.lsh.firmas),
                "hnsw": self.hnsw.n, "aristas_grafo": len(pesos),
                "top3_pagerank": [int(self.nodos[i]) for i in np.argsort(-self.pr)[:3]] if len(self.pr) else [],
                "segundos": round(sum(tiempos.values()), 2)}
        if self._seguido is not None and self._seguido[1] in prods:
            foto["recomendacion_co_compra_producto_seguido"] = self.recomendar(self._seguido[1])["co_compra_ppr"]
        self.refrescos.append(foto)
        self._desde_refresco = 0

    # ---------------- servicio (etapa 9) ----------------
    def autocompletar(self, prefijo, k=5):
        return self.trie.autocompletar(prefijo, k)

    def productos_por_rango(self, category_id, pmin, pmax, solo_disponibles=True):
        pids = [pid for _, pid in self.almacen.bpt.rango((category_id, pmin, -1), (category_id, pmax, 2 ** 62))]
        if solo_disponibles:
            pids = [p for p in pids if self.almacen.productos[p]["stock"] > 0]
        return pids

    def recomendar(self, pid, k=5):
        pid = int(pid)
        out = {"producto": self.almacen.producto(pid)}
        out["similares_comportamiento_lsh"] = [int(x) for _, x in self.lsh.similares(pid, k)] if pid in self.lsh.firmas else []
        if pid in self.hnsw_pos:
            v = self.emb[self.hnsw_pos[pid]]
            out["complementarios_hnsw"] = [int(self.hnsw_ids[i]) for _, i in self.hnsw.buscar(v, k + 1, ef=50)
                                           if self.hnsw_ids[i] != pid][:k]
        else:
            out["complementarios_hnsw"] = []
        if pid in self.nodo_pos:
            v = np.zeros(len(self.nodos)); v[self.nodo_pos[pid]] = 1
            ppr, _ = gr.pagerank(self.A, personalizacion=v)
            out["co_compra_ppr"] = [int(self.nodos[i]) for i in np.argsort(-ppr) if self.nodos[i] != pid][:k]
        else:
            out["co_compra_ppr"] = []
        if pid in self.kd_pos:
            out["atributos_knn_exacto"] = [int(self.kd_ids[i]) for _, i in self.kd.knn(self.kd_F[self.kd_pos[pid]], k + 1)
                                           if self.kd_ids[i] != pid][:k]
        return out

    def resumen(self):
        a, f = self.almacen, self.filtro
        top_pr = [int(self.nodos[i]) for i in np.argsort(-self.pr)[:10]] if len(self.pr) else []
        grado = gr.ranking_grado(self.A) if len(self.nodos) else np.array([])
        por_motivo = Counter(x["motivo"] for x in self.alertas)
        fraude = [x for x in self.alertas if x["motivo"] in MOTIVOS_FRAUDE]
        return {
            "eventos_ingestados": self.ingestados,
            "preprocesamiento": {"invalidos": self.pre.invalidos, "invalidos_por_motivo": dict(self.pre.invalidos_por_motivo),
                                 "duplicados_eliminados": self.pre.duplicados, "tardios_procesados": self.pre.tardios,
                                 "tardios_fuera_de_ventana_cms": f.cms.tardios_descartados},
            "atendidos_por_nivel_heap": {str(k): v for k, v in sorted(self.atendidos_por_nivel.items())},
            "filtrado": dict(f.m),
            "alertas": {"total": len(self.alertas), "por_motivo": dict(por_motivo),
                        "precision_fraude_vs_inyectado": round(float(np.mean([x["es_fraude_real"] for x in fraude])), 3) if fraude else None,
                        "ejemplos": [x for m in ("lista_negra", "intensidad", "card_testing", "pago_rechazado")
                                     for x in self.alertas if x["motivo"] == m][:8]},
            "memoria_estructuras_bytes": {"bloom": f.bloom.memoria_bytes(), "count_min_ventana": f.cms.memoria_bytes(),
                                          "hnsw": self.hnsw.memoria_bytes()},
            "almacen": {"productos": len(a.productos), "usuarios": len(a.usuarios), "sesiones": len(a.sesiones),
                        "ordenes": len(a.ordenes), "densidad_matriz": self.densidad,
                        "lru_tasa_aciertos": round(a.cache.tasa_aciertos, 3),
                        "productos_agotados": sum(1 for p in a.productos.values() if p["stock"] == 0),
                        "ventas_sin_stock": a.ventas_sin_stock, "cambios_de_precio": a.cambios_precio},
            "bplustree": {"claves": len(a.bpt), "altura": a.bpt.altura(), "lecturas_pagina": a.bpt.lecturas_pagina},
            "busquedas": {"total": sum(self.busquedas.c.values()), "top10": self.busquedas.top(10),
                          "terminos_en_trie": self.trie.n},
            "lsh": {"productos_indexados": len(self.lsh.firmas), "umbral_jaccard": round(self.lsh.umbral(), 3)},
            "hnsw": {"productos_indexados": self.hnsw.n, "niveles": self.hnsw.nivel_max + 1},
            "grafo": {"fuente": self.grafo_fuente, "nodos": len(self.nodos), "aristas": len(self.pesos),
                      "pagerank_iteraciones": self.pr_iter, "top10_pagerank": top_pr,
                      "top10_grado": [int(self.nodos[i]) for i in np.argsort(-grado)[:10]] if len(grado) else [],
                      "spearman_pagerank_vs_grado": round(gr.spearman(self.pr, grado), 3) if len(grado) > 2 else None,
                      "top10_comunes": len(set(top_pr) & {int(self.nodos[i]) for i in np.argsort(-grado)[:10]}) if len(grado) else 0,
                      "comunidades": {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "comunidades"}
                                      | {"estabilidad_nmi": round(self.estab[m], 3)} for m, r in getattr(self, "com", {}).items()}},
            "refrescos_micro_lote": self.refrescos,
            "top10_mas_vendidos": [(int(p), n) for p, n in a.ventas.top(10)],
            "tiempos_seg": {k: round(v, 2) for k, v in self.tiempos.items()},
            "traza_evento": self.traza,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--blacklist", required=True)
    ap.add_argument("--stock", default=None, help="data/catalogo_stock.csv")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--refresco", type=int, default=None, help="eventos por micro-lote (por defecto n/4, máx. 1M)")
    ap.add_argument("--out", default="outputs/resultados.json")
    a = ap.parse_args()

    df = cargar_stream(a.stream, a.n)
    bl = pd.read_csv(a.blacklist, dtype=str)
    entidades = (bl["entity_type"] + ":" + bl["entity"]).tolist()
    stock = {}
    if a.stock and os.path.exists(a.stock):
        s = pd.read_csv(a.stock)
        stock = dict(zip(s["product_id"].astype(int), s["stock"].astype(int)))
    refresco = a.refresco or max(10_000, min(1_000_000, len(df) // 4))
    p = Pipeline(entidades, stock, {"refresco_eventos": refresco})
    print(f"Procesando {len(df):,} eventos (refresco de índices cada {refresco:,})...")
    p.procesar_stream(df)

    extras = {}
    if p._seguido is not None:
        pid = p._seguido[1]
        rec = p.recomendar(pid)
        p._t("5/7/8 Micro-lote", "KD-Tree, MinHash+LSH, HNSW y grafo se actualizan con el producto en cada refresco")
        p._t("9 Servicio", f"recomendaciones Top-5 para el producto {pid}: "
                           + json.dumps({k: v for k, v in rec.items() if k != 'producto'}))
        extras["recomendacion_ejemplo"] = rec
    top_busq = p.busquedas.top(1)
    prefijo = top_busq[0][0][:2] if top_busq else "sa"
    extras["autocompletar_ejemplo"] = {prefijo: p.autocompletar(prefijo)}
    if p.almacen.productos:
        cat = next(iter(p.almacen.productos.values()))["category_id"]
        extras["rango_ejemplo"] = {"category_id": cat, "precio": [50, 200],
                                   "productos_disponibles": p.productos_por_rango(cat, 50, 200)[:10]}
    res = p.resumen() | extras
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: res[k] for k in ("eventos_ingestados", "preprocesamiento", "tiempos_seg")}, ensure_ascii=False, indent=2))
    print("Alertas por motivo:", res["alertas"]["por_motivo"], "| precisión fraude:", res["alertas"]["precision_fraude_vs_inyectado"])
    print("Refrescos por micro-lote:", [(r["eventos"], r["productos"], r["aristas_grafo"]) for r in p.refrescos])
    print("\nTraza de un evento de punta a punta:")
    for paso in p.traza:
        print(f"  [{paso['etapa']}] {paso['detalle']}")
    print(f"\nResultados completos en {a.out}")


if __name__ == "__main__":
    main()
