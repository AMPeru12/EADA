"""Demo en vivo: simula la llegada de registros uno por uno y muestra qué decide cada etapa.

Uso (Colab o local, desde la carpeta del proyecto):
    from src.demo import DemoEnVivo
    d = DemoEnVivo('data/stream_10M.parquet', 'data/blacklist.csv', 'data/catalogo_stock.csv')
    d.compra_normal()        # compra express real que atraviesa todo el pipeline
    d.duplicado()            # el mismo registro llega dos veces
    d.invalido()             # registro con precio negativo
    d.prioridades()          # el heap ordena eventos mezclados por urgencia
    d.lista_negra()          # usuario señalado: Bloom Filter + verificación exacta
    d.rafaga_bot()           # 20 carritos en 2 minutos: Count-Min Sketch dispara la alerta
    d.prueba_tarjetas()      # 3 pagos rechazados con la misma tarjeta
    d.servicio('sams')       # autocompletado, rango de precios y recomendaciones

Primero procesa un "calentamiento" (por defecto 200,000 eventos reales) para que existan
catálogo, índices y grafo; después los eventos de la demo llegan de a uno.
"""
import time
import pandas as pd

from .pipeline import Pipeline, cargar_stream, eventos
from .estructuras.heap_prioridad import nivel_evento

NOMBRE_NIVEL = {0: "ALERTA", 1: "express", 2: "estándar", 3: "programada", 4: "carrito/búsqueda", 5: "vista"}
TIPO = {"view": "vista", "cart": "carrito", "purchase": "compra", "search": "búsqueda"}


class DemoEnVivo:
    def __init__(self, stream, blacklist, stock=None, n_calentamiento=200_000, pausa=0.4):
        self.pausa = pausa
        print(f"Cargando {n_calentamiento:,} eventos reales para el calentamiento...")
        df = cargar_stream(stream, n_calentamiento + 50_000)
        self.base, self.resto = df.iloc[:n_calentamiento], df.iloc[n_calentamiento:]
        bl = pd.read_csv(blacklist, dtype=str)
        self.bl = bl
        stock_d = {}
        if stock:
            s = pd.read_csv(stock)
            stock_d = dict(zip(s["product_id"].astype(int), s["stock"].astype(int)))
        self.p = Pipeline((bl["entity_type"] + ":" + bl["entity"]).tolist(), stock_d,
                          {"refresco_eventos": max(10_000, n_calentamiento // 2)})
        t = time.perf_counter()
        self.p.procesar_stream(self.base)
        print(f"Listo en {time.perf_counter() - t:.0f} s: {len(self.p.almacen.productos):,} productos, "
              f"{len(self.p.alertas):,} alertas, índices y grafo construidos.\n")
        self.ts = int(self.base["ts"].max())
        self._usados = set()
        self.ultimo_cms = None
        orig = self.p.filtro._intensidad

        def espia(clave, ts, umbral):                      # guarda la estimación del Count-Min para mostrarla
            d, est = orig(clave, ts, umbral)
            self.ultimo_cms = (clave, est, umbral, d)
            return d, est
        self.p.filtro._intensidad = espia

    # ------------------------------------------------------------ utilidades
    def _paso(self, etapa, texto):
        print(f"  [{etapa:<22}] {texto}")
        time.sleep(self.pausa)

    def _plantilla(self, tipo="purchase", filtro=None, **cond):
        """Toma un evento real (no usado) del resto del stream como base."""
        r = self.resto
        m = (r["event_type"] == tipo) & (~r["synthetic"]) & (~r["is_fraud"]) & r["price"].gt(0)
        for k, v in cond.items():
            m &= r[k] == v
        for i in r.index[m]:
            if i not in self._usados:
                ev = next(eventos(r.loc[[i]]))
                if filtro is not None and not filtro(ev):
                    continue
                self._usados.add(i)
                return ev
        raise ValueError("No hay más eventos de ese tipo en el resto del stream")

    def _nuevo_ts(self, seg=5):
        self.ts += seg
        return self.ts

    # ------------------------------------------------------------ un evento por todo el pipeline
    def procesar(self, ev, titulo=None, detalle=True):
        p = self.p
        if titulo:
            print(f"\n▶ {titulo}")
        tipo = TIPO.get(ev["event_type"], ev["event_type"])
        if detalle:
            self._paso("0 Ingesta · Cola FIFO", f"llega {tipo} del usuario {ev['user_id']} "
                       f"(producto {ev.get('product_id')}, precio {self._precio(ev)}) y entra a la fila")
        p.cola.encolar(ev)
        lote = p.cola.desencolar_lote(len(p.cola))

        inv, dup, tar = p.pre.invalidos, p.pre.duplicados, p.pre.tardios
        limpio = p.pre.procesar(lote)
        if not limpio:
            if p.pre.invalidos > inv:
                self._paso("1 Preprocesamiento", f"❌ DESCARTADO por inválido ({self._ultimo_motivo(ev)})")
            elif p.pre.duplicados > dup:
                self._paso("1 Preprocesamiento", "❌ DESCARTADO: es un duplicado del registro anterior")
            return "descartado"
        if detalle:
            extra = " · llegó tarde, se procesa igual (tolerancia)" if p.pre.tardios > tar else ""
            self._paso("1 Preprocesamiento", f"✅ válido, normalizado y sin duplicado{extra}")

        for e in limpio:
            p.heap.insertar(e)
        ev = p.heap.extraer()
        p.atendidos_por_nivel[ev["prioridad"]] += 1
        if detalle:
            self._paso("2 Prioridad · Heap", f"nivel {ev['prioridad']} ({NOMBRE_NIVEL[ev['prioridad']]})")

        # etapa 6: se muestra qué responde cada control
        if detalle and ev["event_type"] != "search":
            ents = p.filtro.entidades_evento(ev)
            sospecha = [e for e in ents if e in p.filtro.bloom]
            if sospecha:
                reales = [e for e in sospecha if e in p.filtro.exactas]
                txt = (f"Bloom dice 'quizás' para {sospecha} → lista exacta: "
                       + ("CONFIRMADO" if reales else "era falso positivo, se deja pasar"))
            else:
                txt = "Bloom: usuario, tarjeta y dispositivo NO están en la lista negra (seguro)"
            self._paso("6 Fraude · Bloom", txt)
        self.ultimo_cms = None
        aprobado, alerta = p.filtro.evaluar(ev)
        if detalle and self.ultimo_cms and alerta is None:
            clave, est, umbral, _ = self.ultimo_cms
            self._paso("6 Fraude · Count-Min", f"{ {'cart': 'carritos', 'purchase': 'compras'}[clave.split('|')[1]]} del usuario en 10 min ≈ {est} (umbral {umbral}) → normal")
        if alerta is not None:
            p.heap.insertar(alerta, nivel=0)
            a = p.heap.extraer()
            p.alertas.append(a)
            est = f" (Count-Min estimó {a['estimado']}, verificado con el conteo exacto)" if a["estimado"] else ""
            accion = ("alerta de pago para el equipo de riesgo (el pago ya fue rechazado)" if a["motivo"] == "pago_rechazado"
                      else "la operación se BLOQUEA")
            self._paso("6 Fraude", f"🚨 ALERTA '{a['motivo']}'{est} → vuelve al heap con prioridad 0; {accion}")
            return "alerta"
        if not aprobado:
            self._paso("6 Fraude", "⏸ retenido: ya hay una alerta de este usuario en la ventana")
            return "retenido"

        if ev["event_type"] == "search":
            p.trie.insertar(ev["query"], 1)
            p.busquedas.agregar(ev["query"])
            self._paso("9 Servicio · Trie", f"búsqueda '{ev['query']}' agregada al autocompletado")
            return "aprobado"

        pid = int(ev["product_id"])
        antes = p.almacen.productos.get(pid, {}).get("stock")
        p.almacen.registrar(ev)
        prod = p.almacen.productos[pid]
        if detalle:
            txt = f"ficha del producto {pid} en la tabla hash"
            if ev["event_type"] == "purchase":
                txt += (f" · stock {antes} → {prod['stock']} · orden registrada" if antes
                        else " · sin stock: queda como pedido pendiente")
            self._paso("3 Almacén · Hash/LRU", "✅ " + txt)
            cat, precio = prod["category_id"], prod["price"]
            vecinos = p.productos_por_rango(cat, precio * 0.9, precio * 1.1)
            self._paso("4 Índice · B+Tree", f"indexado por (categoría, precio {precio}); "
                       f"{len(vecinos)} productos de su categoría con ±10% de precio y stock")
        return "aprobado"

    @staticmethod
    def _precio(ev):
        v = ev.get('price')
        return round(float(v), 2) if isinstance(v, (int, float)) else v

    def _ultimo_motivo(self, ev):
        return self.p.pre._motivo_invalido(ev) or "inválido"

    # ------------------------------------------------------------ escenarios
    def compra_normal(self):
        p = self.p

        def conocido(e):                     # producto con stock y presente en los índices: muestra todo el recorrido
            pid = int(e["product_id"])
            return (p.almacen.productos.get(pid, {}).get("stock", 0) > 0 and pid in p.lsh.firmas
                    and pid in p.nodo_pos and pid in p.hnsw_pos)
        try:
            ev = self._plantilla("purchase", filtro=conocido, order_type="express", payment_status="aprobado")
        except ValueError:
            ev = self._plantilla("purchase", order_type="express", payment_status="aprobado")
        ev["ts"] = self._nuevo_ts()
        self._compra = ev
        r = self.procesar(ev, "Escenario 1 · Una compra express real")
        if r == "aprobado":
            self._indices(int(ev["product_id"]))
            self._paso("9 Servicio", "recomendaciones para este producto (salida observable):")
            self._recomendaciones(int(ev["product_id"]))
            print("  → El evento atravesó las 10 etapas: 0 → 1 → 2 → 6 → 3 → 4 → 5/7/8 (micro-lote) → 9")
        return r

    def _indices(self, pid):
        """Etapas 5, 7 y 8: dónde está el producto en los índices del último micro-lote."""
        p = self.p
        en = lambda d: "sí" if pid in d else "aún no (entra en el próximo micro-lote)"
        self._paso("5 KD-Tree", f"producto en el índice de atributos: {en(p.kd_pos)} ({len(p.kd_pos):,} productos)")
        self._paso("7 MinHash+LSH / HNSW", f"firma de 256 valores en LSH: {en(p.lsh.firmas)} · embedding en HNSW: {en(p.hnsw_pos)}")
        if pid in p.nodo_pos:
            i = p.nodo_pos[pid]
            grado = int(p.A[i].nnz)
            puesto = int((p.pr > p.pr[i]).sum()) + 1
            self._paso("8 Grafo · PageRank", f"nodo del grafo con {grado} productos co-comprados · puesto {puesto:,} de {len(p.nodos):,} en PageRank")
        else:
            self._paso("8 Grafo · PageRank", "aún no tiene co-compras en el grafo")

    def duplicado(self):
        ev = self._plantilla("cart")
        ev["ts"] = self._nuevo_ts()
        print("\n▶ Escenario 2 · El mismo registro llega dos veces (por ejemplo, un doble clic)")
        self.procesar(dict(ev), "Primera vez:")
        self.procesar(dict(ev), "Segunda vez (idéntico):")

    def invalido(self):
        ev = self._plantilla("view")
        ev["ts"] = self._nuevo_ts()
        ev["price"] = -5.0
        self.procesar(ev, "Escenario 3 · Registro con precio negativo")

    def prioridades(self):
        print("\n▶ Escenario 4 · Llegan 6 eventos mezclados al mismo tiempo; el heap decide el orden")
        mezcla = [self._plantilla("view"), self._plantilla("purchase", order_type="programada"),
                  self._plantilla("cart"), self._plantilla("purchase", order_type="express"),
                  self._plantilla("purchase", order_type="estandar"), self._plantilla("view")]
        ts = self._nuevo_ts()
        print("  Orden de llegada:  ", " → ".join(self._etiqueta(e) for e in mezcla))
        for e in mezcla:
            e["ts"] = ts
            self.p.heap.insertar(e)
        orden = [self.p.heap.extraer() for _ in mezcla]
        time.sleep(self.pausa)
        print("  Orden de atención: ", " → ".join(self._etiqueta(e) for e in orden))
        print("  (el heap siempre entrega primero el más urgente, sin reordenar toda la fila)")

    @staticmethod
    def _etiqueta(e):
        return NOMBRE_NIVEL[nivel_evento(e)] if e["event_type"] == "purchase" else TIPO[e["event_type"]]

    def lista_negra(self):
        uid = int(self.bl.loc[self.bl["entity_type"] == "user", "entity"].iloc[0])
        ev = self._plantilla("purchase", payment_status="aprobado")
        ev["user_id"], ev["ts"] = uid, self._nuevo_ts()
        self.procesar(ev, f"Escenario 5 · Compra de un usuario que está en la lista negra ({uid})")

    def rafaga_bot(self, n=20, cada_seg=6):
        base = self._plantilla("cart")
        uid = 999_000_001
        print(f"\n▶ Escenario 6 · Un bot agrega {n} productos al carrito en {n * cada_seg // 60} minutos "
              f"(usuario nuevo {uid}, no está en la lista negra)")
        for i in range(n):
            ev = dict(base, user_id=uid, user_session=f"bot-{uid}", ts=self._nuevo_ts(cada_seg),
                      product_id=int(base["product_id"]) + i, payment_token=uid * 10, device_id=uid * 10)
            r = self.procesar(ev, detalle=False)
            est = self.ultimo_cms[1] if self.ultimo_cms else "-"
            print(f"  carrito {i + 1:>2}: Count-Min ≈ {est:>2} → {r}")
            time.sleep(self.pausa / 3)
            if r == "alerta":
                print("  🚨 Se superó el umbral de 18 carritos en 10 minutos; se verificó con el conteo exacto y se alertó.")
                break

    def prueba_tarjetas(self):
        base = self._plantilla("purchase")
        print("\n▶ Escenario 7 · Tres pagos rechazados con la misma tarjeta en pocos minutos")
        for i in range(3):
            ev = dict(base, payment_status="rechazado", ts=self._nuevo_ts(40),
                      user_session=f"{base['user_session']}-{i}")
            self.procesar(ev, f"Intento {i + 1}:")

    def servicio(self, prefijo="sa"):
        p = self.p
        print("\n▶ Escenario 8 · Lo que ve el cliente y el gerente (etapa 9)")
        self._paso("Trie", f"autocompletar '{prefijo}': {p.autocompletar(prefijo)}")
        self._paso("Top-K", f"10 más vendidos: {[pid for pid, _ in p.almacen.ventas.top(10)]}")
        prod = next(iter(p.almacen.productos.values()))
        cat = prod["category_id"]
        rango = p.productos_por_rango(cat, 50, 500)
        self._paso("B+Tree", f"categoría {cat}, precio 50–500, con stock: {len(rango)} productos, por ejemplo {rango[:5]}")
        self._paso("Alertas", f"total {len(p.alertas):,}")

    def _recomendaciones(self, pid):
        rec = self.p.recomendar(pid)
        nombres = {"similares_comportamiento_lsh": "Similares (MinHash+LSH)",
                   "complementarios_hnsw": "Complementarios (HNSW)",
                   "co_compra_ppr": "Se compra junto con (PageRank)",
                   "atributos_knn_exacto": "Parecidos por atributos (KD-Tree)"}
        for k, n in nombres.items():
            v = rec.get(k) or "— (producto aún sin datos suficientes; se llena en el próximo micro-lote)"
            print(f"      · {n}: {v}")
            time.sleep(self.pausa / 2)

    # ------------------------------------------------------------ micro-lote (etapas 5, 7 y 8)
    def micro_lote(self, n=20_000):
        """Llegan n eventos reales nuevos y se refrescan KD-Tree, LSH, HNSW y grafo; se ve cómo cambia la recomendación."""
        p = self.p
        pid = int(self._compra["product_id"]) if hasattr(self, "_compra") else None
        antes = p.recomendar(pid) if pid else None
        print(f"\n▶ Escenario 9 · Llegan {n:,} eventos reales más y se ejecuta un micro-lote (etapas 5, 7 y 8)")
        bloque = self.resto.iloc[-n:]
        self._usados.update(bloque.index)
        t = time.perf_counter()
        for ev in eventos(bloque):
            p.ingestados += 1
            p.cola.encolar(ev)
            if len(p.cola) >= p.cfg["tam_lote"]:
                p._procesar_lote(p.cola.desencolar_lote(len(p.cola)))
        if len(p.cola):
            p._procesar_lote(p.cola.desencolar_lote(len(p.cola)))
        self._paso("0–4, 6", f"{n:,} eventos procesados por evento en {time.perf_counter() - t:.1f} s")
        p.refrescar(final=False)
        f = p.refrescos[-1]
        self._paso("5/7/8 micro-lote", f"en {f['segundos']} s: {f['productos']:,} productos · {f['lsh_indexados']:,} firmas LSH "
                   f"({f['lsh_actualizados']:,} actualizadas) · {f['hnsw']:,} vectores HNSW · {f['aristas_grafo']:,} aristas")
        if pid:
            despues = p.recomendar(pid)
            for k, n_ in (("co_compra_ppr", "Se compra junto con"), ("similares_comportamiento_lsh", "Similares")):
                cambio = "cambió" if antes.get(k) != despues.get(k) else "se mantuvo"
                print(f"      · {n_} del producto {pid}: {antes.get(k)} → {despues.get(k)} ({cambio})")

    # ------------------------------------------------------------ similitud frente al método exacto
    def similitud_vs_exacto(self, pid=None, consultas=100):
        from .estructuras.minhash_lsh import jaccard
        from .estructuras.hnsw import knn_coseno_exacto, recall_at_k
        import numpy as np
        p, a = self.p, self.p.almacen
        def n_similares(q):                                 # cuántos productos tienen Jaccard ≥ 0.125 con q
            b_ = a.sesiones_producto[q]
            return sum(jaccard(b_, a.sesiones_producto[r]) >= 0.125 for r in p.lsh.firmas if r != q)
        if pid is None:                                     # el producto comprado o, si no tiene similares, uno de los más vistos
            cands = sorted(p.lsh.firmas, key=lambda x: -len(a.sesiones_producto[x]))[:10]
            if hasattr(self, "_compra") and int(self._compra["product_id"]) in p.lsh.firmas:
                cands = [int(self._compra["product_id"])] + cands
            pid = max(cands, key=n_similares)
        print(f"\n▶ Escenario 10 · Similitud aproximada frente al método exacto (producto {pid})")
        # MinHash+LSH frente a Jaccard por fuerza bruta
        t = time.perf_counter(); lsh = p.lsh.similares(pid, 10); t_lsh = time.perf_counter() - t
        base = a.sesiones_producto[pid]
        t = time.perf_counter()
        exacto = sorted(((jaccard(base, a.sesiones_producto[q]), q) for q in p.lsh.firmas if q != pid), reverse=True)
        t_ex = time.perf_counter() - t
        reales = {q for j, q in exacto if j >= 0.125}
        encontrados = reales & {q for _, q in lsh}
        self._paso("MinHash+LSH", f"{len(lsh)} candidatos en {t_lsh * 1000:.1f} ms: {[q for _, q in lsh[:5]]}")
        self._paso("Jaccard exacto", f"compara contra {len(p.lsh.firmas):,} productos en {t_ex * 1000:.1f} ms; los 5 más similares: "
                   + ", ".join(f"{q} (J={j:.2f})" for j, q in exacto[:5]))
        if reales:
            txt = f"LSH encontró {len(encontrados)} de los {len(reales)} productos con Jaccard ≥ 0.125 (recall {len(encontrados) / len(reales):.2f})"
        else:
            txt = "ningún producto llega a Jaccard 0.125 con este, así que no hay pares que LSH deba encontrar"
        self._paso("Resultado", f"{txt} · LSH respondió {t_ex / max(t_lsh, 1e-9):.0f} veces más rápido")
        # HNSW frente a KNN exacto
        rng = np.random.default_rng(42)
        idx = rng.choice(len(p.emb), min(consultas, len(p.emb)), replace=False)
        t = time.perf_counter(); aprox = [p.hnsw.buscar(p.emb[i], 10, ef=50) for i in idx]; t_h = time.perf_counter() - t
        t = time.perf_counter(); exac = [knn_coseno_exacto(p.emb, p.emb[i], 10) for i in idx]; t_k = time.perf_counter() - t
        rec = float(np.mean([recall_at_k(x, y) for x, y in zip(aprox, exac)]))
        self._paso("HNSW vs KNN exacto", f"{len(idx)} consultas de 10 vecinos sobre {len(p.emb):,} productos: "
                   f"HNSW {t_h * 1000:.0f} ms · exacto {t_k * 1000:.0f} ms · recall@10 = {rec:.3f}")
        print("      (con pocos miles de productos la fuerza bruta de NumPy puede ganar; la ventaja de HNSW aparece a gran escala:\n"
              "       en el benchmark fue 16 veces más rápido que el KNN exacto con 100,000 vectores)")

    # ------------------------------------------------------------ análisis del grafo
    def grafo(self):
        import numpy as np
        from .estructuras import grafo as gr
        p = self.p
        print(f"\n▶ Escenario 11 · Análisis del grafo de co-compra ({len(p.nodos):,} productos, {len(p.pesos):,} aristas; fuente: {p.grafo_fuente})")
        grado = gr.ranking_grado(p.A)
        top_pr = [int(p.nodos[i]) for i in np.argsort(-p.pr)[:5]]
        top_gr = [int(p.nodos[i]) for i in np.argsort(-grado)[:5]]
        self._paso("PageRank (influencia)", f"Top-5: {top_pr}")
        self._paso("Grado (conexiones)", f"Top-5: {top_gr} · Spearman con PageRank = {gr.spearman(p.pr, grado):.3f}")
        G = gr.a_networkx(p.pesos)                          # comunidades sobre el grafo actual
        com = {m: gr.comunidades(G, m) for m in ("louvain", "label_propagation")}
        if com:
            for m, nombre in (("louvain", "Louvain"), ("label_propagation", "Label Propagation")):
                c = com[m]
                est = gr.estabilidad(G, m, range(5))
                self._paso(nombre, f"modularidad {c['modularidad']:.3f} · {c['n_comunidades']:,} comunidades · "
                           f"tamaño máx. {c['tam_max']} · {c['segundos']:.2f} s · estabilidad (NMI, 5 semillas) {est:.3f}")
            grande = max(com["louvain"]["comunidades"], key=len)
            self._paso("Comunidad más grande", f"{len(grande)} productos que se compran juntos, por ejemplo {sorted(grande)[:8]}")
        pid = top_pr[0]
        v = np.zeros(len(p.nodos)); v[p.nodo_pos[pid]] = 1
        ppr, _ = gr.pagerank(p.A, personalizacion=v)
        self._paso("Personalized PageRank", f"cross-selling para el producto más influyente {pid}: "
                   f"{[int(p.nodos[i]) for i in np.argsort(-ppr) if p.nodos[i] != pid][:5]}")

    def todo(self, prefijo="sa"):
        self.compra_normal(); self.duplicado(); self.invalido(); self.prioridades()
        self.lista_negra(); self.rafaga_bot(); self.prueba_tarjetas(); self.servicio(prefijo)
        self.micro_lote(); self.similitud_vs_exacto(); self.grafo()
