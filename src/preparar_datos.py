"""
Fase 1 del proyecto: construye el stream de trabajo a partir del CSV de REES46.

Base real (REES46): catálogo de productos e historial de vistas, carritos y compras con
timestamps reales. Componente sintético (lo que el dataset no trae), con semilla fija:

  S3  Tipo de orden de cada compra: express 10%, estándar 70%, programada 20%.
  S4  Medios de pago (1-3) y dispositivos (1-2) por usuario.
  S5  Fraude: 0.5% de usuarios; cada uno hace una ráfaga de 20 carritos y 6 compras en 2 minutos.
      Solo la mitad de los defraudadores está en la lista negra.
  S6  Estado del pago: 2% de rechazos en compras normales, 50% en compras de defraudadores
      y 60% en las ráfagas (patrón de prueba de tarjetas o "card testing").
  S7  Ruido: ~1% de duplicados exactos y ~0.5% de registros inválidos
      (precio negativo, tipo de evento mal escrito o sesión vacía).
  S2  Desorden: ~1% de eventos llega con 1 a 30 minutos de retraso (arrival_time > event_time).
  S8  Búsquedas: 15% de las sesiones empieza con una búsqueda (marca o tipo de producto
      del primer producto visto), 1 a 30 segundos antes de esa vista.
  S9  Stock inicial por producto (5% agotado) en data/catalogo_stock.csv.

Uso:
  pip install pandas pyarrow numpy
  python src/preparar_datos.py --input 2019-Oct.csv --n 10000000 --out data
"""
import argparse, json, os, time
import numpy as np
import pandas as pd

DTYPES = {
    "event_type": "string", "product_id": "int64", "category_id": "int64",
    "category_code": "string", "brand": "string", "price": "float32",
    "user_id": "int64", "user_session": "string",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV de REES46, p. ej. 2019-Oct.csv")
    ap.add_argument("--n", type=int, default=10_000_000, help="eventos reales a tomar")
    ap.add_argument("--out", default="data", help="carpeta de salida")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraud-rate", type=float, default=0.005)
    ap.add_argument("--dup-rate", type=float, default=0.01)
    ap.add_argument("--invalid-rate", type=float, default=0.005)
    ap.add_argument("--late-rate", type=float, default=0.01)
    ap.add_argument("--search-rate", type=float, default=0.15, help="fracción de sesiones con búsqueda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    # ---------------------------------------------------------------- base real
    print(f"Leyendo {a.n:,} eventos de {a.input} ...")
    df = pd.read_csv(a.input, nrows=a.n, dtype=DTYPES)
    df["event_time"] = pd.to_datetime(df["event_time"].str.replace(" UTC", "", regex=False))
    df["synthetic"] = False
    df["invalido"] = False
    df["query"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # ---------------------------------------------------------------- S4 pago y dispositivo
    users = df["user_id"].unique()
    tok_map = pd.Series(rng.integers(1, 4, size=len(users)), index=users)
    dev_map = pd.Series(rng.integers(1, 3, size=len(users)), index=users)
    # Enteros (user_id*10 + n.º de medio/dispositivo) para ahorrar memoria con 10M filas
    df["payment_token"] = df["user_id"] * 10 + (rng.random(len(df)) * df["user_id"].map(tok_map).values).astype("int64")
    df["device_id"] = df["user_id"] * 10 + (rng.random(len(df)) * df["user_id"].map(dev_map).values).astype("int64")

    # ---------------------------------------------------------------- S3 tipo de orden
    is_buy = (df["event_type"] == "purchase").values
    order = np.full(len(df), None, dtype=object)
    order[is_buy] = rng.choice(["express", "estandar", "programada"], size=is_buy.sum(), p=[0.10, 0.70, 0.20])
    df["order_type"] = order

    # ---------------------------------------------------------------- S5 y S6 fraude y pagos
    fraud_users = rng.choice(users, size=max(1, int(len(users) * a.fraud_rate)), replace=False)
    df["is_fraud"] = df["user_id"].isin(fraud_users)
    p_rechazo = np.where(df["is_fraud"].values, 0.50, 0.02)
    estado = np.full(len(df), None, dtype=object)
    estado[is_buy] = np.where(rng.random(is_buy.sum()) < p_rechazo[is_buy], "rechazado", "aprobado")
    df["payment_status"] = estado

    t_min, t_max = df["event_time"].min(), df["event_time"].max()
    products = df[["product_id", "category_id", "category_code", "brand", "price"]].drop_duplicates("product_id")
    bursts = []
    for u in fraud_users:                                  # S5: cada defraudador hace una ráfaga
        start = t_min + (t_max - t_min) * rng.random()
        b = products.sample(26, replace=True, random_state=int(rng.integers(1 << 31))).copy()
        b["event_type"] = ["cart"] * 20 + ["purchase"] * 6
        b["event_time"] = start + pd.to_timedelta(np.sort(rng.random(26)) * 120, unit="s")
        b["user_id"] = u
        b["user_session"] = f"burst_{u}_{start.value}"
        b["payment_token"] = u * 10
        b["device_id"] = u * 10
        b["order_type"] = [None] * 20 + list(rng.choice(["express", "estandar"], size=6))
        b["payment_status"] = [None] * 20 + list(np.where(rng.random(6) < 0.6, "rechazado", "aprobado"))
        b["is_fraud"] = True
        b["synthetic"] = True
        b["invalido"] = False
        bursts.append(b)
    df = pd.concat([df] + bursts, ignore_index=True)

    # Lista negra: solo la mitad de los defraudadores es conocida
    conocidos = rng.choice(fraud_users, size=max(1, len(fraud_users) // 2), replace=False)
    fr = df[df["user_id"].isin(conocidos)]
    blacklist = pd.concat([
        pd.DataFrame({"entity_type": "user", "entity": fr["user_id"].astype(str).unique()}),
        pd.DataFrame({"entity_type": "payment_token", "entity": fr["payment_token"].astype(str).unique()}),
        pd.DataFrame({"entity_type": "device", "entity": fr["device_id"].astype(str).unique()}),
    ])
    blacklist.to_csv(os.path.join(a.out, "blacklist.csv"), index=False)

    # ---------------------------------------------------------------- S8 búsquedas
    vistas = df[(df["event_type"] == "view") & ~df["synthetic"]]
    primeras = vistas.sort_values("event_time").drop_duplicates("user_session")
    primeras = primeras.sample(frac=a.search_rate, random_state=a.seed)
    tipo = primeras["category_code"].str.split(".").str[-1]
    usar_marca = (rng.random(len(primeras)) < 0.6) & primeras["brand"].notna()
    consulta = np.where(usar_marca, primeras["brand"], tipo)
    busq = primeras.assign(query=pd.Series(consulta, index=primeras.index, dtype="string")).dropna(subset=["query"]).copy()
    busq["event_type"] = "search"
    busq["event_time"] = busq["event_time"] - pd.to_timedelta(rng.uniform(1, 30, len(busq)), unit="s")
    busq[["product_id", "category_id"]] = 0
    busq["price"] = np.nan
    busq[["category_code", "brand"]] = pd.NA
    busq["synthetic"] = True
    df = pd.concat([df, busq], ignore_index=True)

    # ---------------------------------------------------------------- S7 ruido
    dups = df.sample(frac=a.dup_rate, random_state=a.seed).assign(synthetic=True)
    inval = df[df["event_type"] != "search"].sample(frac=a.invalid_rate, random_state=a.seed + 1).copy()
    tipo_err = rng.integers(0, 3, len(inval))
    inval.loc[tipo_err == 0, "price"] = -inval.loc[tipo_err == 0, "price"].abs()
    inval.loc[tipo_err == 1, "event_type"] = rng.choice(["purchse", "vew", "crt"], int((tipo_err == 1).sum()))
    inval.loc[tipo_err == 2, "user_session"] = pd.NA
    inval["synthetic"] = True
    inval["invalido"] = True
    df = pd.concat([df, dups, inval], ignore_index=True)

    # ---------------------------------------------------------------- S2 llegada con retraso
    retraso = np.zeros(len(df))
    tarde = rng.random(len(df)) < a.late_rate
    retraso[tarde] = rng.uniform(60, 1800, tarde.sum())
    df["arrival_time"] = df["event_time"] + pd.to_timedelta(retraso, unit="s")
    df["llega_tarde"] = tarde

    # El stream se ordena por llegada (como lo vería el sistema), no por ocurrencia
    df = df.sort_values("arrival_time", kind="stable").reset_index(drop=True)
    df["event_type"] = df["event_type"].astype("category")
    out_path = os.path.join(a.out, f"stream_{a.n // 1_000_000}M.parquet" if a.n >= 1_000_000 else f"stream_{a.n}.parquet")
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:  # sin pyarrow se guarda en pickle (instala pyarrow para usar Parquet)
        out_path = out_path.replace(".parquet", ".pkl")
        df.to_pickle(out_path)

    # ---------------------------------------------------------------- S9 stock inicial
    cat = products[["product_id"]].copy()
    stock = rng.negative_binomial(2, 0.05, len(cat))
    stock[rng.random(len(cat)) < 0.05] = 0
    cat["stock"] = stock
    cat.to_csv(os.path.join(a.out, "catalogo_stock.csv"), index=False)

    # ---------------------------------------------------------------- resumen exploratorio
    real = df[~df["synthetic"]]
    buys = real[real["event_type"] == "purchase"]
    prods_per_session = buys.groupby("user_session")["product_id"].nunique()
    carts = real[real["event_type"] == "cart"]
    cart_10 = carts.groupby(["user_id", pd.Grouper(key="event_time", freq="10min")]).size()
    buy_10 = buys.groupby(["user_id", pd.Grouper(key="event_time", freq="10min")]).size()
    summary = {
        "eventos_totales": int(len(df)),
        "eventos_reales": int(len(real)),
        "eventos_sinteticos": {
            "rafagas_fraude": int(sum(len(b) for b in bursts)),
            "busquedas": int(len(busq)),
            "duplicados": int(len(dups)),
            "invalidos": int(len(inval)),
            "llegan_tarde": int(tarde.sum()),
        },
        "rango_temporal": [str(df["event_time"].min()), str(df["event_time"].max())],
        "distribucion_event_type_%": (real["event_type"].astype(str).value_counts(normalize=True) * 100).round(2).to_dict(),
        "usuarios_unicos": int(real["user_id"].nunique()),
        "productos_unicos": int(real["product_id"].nunique()),
        "sesiones_unicas": int(real["user_session"].nunique()),
        "categorias_unicas": int(real["category_id"].nunique()),
        "category_code_nulo_%": round(float(real["category_code"].isna().mean() * 100), 2),
        "brand_nulo_%": round(float(real["brand"].isna().mean() * 100), 2),
        "compras": int(len(buys)),
        "pagos_rechazados": int((df["payment_status"] == "rechazado").sum()),
        "ordenes (sesiones con compra)": int(prods_per_session.size),
        "ordenes con >=2 productos (aristas del grafo)": int((prods_per_session >= 2).sum()),
        "usuarios_fraudulentos": int(len(fraud_users)),
        "fraudulentos_en_lista_negra": int(len(conocidos)),
        "entidades_en_lista_negra": int(len(blacklist)),
        "productos_sin_stock_inicial": int((cat["stock"] == 0).sum()),
        "carritos_por_usuario_10min_p99.9": float(cart_10.quantile(0.999)) if len(cart_10) else 0,
        "compras_por_usuario_10min_p99.9": float(buy_10.quantile(0.999)) if len(buy_10) else 0,
        "semilla": a.seed,
        "segundos": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "eda_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nStream guardado en {out_path}")


if __name__ == "__main__":
    main()
