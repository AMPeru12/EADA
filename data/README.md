# Datos

Los archivos de datos no se suben al repositorio por su tamaño. Para generarlos:

1. Descargar un CSV mensual de *eCommerce behavior data from multi category store* (REES46) desde Kaggle:
   https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store (por ejemplo `2019-Oct.csv`).
2. Ejecutar desde la raíz del repositorio:
   ```bash
   python src/preparar_datos.py --input ruta/2019-Oct.csv --n 10000000 --out data
   ```

Se generan en esta carpeta: `stream_10M.parquet`, `blacklist.csv` y `eda_resumen.json`.
`eda_resumen.json` sí puede subirse (es pequeño) para documentar el dataset usado.
