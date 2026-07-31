# -*- coding: utf-8 -*-
# ^^^ Declara la codificación UTF-8 para caracteres como tildes y la ñ.

"""
API v1 — Implementación Mínima
═══════════════════════════════════════════════════════════════════════
FastAPI con un solo endpoint de predicción.
Sin validación ni modelos Pydantic — ideal para entender los fundamentos
antes de añadir capas de complejidad.

¿Qué hace esta API?
  1. Carga un modelo Random Forest desde un archivo .pkl
  2. Expone un endpoint GET /predict que recibe 9 parámetros por URL
  3. Devuelve un JSON con el precio estimado en USD

Cómo ejecutar:
  python api_v1_minima.py
  # o bien:
  uvicorn api_v1_minima:app --reload

Probar en el navegador:
  http://localhost:8000/docs   ← documentación interactiva (Swagger)
  http://localhost:8000/predict?bedrooms=3&bathrooms=2&parking_spots=2&area_m2=200&lat=-0.18&lon=-78.48&city_quito=1
"""

# ── 1. IMPORTACIÓN DE LIBRERÍAS ─────────────────────────────────────────
# Cada import trae funcionalidad específica de una librería externa:

import joblib
# joblib: librería para guardar/cargar objetos Python en archivos binarios.
#         Aquí la usamos para leer el modelo entrenado (modelo_inmobiliario.pkl).
#         Alternativas: pickle (módulo estándar) o cloudpickle.

import pandas as pd
# pandas: librería de manipulación de datos tabulares.
#         Creamos un DataFrame de 1 fila con las 9 características
#         en el orden exacto que espera el modelo Random Forest.

from fastapi import FastAPI, Query
# FastAPI: la clase principal para crear la aplicación web.
#          Crea rutas (endpoints), valida parámetros, genera /docs automáticamente.
# Query:   declara que un parámetro se recibe desde la URL (query string).
#          Ejemplo: /predict?bedrooms=3 → Query(...) captura "bedrooms=3".

from fastapi.responses import JSONResponse
# JSONResponse: clase para devolver respuestas JSON explícitamente.
#               En v1 no se usa directamente, pero está importada por si acaso.
#               En v3 sí la usaremos para respuestas de error personalizadas.


# ── 2. CARGA DEL MODELO ─────────────────────────────────────────────────
# El modelo se carga UNA SOLA VEZ al iniciar el servidor (cuando Python
# ejecuta este archivo). Esto es eficiente porque no se recarga en cada
# petición, pero tiene la desventaja de que ocupa RAM constantemente.

model = joblib.load("modelo_inmobiliario.pkl")
# joblib.load() lee el archivo binario .pkl y reconstruye el objeto Python
# original (en este caso, un RandomForestRegressor de scikit-learn).
# Después de esta línea, "model" tiene un método .predict() que podemos llamar.

FEATURES = [
    "BEDROOMS", "BATHROOMS", "PARKING_SPOTS", "CONSTRUCTION_AREA_SQM",
    "LATITUDE", "LONGITUDE",
    "CITY_Guayaquil", "CITY_Manta", "CITY_Quito",
]
# Lista con los nombres de las 9 columnas que el modelo espera como entrada.
# El ORDEN ES CRÍTICO: el modelo se entrenó con las columnas en este orden
# exacto. Si cambiamos el orden, las predicciones serán incorrectas porque
# el modelo usará el valor de "BEDROOMS" donde esperaba "BATHROOMS", etc.
#
# Las 3 últimas columnas (CITY_Guayaquil, CITY_Manta, CITY_Quito) son
# el resultado de aplicar One-Hot Encoding a la columna original "CITY".
# Son variables binarias: 0 = "no es esta ciudad", 1 = "sí es esta ciudad".


# ── 3. CREACIÓN DE LA APLICACIÓN FASTAPI ────────────────────────────────

app = FastAPI(title="API Inmobiliaria v1 (Mínima)")
# FastAPI() crea una instancia de la aplicación web.
#   title: nombre que aparece en la documentación /docs (Swagger UI).
#   La variable "app" es la que uvicorn usará para servir la aplicación.


# ── 4. ENDPOINTS (RUTAS) ────────────────────────────────────────────────
# Un "endpoint" es una URL que el servidor expone y que el cliente puede
# consultar. Cada endpoint se define con un decorador (@app.get, @app.post)
# y una función que se ejecuta cuando se visita esa URL.

@app.get("/")
# @app.get("/"): DECORADOR que registra la función "root" como manejador
#                de peticiones GET a la ruta raíz "/".
#                GET = método HTTP para leer datos (el navegador usa GET).
#                "/" = la página principal (ej: http://localhost:8000/).
def root():
    # Esta función se ejecuta cuando alguien visita http://localhost:8000/
    # No recibe parámetros.
    return {"mensaje": "API Inmobiliaria — v1 mínima", "estado": "activa"}
    # FastAPI convierte automáticamente el diccionario Python a JSON.
    # El navegador recibirá: {"mensaje": "...", "estado": "activa"}


@app.get("/health")
# /health: endpoint de "health check" (verificación de salud).
#          Herramientas de monitoreo (como Kubernetes, Docker, o balances
#          de carga) consultan esta ruta para saber si el servidor está vivo.
def health():
    return {"status": "ok"}
    # Respuesta mínima: si devuelve 200 OK, el servidor funciona.


@app.get("/predict")
# /predict: endpoint principal de predicción.
#           Al ser GET, los parámetros se envían en la URL (query string).
def predict(
    # ── PARÁMETROS DE ENTRADA ────────────────────────────────────────────
    # Cada línea declara un parámetro que el usuario DEBE enviar en la URL.

    bedrooms: int = Query(..., description="Número de habitaciones"),
    # bedrooms: nombre del parámetro (aparece como ?bedrooms=3 en la URL).
    # int:      tipo de dato esperado. FastAPI convierte el string de la URL
    #           a entero. Si no se puede convertir, devuelve error 422.
    # Query():  indica que este parámetro viene de la query string.
    #   ...     (Ellipsis = 3 puntos) significa OBLIGATORIO.
    #           Si el usuario no lo envía, FastAPI devuelve error 422.
    #   description: texto que aparece en /docs (Swagger UI).

    bathrooms: int = Query(..., description="Número de baños"),
    # Misma estructura: obligatorio, entero, documentado.

    parking_spots: int = Query(..., description="Plazas de estacionamiento"),

    construction_area_sqm: float = Query(..., alias="area_m2", description="Área de construcción en m²"),
    # alias="area_m2": el usuario escribe ?area_m2=200 en la URL,
    #                  pero en el código Python usamos construction_area_sqm.
    #                  Esto permite URLs más cortas y amigables.
    # float: el área puede tener decimales (ej: 200.5 m²).

    latitude: float = Query(..., alias="lat", description="Latitud"),
    # alias="lat": el usuario escribe ?lat=-0.18.

    longitude: float = Query(..., alias="lon", description="Longitud"),
    # alias="lon": el usuario escribe ?lon=-78.48.

    city_guayaquil: int = Query(0, alias="city_guayaquil", description="1 si es Guayaquil, 0 si no"),
    # Query(0): el 0 es el VALOR POR DEFECTO. Si el usuario no envía
    #           ?city_guayaquil=..., se asume 0 (no es Guayaquil).
    #           A diferencia de Query(...) que es obligatorio.
    city_manta: int = Query(0, alias="city_manta", description="1 si es Manta, 0 si no"),
    city_quito: int = Query(0, alias="city_quito", description="1 si es Quito, 0 si no"),
):
    """Predice el precio de una propiedad en USD."""
    # ── 5. CONSTRUCCIÓN DEL DATAFRAME DE ENTRADA ────────────────────────
    # El modelo espera un DataFrame de pandas con las columnas en un orden
    # específico. Creamos un DataFrame con exactamente UNA fila (los datos
    # de la propiedad que queremos valuar).

    data = pd.DataFrame([[
        # Los DOBLES corchetes [[...]] crean una lista DENTRO de otra lista.
        # La lista interior = 1 fila de datos.
        # La lista exterior = el conjunto de filas (solo 1 en este caso).
        bedrooms, bathrooms, parking_spots, construction_area_sqm,
        latitude, longitude,
        city_guayaquil, city_manta, city_quito,
    ]], columns=FEATURES)
    # columns=FEATURES asigna los nombres de columna en el orden de la lista.
    # Resultado: un DataFrame de 1 fila × 9 columnas.

    # ── 6. PREDICCIÓN ────────────────────────────────────────────────────
    precio = float(model.predict(data)[0])
    # model.predict(data):
    #   - Recibe el DataFrame y devuelve un array de NumPy con las predicciones.
    #   - Como solo hay 1 fila, el array tiene 1 elemento: array([287452.63]).
    # [0]:
    #   - Extrae el primer (y único) elemento del array.
    # float(...):
    #   - Convierte numpy.float64 → float de Python nativo.
    #   - Necesario porque los numpy floats no son serializables a JSON
    #     en todas las versiones.

    return {"precio_usd": round(precio, 2)}
    # round(precio, 2): redondea a 2 decimales (centavos de dólar).
    # El diccionario se convierte a JSON automáticamente:
    # {"precio_usd": 287452.63}


# ── 7. PUNTO DE ENTRADA ─────────────────────────────────────────────────
# Este bloque solo se ejecuta si el archivo se corre directamente con
# "python api_v1_minima.py". NO se ejecuta si el archivo es importado
# desde otro script.

if __name__ == "__main__":
    # __name__ es una variable especial de Python.
    # Vale "__main__" cuando el archivo es el punto de entrada del programa.
    # Vale el nombre del módulo cuando es importado (ej: "api_v1_minima").

    import uvicorn
    # uvicorn: servidor ASGI (Asynchronous Server Gateway Interface).
    #          Es el servidor web que ejecuta FastAPI en producción.
    #          Similar a como Apache/Nginx sirven PHP, uvicorn sirve FastAPI.

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run():
    #   app:             la aplicación FastAPI que creamos arriba.
    #   host="0.0.0.0":  escucha en TODAS las interfaces de red.
    #                     "127.0.0.1" = solo localhost.
    #                     "0.0.0.0" = accesible desde otras máquinas.
    #   port=8000:       puerto TCP donde escucha el servidor.
    #                     8000 es el puerto por defecto de FastAPI/uvicorn.
