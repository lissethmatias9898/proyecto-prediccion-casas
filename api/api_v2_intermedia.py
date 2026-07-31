# -*- coding: utf-8 -*-
"""
API v2 — Intermedia
═══════════════════════════════════════════════════════════════════════
Añade a v1: Pydantic models, validación de datos, logging, múltiples
endpoints (incluyendo POST), documentación OpenAPI detallada y
respuestas tipadas con response_model.

¿Qué aporta sobre v1?
  1. Modelos Pydantic ─ validación automática de rangos y tipos.
  2. Logging ── registro de eventos con timestamp (mejor que print).
  3. Endpoint POST ─ los datos viajan en el body (más seguro/profesional).
  4. Validación de ciudad única ─ exactamente una ciudad = 1.
  5. Manejo de errores ── HTTPException con códigos HTTP correctos.
  6. /features ── endpoint que describe las variables del modelo.

Cómo ejecutar:
  python api_v2_intermedia.py
  uvicorn api_v2_intermedia:app --reload
"""

# ── 1. IMPORTACIONES ────────────────────────────────────────────────────

import logging
# logging: módulo estándar de Python para registrar eventos.
#          Mejor que print() porque incluye timestamps, niveles
#          (DEBUG, INFO, WARNING, ERROR, CRITICAL) y se puede
#          redirigir a archivos o sistemas externos.

import joblib
# joblib: carga el modelo .pkl (igual que en v1).

import pandas as pd
# pandas: construye el DataFrame de entrada para el modelo.

from pydantic import BaseModel, Field, field_validator
# pydantic: librería de validación de datos usada por FastAPI.
#   BaseModel:       clase base para definir modelos de datos.
#                    Similar a un dataclass pero con validación automática.
#   Field:           función para añadir metadatos/restricciones a cada campo.
#                    ge=1 significa "greater than or equal to 1".
#   field_validator: decorador para crear validadores personalizados
#                    que se ejecutan sobre campos individuales.

from fastapi import FastAPI, HTTPException
# HTTPException: excepción que FastAPI convierte en respuesta HTTP
#                con código de error (422, 500, etc.).

from fastapi.responses import JSONResponse
# JSONResponse: respuesta JSON explícita (útil para errores personalizados).


# ── 2. CONFIGURACIÓN DE LOGGING ─────────────────────────────────────────
# Configuramos el sistema de logging UNA vez al inicio del programa.

logging.basicConfig(
    level=logging.INFO,
    # level: nivel mínimo de mensajes a mostrar.
    #        Orden: DEBUG < INFO < WARNING < ERROR < CRITICAL.
    #        Con INFO, los DEBUG no se muestran. En producción se usa WARNING.

    format="%(asctime)s | %(levelname)s | %(message)s",
    # format: plantilla para cada línea de log.
    #   %(asctime)s:   timestamp automático (ej: 2025-07-05 13:01:42,123).
    #   %(levelname)s: nivel del mensaje (INFO, ERROR, etc.).
    #   %(message)s:   el texto que pasamos a logger.info(...).
)
logger = logging.getLogger(__name__)
# getLogger(__name__): crea un logger con el nombre del módulo actual.
#   __name__ = "api_v2_intermedia" cuando se ejecuta directamente.
#   Esto permite identificar qué archivo generó cada mensaje de log.
#   Ahora usamos logger.info(...) en vez de print(...).


# ── 3. CARGA DEL MODELO (CON MANEJO DE ERRORES) ────────────────────────
# A diferencia de v1, aquí envolvemos la carga en try/except.

try:
    # Intenta cargar el modelo. Si el archivo no existe o está corrupto,
    # saltará al bloque except en vez de crashear con un traceback feo.
    model = joblib.load("modelo_inmobiliario.pkl")
    logger.info("Modelo cargado correctamente")
    # logger.info(): registra un evento informativo (nivel INFO).
except Exception as e:
    # Captura CUALQUIER excepción (archivo no encontrado, versión
    # incompatible de scikit-learn, memoria insuficiente, etc.).
    logger.critical("No se pudo cargar el modelo: %s", e)
    # logger.critical(): nivel CRITICAL, el más alto. Indica que la
    #                    aplicación no puede funcionar sin el modelo.
    raise
    # raise (sin argumentos) re-lanza la excepción original.
    # Esto DETIENE el programa. Sin el modelo, la API no puede funcionar.

FEATURES = [
    "BEDROOMS", "BATHROOMS", "PARKING_SPOTS", "CONSTRUCTION_AREA_SQM",
    "LATITUDE", "LONGITUDE",
    "CITY_Guayaquil", "CITY_Manta", "CITY_Quito",
]
# Misma lista de 9 features que en v1. El orden debe coincidir con
# el orden en que se entrenó el modelo.


# ── 4. CREACIÓN DE LA APP (CON METADATOS ENRIQUECIDOS) ─────────────────

app = FastAPI(
    title="API Inmobiliaria v2 (Intermedia)",
    # title: nombre que aparece en Swagger UI (/docs).

    description=(
        "Predicción de precios inmobiliarios usando Random Forest. "
        "El modelo fue entrenado con datos de Ecuador (Quito, Guayaquil, Manta). "
        "Usar una y solo una ciudad = 1; el resto = 0."
    ),
    # description: texto explicativo en la cabecera de /docs.
    #              Soporta Markdown (negritas, listas, etc.).
    #              Aquí explicamos al consumidor de la API cómo usar las ciudades.

    version="2.0.0",
    # version: útil para saber qué versión de la API está corriendo
    #          (importante en producción cuando hay múltiples despliegues).

    contact={"name": "Equipo EELA - Ciencia de Datos"},
    # contact: información de contacto que aparece en /docs.
)


# ── 5. MODELOS PYDANTIC ─────────────────────────────────────────────────
# Pydantic es el "pegamento" entre la petición HTTP y el código Python.
# Convierte JSON → objeto Python CON validación automática.
#
# Ventajas sobre usar parámetros Query como en v1:
#   - Validación de rangos (ge=1, le=20)
#   - Documentación automática en /docs
#   - Reutilización: el mismo modelo sirve para POST y GET
#   - Errores claros: "bedrooms must be >= 1" en vez de un traceback

class PropiedadInput(BaseModel):
    # BaseModel: heredar de esta clase activa toda la magia de Pydantic.
    #            Cada atributo de clase = un campo del JSON de entrada.

    bedrooms: int = Field(..., ge=1, le=20, description="Número de habitaciones")
    # bedrooms:      nombre del campo en el JSON.
    # int:           tipo de dato. Si llega "tres", Pydantic devuelve error.
    # Field(...):    función que configura restricciones adicionales.
    #   ...:         Ellipsis = campo OBLIGATORIO (no tiene valor por defecto).
    #   ge=1:        "greater than or equal to 1" → bedrooms >= 1.
    #                Si el usuario envía 0, Pydantic rechaza con error claro.
    #   le=20:       "less than or equal to 20" → bedrooms <= 20.
    #   description: texto que aparece en la documentación /docs.

    bathrooms: int = Field(..., ge=1, le=20, description="Número de baños")
    parking_spots: int = Field(..., ge=0, le=20, description="Plazas de estacionamiento")
    # parking_spots permite 0 (ge=0) porque hay propiedades sin estacionamiento.

    construction_area_sqm: float = Field(..., alias="area_m2", ge=10, le=10000, description="Área de construcción en m²")
    # alias="area_m2": en el JSON el campo se llama "area_m2", pero en Python
    #                  lo accedemos como .construction_area_sqm.
    #                  Esto permite un JSON más corto sin sacrificar claridad en código.
    # ge=10:   mínimo 10 m² (una propiedad más pequeña no es realista).
    # le=10000: máximo 10,000 m² (el capping del dataset fue al percentil 99).

    latitude: float = Field(..., alias="lat", ge=-90, le=90, description="Latitud")
    # ge=-90, le=90: rango válido de latitud en el planeta Tierra.
    #                 Esto detecta errores de tipeo (ej: 91 en vez de 19).

    longitude: float = Field(..., alias="lon", ge=-180, le=180, description="Longitud")
    # ge=-180, le=180: rango válido de longitud.

    city_guayaquil: int = Field(0, ge=0, le=1, description="Guayaquil (0/1)")
    # Field(0): el 0 es el VALOR POR DEFECTO. Si el usuario no envía
    #           este campo en el JSON, se asume 0 (no es esa ciudad).
    #           Esto es diferente de Field(...) que es obligatorio.
    city_manta: int = Field(0, ge=0, le=1, description="Manta (0/1)")
    city_quito: int = Field(0, ge=0, le=1, description="Quito (0/1)")

    @field_validator("city_guayaquil", "city_manta", "city_quito")
    @classmethod
    def check_ciudad_unica(cls, v, info):
        # field_validator: decorador que registra esta función como validador
        #                  de los campos listados en el primer argumento.
        # "city_guayaquil", "city_manta", "city_quito": nombres de los campos
        #                  a los que se aplica este validador.
        # cls:             la clase (PropiedadInput) — es un classmethod.
        # v:               el valor del campo DESPUÉS de la validación básica.
        # info:            objeto con metadatos (nombre del campo, etc.).
        #
        # En v2 este validador es SIMBÓLICO (solo retorna v sin modificar).
        # La validación real de "exactamente una ciudad = 1" se hace en el
        # endpoint con HTTPException. En v3 la movemos aquí con model_validator.
        # La razón: field_validator solo ve UN campo a la vez, no puede
        # comparar city_guayaquil con city_manta. Para eso necesitamos
        # model_validator (que ve todos los campos juntos).
        return v

    model_config = {"populate_by_name": True, "json_schema_extra": {
        # model_config: diccionario de configuración de Pydantic v2.
        # populate_by_name=True: permite usar el nombre original del campo
        #                        O su alias. Ej: acepta tanto "construction_area_sqm"
        #                        como "area_m2" en el JSON de entrada.
        # json_schema_extra: datos adicionales para el esquema JSON que
        #                    FastAPI usa para generar /docs.

        "example": {
            # example: aparece como ejemplo precargado en Swagger UI.
            #          El usuario puede hacer clic en "Try it out" y ya
            #          tiene valores de prueba razonables.
            "bedrooms": 3, "bathrooms": 2, "parking_spots": 2,
            "area_m2": 200.0, "lat": -0.18, "lon": -78.48,
            "city_guayaquil": 0, "city_manta": 0, "city_quito": 1,
        }
    }}

    def to_dataframe(self) -> pd.DataFrame:
        # Método de conveniencia: convierte ESTA instancia de PropiedadInput
        # en un DataFrame de pandas con 1 fila lista para el modelo.
        # self se refiere a la instancia concreta (los datos que envió el usuario).
        return pd.DataFrame([[
            self.bedrooms, self.bathrooms, self.parking_spots,
            self.construction_area_sqm, self.latitude, self.longitude,
            self.city_guayaquil, self.city_manta, self.city_quito,
        ]], columns=FEATURES)


class PropiedadOutput(BaseModel):
    # Modelo para la RESPUESTA de la API. Define la estructura del JSON
    # que el cliente recibirá. Esto sirve para:
    #   1. Documentar la respuesta en /docs.
    #   2. Validar que la API siempre devuelve el formato correcto.
    #   3. Filtrar campos: si accidentalmente devolvemos datos extra,
    #      FastAPI los elimina automáticamente.

    precio_usd: float = Field(..., description="Precio estimado en USD")
    modelo: str = "Random Forest"
    # "Random Forest": valor por defecto fijo. El cliente siempre sabrá
    #                  qué tipo de modelo generó la predicción.
    version_api: str = "2.0.0"
    # Incluir la versión en la respuesta es una buena práctica:
    # el cliente puede verificar compatibilidad.


class ErrorResponse(BaseModel):
    # Modelo para respuestas de error estructuradas.
    # Aunque no se usa en todos los endpoints, está definido para
    # mantener consistencia en el formato de errores.
    detalle: str
    tipo: str = "error_de_validacion"


# ── 6. ENDPOINTS ────────────────────────────────────────────────────────

@app.get("/")
def root():
    # Endpoint raíz: devuelve un índice de todos los endpoints disponibles.
    # Similar a la tabla de contenido de la API.
    return {
        "api": "Predicción Inmobiliaria",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Estado del servicio",
            "/predict": "Predicción individual (POST)",
            "/predict_from_query": "Predicción individual (GET)",
            "/features": "Lista de características del modelo",
        }
    }


@app.get("/health")
def health():
    # Health check mejorado respecto a v1: ahora también informa
    # si el modelo se cargó correctamente.
    return {"status": "ok", "modelo_cargado": True}
    # modelo_cargado=True: si el modelo falló al cargar, el servidor
    # ni siquiera arrancaría (por el raise en el try/except de arriba).
    # Pero este campo es útil si en el futuro implementamos "lazy loading".


@app.get("/features")
def features():
    """Devuelve las características que el modelo espera y su importancia."""
    # ── feature_importances_: atributo de RandomForestRegressor ─────────
    # Cada árbol del bosque evalúa cuánto contribuye cada variable a
    # reducir el error. El promedio de todos los árboles es la importancia.
    # Valores más altos = variable más determinante para el precio.
    importances = model.feature_importances_.tolist()
    # .tolist(): convierte el array de NumPy a lista de Python nativa
    #            (serializable a JSON sin problemas).

    return {
        "features": [
            # List comprehension: para cada par (nombre, importancia)...
            {"nombre": f, "importancia": round(imp, 4)}
            for f, imp in sorted(
                # zip(FEATURES, importances): empareja cada nombre con su importancia.
                # Ej: [("BEDROOMS", 0.012), ("BATHROOMS", 0.060), ...]
                zip(FEATURES, importances),
                key=lambda x: -x[1]  # ordena por importancia DESCENDENTE (negativo = inverso).
            )
        ],
        "total_features": len(FEATURES),  # 9
    }


@app.post("/predict", response_model=PropiedadOutput)
# @app.post: esta vez usamos POST en vez de GET.
#   POST: los datos viajan en el BODY de la petición (no en la URL).
#         Mejor práctica para datos complejos o cuando no quieres
#         que los parámetros queden registrados en logs del servidor.
# response_model=PropiedadOutput: FastAPI valida que la respuesta
#   cumpla el esquema de PropiedadOutput. Si olvidamos un campo o
#   devolvemos datos extra, FastAPI lo detecta y/o filtra.
async def predict_post(propiedad: PropiedadInput):
    # async def: declara una corrutina asíncrona. FastAPI la ejecuta
    #            en un event loop, permitiendo concurrencia.
    #            En este caso no hay operaciones I/O, pero es buena práctica.
    #
    # propiedad: PropiedadInput — FastAPI automáticamente:
    #   1. Lee el JSON del body de la petición.
    #   2. Valida tipos y rangos según PropiedadInput.
    #   3. Si falla, devuelve 422 con detalles de qué campo falló.
    #   4. Si pasa, crea una instancia de PropiedadInput con los datos.

    """Predice el precio de una propiedad (envío por POST con JSON)."""

    # ── Validación manual de ciudad única ───────────────────────────────
    # Como field_validator no puede ver todos los campos a la vez,
    # hacemos esta comprobación en el endpoint.
    ciudades = [propiedad.city_guayaquil, propiedad.city_manta, propiedad.city_quito]
    # Lista con los 3 valores (0 o 1).
    if sum(ciudades) != 1:
        # Si la suma no es 1, el usuario envió 0 en todas o 1 en varias.
        # Ej: Guayaquil=1, Quito=1 → suma=2 → error.
        raise HTTPException(
            # HTTPException: excepción especial de FastAPI.
            # NO es lo mismo que un return {"error": "..."}.
            # Al lanzar HTTPException, FastAPI:
            #   1. NO ejecuta el resto de la función.
            #   2. Devuelve una respuesta HTTP con el status_code indicado.
            #   3. Incluye el detalle en el body de la respuesta.
            status_code=422,
            # 422 Unprocessable Entity: "entiendo el JSON, pero los datos no
            # son válidos semánticamente". Es el código estándar para errores
            # de validación en APIs REST.
            detail=(
                "Debe seleccionar exactamente una ciudad "
                "(una de city_guayaquil, city_manta, city_quito = 1, las demás 0)."
            ),
        )

    try:
        # ── Convertir entrada a DataFrame ───────────────────────────────
        df = propiedad.to_dataframe()
        # Usamos el método que definimos en PropiedadInput.
        # Ventaja: la lógica de conversión está encapsulada en el modelo.

        # ── Predecir ────────────────────────────────────────────────────
        precio = float(model.predict(df)[0])
        logger.info("Predicción realizada: %.2f USD", precio)
        # %.2f: formatea el precio con 2 decimales en el mensaje de log.
        # Ej: "Predicción realizada: 287452.63 USD"

        return PropiedadOutput(precio_usd=round(precio, 2))
        # Construimos una instancia de PropiedadOutput.
        # FastAPI la convierte a JSON automáticamente.
        # response_model=PropiedadOutput valida que el objeto es correcto.

    except Exception as e:
        # Si algo falla DENTRO de la predicción (modelo corrupto, error
        # de memoria, etc.), capturamos la excepción y devolvemos
        # un error 500 (Internal Server Error) con mensaje descriptivo.
        logger.error("Error en predicción: %s", e)
        # logger.error(): nivel ERROR. Más alto que WARNING, menos que CRITICAL.
        raise HTTPException(
            status_code=500,
            # 500 Internal Server Error: "falló el servidor, no es culpa tuya".
            detail=f"Error interno al predecir: {str(e)}",
            # Incluimos el mensaje de la excepción para debugging.
            # En producción real, NO expondrías el error interno al usuario
            # (podría filtrar información sensible). Mejor: "Error interno. Contacte al administrador."
        )


@app.get("/predict_from_query", response_model=PropiedadOutput)
async def predict_from_query(
    # Este endpoint permite hacer predicciones por GET (query string),
    # igual que en v1, pero reutilizando la validación de PropiedadInput.
    # Todos los parámetros tienen VALOR POR DEFECTO (no son obligatorios).
    # Si no se envían, se usa el ejemplo de Quito.

    bedrooms: int = 3,
    # = 3: valor por defecto. Si el usuario visita /predict_from_query
    #       sin parámetros, se asume una propiedad de 3 habitaciones.
    bathrooms: int = 2,
    parking_spots: int = 2,
    area_m2: float = 200.0,
    lat: float = -0.18,
    lon: float = -78.48,
    city_guayaquil: int = 0,
    city_manta: int = 0,
    city_quito: int = 1,
    # Por defecto: Quito (city_quito=1, las demás 0).
):
    """Predice el precio de una propiedad (parámetros por query string)."""
    # Construimos una instancia de PropiedadInput con los valores recibidos.
    # FastAPI valida que los valores cumplan las restricciones
    # (ge=1, le=20, etc.) antes de ejecutar esta función.
    propiedades = PropiedadInput(
        bedrooms=bedrooms, bathrooms=bathrooms, parking_spots=parking_spots,
        area_m2=area_m2, lat=lat, lon=lon,
        city_guayaquil=city_guayaquil, city_manta=city_manta, city_quito=city_quito,
    )
    # Reutilizamos el endpoint POST para no duplicar la lógica de predicción.
    # await: como predict_post es async, debemos usar await para llamarla.
    return await predict_post(propiedades)


# ── 7. PUNTO DE ENTRADA ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Solo se ejecuta si corremos este archivo directamente.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    # log_level="info": uvicorn también tiene su propio logging.
    #                   "info" muestra las peticiones HTTP en la consola.
