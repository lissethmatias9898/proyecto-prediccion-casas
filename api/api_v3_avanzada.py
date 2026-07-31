# -*- coding: utf-8 -*-
"""
API v3 — Avanzada
═══════════════════════════════════════════════════════════════════════
Versión profesional lista para producción. Recoge todo lo aprendido en
v1 y v2 y añade características de una API del mundo real.

¿Qué aporta sobre v2?
  1. Configuración por entorno (12-factor app) ─── variables de entorno.
  2. Lifespan ─── carga/descarga controlada del modelo.
  3. CORS ──── permite llamadas desde navegadores web.
  4. Middleware de logging ─── mide tiempo de cada petición.
  5. Validación con model_validator (todos los campos a la vez).
  6. Batch prediction ─── predice N propiedades en una sola llamada.
  7. Errores RFC 7807 ─── formato estándar de errores HTTP.
  8. Manejadores globales de excepciones.
  9. Endpoint /model-info con metadatos completos.

Cómo ejecutar:
  # Con valores por defecto:
  python api_v3_avanzada.py

  # Personalizado con variables de entorno:
  MODEL_PATH=./modelo_inmobiliario.pkl API_PORT=8080 LOG_LEVEL=debug uvicorn api_v3_avanzada:app
"""

from __future__ import annotations
# from __future__ import annotations: permite usar tipos forward-reference
# sin comillas (ej: List[PropiedadInput] en lugar de List["PropiedadInput"]).
# Esto es necesario porque Pydantic usa anotaciones de tipo y a veces las
# clases se referencian entre sí antes de estar definidas.

# ── 1. IMPORTACIONES ────────────────────────────────────────────────────

import logging
# logging: sistema de registro de eventos. Configurado desde variables de entorno.

import os
# os: interactúa con el sistema operativo. Lo usamos para leer variables
#     de entorno con os.getenv().

import time
# time: medición de tiempo. time.perf_counter() da precisión de nanosegundos
#       para medir cuánto tarda una predicción o una petición.

from dotenv import load_dotenv
# python-dotenv: carga variables desde el archivo .env como si fueran
#                variables de entorno reales. Así no hay que exportarlas
#                manualmente en cada terminal.

load_dotenv()
# Lee el archivo .env (si existe) y carga sus variables en os.environ.
# Si no encuentra .env, no falla: usa los defaults definidos en Settings.
# Esto permite que el mismo código funcione con y sin archivo .env.

from contextlib import asynccontextmanager
# asynccontextmanager: decorador que convierte una función async generator
#                      en un context manager asíncrono. Lo usamos para el
#                      lifespan de FastAPI (eventos de inicio/cierre).

from datetime import datetime
# datetime: para timestamps legibles (datetime.now().isoformat()).

from typing import List, Optional
# List:   tipo para listas (List[PropiedadInput]).
# Optional: tipo para valores que pueden ser None.

import joblib
# joblib: carga/guarda modelos scikit-learn.

import pandas as pd
# pandas: DataFrames para las predicciones.

from pydantic import BaseModel, Field, field_validator, model_validator
# model_validator: NUEVO en v3. A diferencia de field_validator (que valida
#                  UN campo), model_validator valida el modelo COMPLETO
#                  (todos los campos ya poblados). Ideal para comprobar
#                  que las 3 ciudades suman exactamente 1.

from fastapi import FastAPI, HTTPException, Request, Response
# Request:   representa la petición HTTP entrante (URL, headers, método, etc.).
#            Lo usamos en middleware y exception handlers.
# Response:  representa la respuesta HTTP saliente. Lo usamos en middleware.

from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware: middleware oficial de FastAPI para CORS.
#                 Permite que navegadores web desde otros dominios
#                 llamen a esta API.

from fastapi.responses import JSONResponse
# JSONResponse: construye respuestas JSON manualmente. Usada en los
#               exception handlers para respuestas de error personalizadas.


# ── 2. CONFIGURACIÓN POR ENTORNO (12-FACTOR APP) ────────────────────────
# En lugar de hardcodear valores (como en v1 y v2), leemos desde variables
# de entorno. Esto permite cambiar el comportamiento sin modificar código:
#   - En desarrollo: puerto 8000, logs DEBUG, CORS abierto.
#   - En producción: puerto 80, logs WARNING, CORS restringido.

class Settings:
    # Clase contenedora de configuración. No es un BaseModel de Pydantic
    # (podría serlo con pydantic-settings), sino una clase normal para
    # mantener el ejemplo simple y sin dependencias extra.

    app_name: str = "API Inmobiliaria v3"
    # Nombre de la aplicación (aparece en /docs y logs).

    version: str = "3.0.0"
    # Versión semántica de la API.

    model_path: str = os.getenv("MODEL_PATH", "modelo_inmobiliario.pkl")
    # os.getenv("MODEL_PATH", "modelo_inmobiliario.pkl"):
    #   Si la variable de entorno MODEL_PATH existe, usa su valor.
    #   Si no existe, usa el valor por defecto "modelo_inmobiliario.pkl".
    #   Esto permite cambiar el modelo sin tocar el código:
    #     export MODEL_PATH=/ruta/compartida/modelo_v2.pkl

    host: str = os.getenv("API_HOST", "0.0.0.0")
    # API_HOST: dirección donde escucha el servidor.
    #           0.0.0.0 = todas las interfaces (accesible desde fuera).
    #           127.0.0.1 = solo localhost.

    port: int = int(os.getenv("API_PORT", "8000"))
    # int(...): os.getenv siempre devuelve string. Convertimos a entero.
    #           Puerto 8000 es el default de FastAPI.

    log_level: str = os.getenv("LOG_LEVEL", "info").lower()
    # LOG_LEVEL: nivel de logging (debug, info, warning, error, critical).
    #            .lower() normaliza a minúsculas (por si el usuario pone "INFO").

    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    # CORS_ORIGINS: orígenes permitidos para CORS.
    #   "*" (default) = permite cualquier origen (solo para desarrollo).
    #   Producción: "https://miapp.com,https://admin.miapp.com"
    #   .split(","): convierte el string en lista: "*" → ["*"]

    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    # MAX_BATCH_SIZE: cuántas propiedades máximo se pueden predecir
    #                 en una sola llamada batch. Límite para evitar
    #                 abusos o sobrecarga de memoria.

    predict_timeout_ms: int = int(os.getenv("PREDICT_TIMEOUT_MS", "10000"))
    # PREDICT_TIMEOUT_MS: timeout para predicciones en milisegundos.
    #                     10,000 ms = 10 segundos.
    #                     No implementado en este ejemplo, pero es un
    #                     placeholder para futuras mejoras.

settings = Settings()
# Instanciamos la configuración UNA vez. Todos los componentes de la API
# leen de esta instancia (settings.model_path, settings.port, etc.).


# ── 3. CONFIGURACIÓN DE LOGGING ─────────────────────────────────────────
# Similar a v2, pero el nivel de log se lee de settings (variable de entorno).

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    # getattr(logging, "INFO"): convierte el string "info" en la constante
    #                           logging.INFO (que vale 20). Si el string no
    #                           existe, usa logging.INFO como fallback.
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    # %(name)s: incluye el nombre del logger (settings.app_name).
    #           Así sabemos si el mensaje viene de la API, de uvicorn, etc.
)
logger = logging.getLogger(settings.app_name)
# getLogger("API Inmobiliaria v3"): logger con nombre descriptivo.


# ── 4. CARGA LAZY DEL MODELO (LIFESPAN) ─────────────────────────────────
# En v1 y v2, el modelo se carga como variable global al hacer import.
# En v3, usamos el PATRÓN LIFESPAN de FastAPI:
#   - Ventaja 1: control preciso de CUÁNDO se carga (después de configurar logs).
#   - Ventaja 2: tareas de limpieza al apagar (liberar memoria).
#   - Ventaja 3: si la carga falla, podemos devolver un error amigable
#                en lugar de un traceback.

model = None
# Inicialmente no hay modelo. Se asigna en el lifespan.
# Los endpoints verifican `if model is None` antes de predecir.

model_info = {}
# Diccionario con metadatos del modelo (tipo, n_estimators, tiempo de carga, etc.).
# Se llena en cargar_modelo() y se expone en el endpoint /model-info.

FEATURES = [
    "BEDROOMS", "BATHROOMS", "PARKING_SPOTS", "CONSTRUCTION_AREA_SQM",
    "LATITUDE", "LONGITUDE",
    "CITY_Guayaquil", "CITY_Manta", "CITY_Quito",
]
# Lista de 9 features que el modelo espera. Mismo orden que v1 y v2.

CIUDADES = ["city_guayaquil", "city_manta", "city_quito"]
# Lista auxiliar con los nombres de los campos de ciudad (para referencias).


def cargar_modelo() -> tuple:
    """
    Carga el modelo desde disco y extrae sus metadatos.

    Returns:
        tuple: (modelo_scikit_learn, dict_con_metadatos)

    ¿Por qué una función separada y no código inline en lifespan?
      - Separación de responsabilidades: una función = una tarea.
      - Testeable: podemos probar cargar_modelo() sin levantar el servidor.
      - Reutilizable: si en el futuro hay un endpoint "recargar modelo",
        podemos llamar a esta función sin duplicar código.
    """
    global model, model_info
    # global: indica que vamos a MODIFICAR las variables globales
    #         model y model_info, no crear variables locales.

    logger.info("Cargando modelo desde %s ...", settings.model_path)

    t0 = time.perf_counter()
    # time.perf_counter(): reloj de alta precisión. Mide tiempo real
    # (wall clock), no tiempo de CPU. Ideal para medir duraciones.
    m = joblib.load(settings.model_path)
    # Carga el archivo .pkl. m ahora es un RandomForestRegressor.

    elapsed = time.perf_counter() - t0
    # Tiempo transcurrido en segundos (con decimales de nanosegundos).

    info = {
        "tipo": type(m).__name__,
        # type(m).__name__: nombre de la clase del modelo.
        # Para RandomForestRegressor → "RandomForestRegressor".

        "n_estimators": getattr(m, "n_estimators", "N/A"),
        # getattr(obj, "attr", default): obtiene el atributo si existe.
        # Si no existe, devuelve "N/A" en vez de lanzar AttributeError.
        # n_estimators: número de árboles en el bosque (100 en nuestro caso).

        "max_features": getattr(m, "max_features", "N/A"),
        # max_features: cuántas features considera cada árbol al hacer un split.
        # Por defecto en RandomForestRegressor es 1.0 (todas).

        "n_features_in_": getattr(m, "n_features_in_", len(FEATURES)),
        # n_features_in_: cuántas features recibió el modelo al entrenarse.
        # Debería coincidir con len(FEATURES) = 9.

        "tiempo_carga_ms": round(elapsed * 1000, 2),
        # Convertimos segundos a milisegundos y redondeamos a 2 decimales.
        # Ej: 0.34215 seg → 342.15 ms.

        "cargado_en": datetime.now().isoformat(),
        # Timestamp ISO 8601 legible: "2025-07-05T13:01:42.123456".

        "features": FEATURES,
        # Lista de features para referencia del cliente.
    }
    logger.info("Modelo cargado en %.2f ms", elapsed * 1000)
    return m, info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de ciclo de vida de la aplicación FastAPI.

    Todo lo que está ANTES del `yield` se ejecuta al INICIAR el servidor
    (startup). Todo lo que está DESPUÉS del `yield` se ejecuta al APAGAR
    el servidor (shutdown).

    @asynccontextmanager convierte esta función en un context manager
    asíncrono. FastAPI lo usa internamente para saber cuándo ejecutar
    cada parte.

    Estructura típica:
        async with lifespan(app):
            # Código antes del yield = startup
            yield
            # Código después del yield = shutdown
    """
    global model, model_info
    # ── STARTUP ────────────────────────────────────────────────────────
    model, model_info = cargar_modelo()
    # Carga el modelo y guarda sus metadatos.
    # Si la carga falla, la excepción se propaga y el servidor no arranca.

    yield
    # yield: cede el control a FastAPI. La aplicación está VIVA entre
    #        el yield de startup y el código de shutdown.
    #        Mientras la app está viva, se procesan peticiones normalmente.

    # ── SHUTDOWN ───────────────────────────────────────────────────────
    logger.info("API detenida. Liberando recursos.")
    model = None
    # Libera la referencia al modelo para que el garbage collector
    # de Python pueda liberar la memoria. En modelos pequeños esto
    # no es crítico, pero en modelos de varios GB es importante.


# ── 5. CREACIÓN DE LA APLICACIÓN ────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    # Título en Swagger UI.

    description=(
        "API avanzada para predicción de precios inmobiliarios.\n\n"
        "**Modelo:** Random Forest entrenado sobre datos de Ecuador.\n"
        "**Features:** área de construcción, habitaciones, baños, parking, "
        "ubicación geográfica y ciudad (Quito, Guayaquil, Manta).\n\n"
        "Seleccione **exactamente una** ciudad; las demás deben ser 0."
    ),
    # description: texto Markdown en la cabecera de /docs.

    version=settings.version,
    # Versión desde configuración (3.0.0).

    lifespan=lifespan,
    # lifespan: NUEVO en v3. Pasamos el context manager asíncrono.
    #           FastAPI ejecutará el código antes del yield al iniciar
    #           y el código después del yield al detenerse.

    contact={"name": "Equipo EELA - Ciencia de Datos", "url": "https://eela.ec"},
    # Información de contacto con URL.
)


# ── 6. MIDDLEWARE ───────────────────────────────────────────────────────
# Middleware = código que se ejecuta ANTES y DESPUÉS de cada petición.
# Es como un "interceptor" que envuelve todos los endpoints.
# Se ejecutan en el orden en que se añaden.

# ── 6a. CORS Middleware ─────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Problema: si tu frontend está en https://miapp.com y tu API en
# https://api.miapp.com, el navegador bloquea las peticiones por seguridad
# (same-origin policy). CORS le dice al navegador "esta API acepta
# peticiones desde estos orígenes".

app.add_middleware(
    CORSMiddleware,
    # CORSMiddleware: implementación oficial de FastAPI para CORS.

    allow_origins=settings.cors_origins,
    # Lista de orígenes permitidos. ["*"] = cualquier origen (desarrollo).
    # En producción: ["https://miapp.com", "https://admin.miapp.com"].

    allow_credentials=True,
    # Permitir envío de cookies/autenticación cross-origin.

    allow_methods=["*"],
    # Métodos HTTP permitidos. ["*"] = GET, POST, PUT, DELETE, etc.

    allow_headers=["*"],
    # Cabeceras HTTP permitidas. ["*"] = todas.
)


# ── 6b. Middleware de Logging Personalizado ─────────────────────────────
# Este middleware mide cuánto tarda CADA petición y lo registra en logs.
# También añade una cabecera X-Response-Time-Ms a todas las respuestas.

@app.middleware("http")
# @app.middleware("http"): registra esta función como middleware HTTP.
#                          Se ejecuta para TODAS las peticiones.
async def log_requests(request: Request, call_next):
    """
    Middleware que registra cada petición con su duración.

    Flujo:
      1. Se recibe la petición (request).
      2. Registramos el tiempo de inicio.
      3. call_next(request) → ejecuta el endpoint correspondiente.
      4. Registramos el tiempo de fin y calculamos la duración.
      5. Añadimos cabecera X-Response-Time-Ms.
      6. Devolvemos la respuesta.

    call_next es una función que FastAPI proporciona. Al llamarla,
    se ejecuta el siguiente middleware en la cadena, y eventualmente
    el endpoint. Devuelve la respuesta que el endpoint generó.
    """
    start = time.perf_counter()
    # Marca de tiempo ANTES de procesar la petición.

    response: Response = await call_next(request)
    # await call_next(request): ejecuta el endpoint (o el siguiente middleware).
    #                           await es necesario porque call_next es async.
    #                           response contiene la respuesta HTTP completa.

    elapsed_ms = (time.perf_counter() - start) * 1000
    # Calcula la duración y convierte a milisegundos.

    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,      # GET, POST, etc.
        request.url.path,    # /predict, /health, etc.
        response.status_code, # 200, 422, 500, etc.
        elapsed_ms,          # duración en ms
    )
    # Ejemplo de log: "POST /predict -> 200 (12.3 ms)"

    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    # Añade cabecera personalizada a la respuesta.
    # El cliente puede usar esta cabecera para monitorear latencia.

    return response
    # Devuelve la respuesta (con la cabecera añadida) al cliente.


# ── 7. MODELOS PYDANTIC ─────────────────────────────────────────────────

class PropiedadInput(BaseModel):
    """
    Modelo de entrada para predicción individual.

    Hereda de BaseModel, lo que activa:
      - Validación automática de tipos.
      - Validación de restricciones (ge, le).
      - Conversión de alias (populate_by_name).
      - Documentación en Swagger (/docs).
      - Errores descriptivos automáticos.
    """
    bedrooms: int = Field(..., ge=1, le=20, description="Habitaciones (1-20)")
    bathrooms: int = Field(..., ge=1, le=20, description="Baños (1-20)")
    parking_spots: int = Field(..., ge=0, le=20, description="Estacionamiento (0-20)")
    construction_area_sqm: float = Field(
        ..., alias="area_m2", ge=10, le=10000,
        description="Área de construcción en m² (10-10000)",
    )
    latitude: float = Field(..., alias="lat", ge=-90, le=90, description="Latitud (-90 a 90)")
    longitude: float = Field(..., alias="lon", ge=-180, le=180, description="Longitud (-180 a 180)")
    city_guayaquil: int = Field(0, ge=0, le=1, description="Guayaquil (0/1)")
    city_manta: int = Field(0, ge=0, le=1, description="Manta (0/1)")
    city_quito: int = Field(0, ge=0, le=1, description="Quito (0/1)")

    @field_validator("city_guayaquil", "city_manta", "city_quito")
    @classmethod
    def validar_binario(cls, v):
        """
        Validador de campo individual: asegura que cada ciudad sea 0 o 1.

        cls:  la clase PropiedadInput (classmethod).
        v:    el valor del campo después del parseo de tipos.

        Se ejecuta UNA VEZ por cada campo listado en el decorador.
        Si el usuario envía city_guayaquil=5, esta función recibe v=5
        y lanza ValueError.
        """
        if v not in (0, 1):
            raise ValueError("Solo valores 0 o 1")
        return v
        # Retornar el valor es obligatorio en field_validator.
        # Podemos modificar v antes de retornarlo (ej: normalizar):
        #   return bool(v)  # convertiría 0→False, 1→True
        # Pero en nuestro caso solo validamos.

    @model_validator(mode="after")
    def validar_una_ciudad(self):
        """
        Validador de modelo COMPLETO: asegura que exactamente una ciudad = 1.

        A diferencia de field_validator (que ve UN campo a la vez),
        model_validator recibe el modelo YA CONSTRUIDO (self) con todos
        los campos poblados y validados.

        mode="after": se ejecuta DESPUÉS de que Pydantic haya validado
                      los tipos y restricciones básicas. Garantiza que
                      los valores ya son enteros y están en rango.

        Flujo completo de validación de un PropiedadInput:
          1. Pydantic parsea el JSON: strings → int/float.
          2. Validaciones de Field: ge, le, tipo.
          3. field_validator: validar_binario (para cada ciudad).
          4. model_validator: validar_una_ciudad (todo junto).
          5. Si todo pasa, se crea la instancia.
        """
        ciudades = [self.city_guayaquil, self.city_manta, self.city_quito]
        # self = instancia ya construida. Podemos acceder a todos los campos.

        if sum(ciudades) != 1:
            raise ValueError(
                "Exactamente una ciudad debe ser 1 (las demás 0). "
                f"Recibido: Guayaquil={self.city_guayaquil}, "
                f"Manta={self.city_manta}, Quito={self.city_quito}"
            )
            # El mensaje de error incluye los valores recibidos para ayudar
            # al usuario a corregir su petición.
        return self
        # model_validator DEBE retornar la instancia (self).

    model_config = {
        "populate_by_name": True,
        # Permite usar el alias (area_m2) o el nombre real (construction_area_sqm)
        # indistintamente en el JSON de entrada.

        "json_schema_extra": {
            # Datos adicionales para el JSON Schema que alimenta Swagger UI.
            "example": {
                "bedrooms": 3, "bathrooms": 2, "parking_spots": 2,
                "area_m2": 200.0, "lat": -0.18, "lon": -78.48,
                "city_guayaquil": 0, "city_manta": 0, "city_quito": 1,
            }
        }
    }

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte esta propiedad en un DataFrame de 1 fila para el modelo."""
        return pd.DataFrame([[
            self.bedrooms, self.bathrooms, self.parking_spots,
            self.construction_area_sqm, self.latitude, self.longitude,
            self.city_guayaquil, self.city_manta, self.city_quito,
        ]], columns=FEATURES)


class PropiedadOutput(BaseModel):
    """Modelo de respuesta para predicción individual."""
    precio_usd: float = Field(..., description="Precio estimado en USD")
    modelo: str = Field("Random Forest", description="Tipo de modelo usado")


class ErrorDetail(BaseModel):
    """
    Formato de error estructurado siguiendo RFC 7807 (Problem Details for HTTP APIs).

    RFC 7807 define un estándar para respuestas de error en APIs REST.
    Ventajas sobre {"error": "mensaje"}:
      - type: URI que identifica el tipo de error (el cliente puede reaccionar).
      - title: resumen legible.
      - status: código HTTP (redundante con la cabecera, pero útil en el body).
      - detail: explicación detallada para humanos.
      - instance: URI del recurso que generó el error (ayuda al debugging).
    """
    type: str = Field("about:blank", description="URI del tipo de error")
    # about:blank: valor por defecto de la RFC cuando no hay una URI específica.

    title: str = Field("Error de validación", description="Título legible")

    status: int = Field(422, description="Código HTTP")
    # 422 por defecto (error de validación del cliente).

    detail: str = Field(..., description="Descripción del error")
    # Campo obligatorio: la explicación concreta de qué falló.

    instance: Optional[str] = Field(None, description="Endpoint que generó el error")
    # Optional[str]: puede ser None si no se conoce.


# ── 8. MODELOS PARA BATCH PREDICTION ────────────────────────────────────

class BatchInput(BaseModel):
    """
    Entrada para predicción masiva (batch).

    Contiene una lista de PropiedadInput. Pydantic valida
    automáticamente cada elemento de la lista.
    """
    propiedades: List[PropiedadInput] = Field(
        ...,
        min_length=1,
        # min_length: la lista debe tener al menos 1 elemento.

        max_length=settings.max_batch_size,
        # max_length: límite configurable por variable de entorno.
        #             Evita que alguien envíe 1 millón de propiedades
        #             y sature el servidor.

        description=f"Lista de propiedades (1-{settings.max_batch_size})",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "propiedades": [
                # Primer elemento: propiedad en Quito
                {"bedrooms": 3, "bathrooms": 2, "parking_spots": 2,
                 "area_m2": 200.0, "lat": -0.18, "lon": -78.48,
                 "city_guayaquil": 0, "city_manta": 0, "city_quito": 1},
                # Segundo elemento: propiedad en Guayaquil
                {"bedrooms": 4, "bathrooms": 3, "parking_spots": 2,
                 "area_m2": 350.0, "lat": -2.19, "lon": -79.89,
                 "city_guayaquil": 1, "city_manta": 0, "city_quito": 0},
            ]
        }
    }}


class BatchOutput(BaseModel):
    """Respuesta para predicción masiva."""
    predicciones: List[PropiedadOutput]
    # Lista de predicciones, una por cada propiedad enviada.

    total: int
    # Número total de predicciones realizadas (útil para verificar).

    procesado_en_ms: float
    # Tiempo que tomó procesar el batch completo.
    # Permite al cliente evaluar rendimiento.


# ── 9. ENDPOINTS ────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Endpoint raíz: muestra información general de la API y los endpoints disponibles.

    Es la primera impresión que un desarrollador tiene de la API.
    Por eso incluimos versión, enlace a /docs y lista de endpoints.
    """
    return {
        "api": settings.app_name,
        "version": settings.version,
        "documentacion": "/docs",
        # /docs es la ruta donde FastAPI sirve Swagger UI automáticamente.
        # No necesitamos crear un endpoint para /docs, FastAPI lo hace solo.

        "endpoints": [
            "/health",
            "/model-info",
            "/features",
            "/predict (POST)",
            "/predict/batch (POST)",
        ],
    }


@app.get("/health")
def health():
    """
    Health check con estado del modelo.

    A diferencia de v1 y v2, aquí verificamos si el modelo está cargado
    (model is not None). Esto permite detectar si algo falló durante el
    lifespan o si el modelo fue descargado por alguna razón.

    El timestamp ayuda a determinar cuándo fue la última vez que el
    servidor respondió correctamente (útil para monitoreo).
    """
    return {
        "status": "ok",
        "modelo_cargado": model is not None,
        # model is not None → True si el modelo se cargó correctamente.
        # Si es False, los endpoints de predicción devolverán 503.

        "timestamp": datetime.now().isoformat(),
        # Timestamp ISO 8601 de este momento exacto.
        # Ej: "2025-07-05T13:01:42.123456"
    }


@app.get("/model-info")
def model_info_endpoint():
    """
    Información detallada del modelo.

    Expone metadatos útiles para monitoreo, auditoría y debugging:
      - tipo de modelo (RandomForestRegressor)
      - parámetros (n_estimators, max_features)
      - tiempo que tomó cargarlo
      - cuándo se cargó
      - importancia de cada feature ordenada de mayor a menor

    Si el modelo no está disponible, devuelve 503 Service Unavailable.
    """
    if model is None:
        # Verificación defensiva: si lifespan falló, no exponemos metadatos.
        raise HTTPException(
            status_code=503,
            # 503 Service Unavailable: "el servidor no puede manejar la
            # petición en este momento (pero podría en el futuro)".
            detail="Modelo no disponible"
        )

    return {
        **model_info,
        # **model_info: "desempaqueta" el diccionario de metadatos.
        # Equivale a escribir todas las claves de model_info una por una.
        # Ej: tipo="RandomForestRegressor", n_estimators=100, etc.

        "importancias": [
            # Lista de features ordenadas por importancia descendente.
            {"feature": f, "importancia": round(imp, 4)}
            for f, imp in sorted(
                zip(FEATURES, model.feature_importances_.tolist()),
                # zip: empareja nombre con importancia.
                # .tolist(): array numpy → lista Python.
                key=lambda x: -x[1],
                # lambda x: -x[1] → ordena por importancia (índice 1) descendente.
                # El signo negativo invierte el orden.
            )
        ],
    }


@app.get("/features")
def features():
    """
    Lista de características que espera el modelo.

    Útil para que los clientes sepan exactamente qué campos enviar
    sin tener que leer la documentación completa.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return {
        "features": FEATURES,
        # La lista completa de 9 features.

        "n_features": len(FEATURES),
        # 9

        "n_features_in_model": model.n_features_in_,
        # Cuántas features recibió el modelo durante el entrenamiento.
        # Debería coincidir con len(FEATURES). Si no coincide, hay
        # una discrepancia entre el modelo y la API.
    }


@app.post("/predict", response_model=PropiedadOutput)
async def predict(propiedad: PropiedadInput):
    """
    Predice el precio de UNA propiedad.

    Recibe un JSON en el body con los 9 campos de PropiedadInput.
    Valida tipos, rangos, y que exactamente una ciudad = 1.

    Returns:
        PropiedadOutput con el precio estimado en USD.

    Errores posibles:
        422: datos inválidos (tipos incorrectos, fuera de rango, ciudades mal).
        500: error interno del modelo.
        503: modelo no disponible (no cargado).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        df = propiedad.to_dataframe()
        # Construye DataFrame de 1 fila.

        precio = float(model.predict(df)[0])
        # Predice y extrae el valor escalar.

        logger.info(
            "Predicción: %.2f USD (ciudad=%d,%d,%d)",
            precio,
            propiedad.city_guayaquil,
            propiedad.city_manta,
            propiedad.city_quito,
        )
        # Log estructurado con precio y ciudad para análisis posteriores.

        return PropiedadOutput(precio_usd=round(precio, 2))

    except Exception as e:
        logger.exception("Error en predicción")
        # logger.exception(): registra el traceback COMPLETO además del mensaje.
        # Muy útil para debugging porque muestra la pila de llamadas.
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    """
    Predice el precio de MÚLTIPLES propiedades en una sola llamada.

    Ventajas del batch vs llamadas individuales:
      1. Menos latencia: 1 round-trip HTTP en vez de N.
      2. Mayor throughput: model.predict() sobre un array es más rápido
         que N llamadas individuales (vectorización interna de scikit-learn).
      3. Mejor para valuación de carteras inmobiliarias completas.

    Límite de batch configurable vía MAX_BATCH_SIZE (default 100).

    Returns:
        BatchOutput con lista de predicciones, total y tiempo de procesamiento.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    t0 = time.perf_counter()
    # Inicia el cronómetro.

    try:
        rows = []
        # Lista para acumular las filas de datos.

        for p in batch.propiedades:
            # Itera sobre cada propiedad en la lista recibida.
            # Cada p ya es una instancia validada de PropiedadInput.
            rows.append([
                p.bedrooms, p.bathrooms, p.parking_spots,
                p.construction_area_sqm, p.latitude, p.longitude,
                p.city_guayaquil, p.city_manta, p.city_quito,
            ])
            # Cada fila es una lista de 9 valores.

        df = pd.DataFrame(rows, columns=FEATURES)
        # Crea DataFrame de N filas × 9 columnas.
        # N = len(batch.propiedades).

        preds = model.predict(df).tolist()
        # model.predict(df): predice TODAS las filas de una vez.
        # .tolist(): array numpy → lista Python.

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        # Tiempo total del batch en milisegundos.

        logger.info("Batch: %d predicciones en %.2f ms", len(preds), elapsed_ms)

        return BatchOutput(
            predicciones=[
                PropiedadOutput(precio_usd=round(p, 2))
                for p in preds
                # List comprehension: crea un PropiedadOutput por cada predicción.
            ],
            total=len(preds),
            procesado_en_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception("Error en predicción batch")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


# ── 10. MANEJADORES GLOBALES DE EXCEPCIONES ─────────────────────────────
# Capturan TODAS las excepciones (incluso las no previstas) y las
# convierten en respuestas JSON con formato RFC 7807.
# Sin estos handlers, una excepción no capturada devolvería un
# traceback HTML (el comportamiento por defecto de Starlette/FastAPI).

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Captura HTTPException (errores previstos como 422 o 503).

    Convierte la excepción en una respuesta JSON estructurada con:
      - status: código HTTP (422, 503, etc.)
      - detail: mensaje de error
      - instance: URL que generó el error (para debugging)
    """
    logger.warning(
        "HTTP %d en %s: %s",
        exc.status_code,      # 422, 500, 503...
        request.url.path,     # /predict, /predict/batch...
        exc.detail,           # mensaje de error
    )

    return JSONResponse(
        status_code=exc.status_code,
        # Mantiene el código HTTP de la excepción original.

        content=ErrorDetail(
            status=exc.status_code,
            title="Error de petición",
            detail=str(exc.detail),
            instance=str(request.url),
            # request.url: la URL completa que el cliente solicitó.
            # Ej: "http://localhost:8000/predict?foo=bar"
        ).model_dump(),
        # .model_dump(): convierte el modelo Pydantic a diccionario Python.
        # JSONResponse lo serializa a JSON.
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Captura CUALQUIER excepción no manejada (errores inesperados).

    Este es el "último recurso". Si algo falla y no fue capturado
    por un try/except ni por http_exception_handler, llega aquí.

    En producción, NUNCA deberías exponer el mensaje real de la excepción
    al cliente (podría filtrar información sensible como rutas del servidor,
    nombres de tablas, etc.). Por eso el detail es un mensaje genérico.

    El traceback completo se registra en logs para que el equipo de
    desarrollo pueda investigar.
    """
    logger.exception("Excepción no manejada en %s", request.url.path)
    # logger.exception() incluye el traceback completo en los logs.

    return JSONResponse(
        status_code=500,
        # Siempre 500 para errores inesperados.

        content=ErrorDetail(
            status=500,
            title="Error interno del servidor",
            detail="Ocurrió un error inesperado. Contacte al administrador.",
            instance=str(request.url),
        ).model_dump(),
    )


# ── 11. PUNTO DE ENTRADA ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # uvicorn: servidor ASGI de alto rendimiento.

    uvicorn.run(
        "api_v3_avanzada:app",
        # Se pasa como string "modulo:variable" en lugar del objeto app.
        # Esto permite que uvicorn haga reload correctamente en desarrollo.
        # "api_v3_avanzada" = nombre de este archivo (sin .py).
        # "app" = nombre de la variable FastAPI.

        host=settings.host,
        # Desde variable de entorno API_HOST (default "0.0.0.0").

        port=settings.port,
        # Desde variable de entorno API_PORT (default 8000).

        log_level=settings.log_level,
        # Desde variable de entorno LOG_LEVEL (default "info").

        reload=False,
        # reload=True recarga el servidor al detectar cambios en el código.
        # En producción DEBE ser False por seguridad y rendimiento.
        # En desarrollo se puede activar pasando --reload a uvicorn.
    )
