# -*- coding: utf-8 -*-
"""
Streamlit v3 — Avanzada
═══════════════════════════════════════════════════════════════════════
Interfaz gráfica completa para la API v3 de predicción inmobiliaria.

¿Qué aporta sobre v2?
  1. Tres modos: predicción individual, batch (manual + CSV) e info del modelo.
  2. Batch prediction: evalúa múltiples propiedades de una sola vez.
  3. Subida de archivo CSV para predicción masiva.
  4. Descarga de resultados como CSV.
  5. Información completa del modelo desde /model-info.
  6. Sidebar con conexión, configuración y diagnóstico.
  7. Gráfico de barras con importancias.
  8. Métricas de rendimiento del batch (tiempo de procesamiento).

Cómo ejecutar:
  streamlit run streamlit_v3_avanzada.py

Requisitos:
  - La API v3 debe estar corriendo: python api_v3_avanzada.py
  - pip install streamlit requests pandas
"""

# ── 1. IMPORTACIONES ────────────────────────────────────────────────────
import streamlit as st
# Framework para aplicaciones web de datos.

import requests
# Cliente HTTP para consumir la API v3.

import pandas as pd
# pandas: para leer CSVs subidos por el usuario y mostrar tablas.

import io
# io: operaciones de entrada/salida en memoria.
#      io.BytesIO: convierte el archivo subido en un buffer leíble por pandas.
#      io.StringIO: convierte strings en buffers para pd.read_csv().

import time
# time: para medir tiempos de respuesta de la API.

from datetime import datetime
# datetime: timestamps para logs y nombres de archivo.


# ── 2. CONFIGURACIÓN DE LA PÁGINA ───────────────────────────────────────
st.set_page_config(
    page_title="Inmobiliaria — v3 Avanzada",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── 3. CONSTANTES ───────────────────────────────────────────────────────
# Coordenadas representativas por ciudad (centro aproximado).
COORDENADAS = {
    "Quito":     {"lat": -0.18, "lon": -78.48},
    "Guayaquil": {"lat": -2.19, "lon": -79.89},
    "Manta":     {"lat": -0.95, "lon": -80.73},
}

# Columnas que espera el modelo (deben coincidir con FEATURES en la API).
FEATURES = [
    "BEDROOMS", "BATHROOMS", "PARKING_SPOTS", "CONSTRUCTION_AREA_SQM",
    "LATITUDE", "LONGITUDE",
    "CITY_Guayaquil", "CITY_Manta", "CITY_Quito",
]

# Nombres legibles para las columnas en tablas.
FEATURE_LABELS = {
    "BEDROOMS": "Habitaciones",
    "BATHROOMS": "Baños",
    "PARKING_SPOTS": "Estacionamientos",
    "CONSTRUCTION_AREA_SQM": "Área (m²)",
    "LATITUDE": "Latitud",
    "LONGITUDE": "Longitud",
    "CITY_Guayaquil": "Guayaquil",
    "CITY_Manta": "Manta",
    "CITY_Quito": "Quito",
}


# ── 4. INICIALIZACIÓN DE ESTADO DE SESIÓN ───────────────────────────────
# Streamlit re-ejecuta el script en cada interacción. session_state
# persiste datos entre re-ejecuciones mientras dure la sesión del navegador.

if "model_info" not in st.session_state:
    st.session_state["model_info"] = None
    # Datos del endpoint /model-info (tipo de modelo, n_estimators, etc.).

if "features_list" not in st.session_state:
    st.session_state["features_list"] = None
    # Lista de features con importancias.

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None
    # Resultados de la última predicción batch (para descargar).

if "ultima_prediccion" not in st.session_state:
    st.session_state["ultima_prediccion"] = None
    # Última predicción individual.


# ── 5. FUNCIONES AUXILIARES ─────────────────────────────────────────────
# Definimos funciones reutilizables para evitar código duplicado.

def conectar_api(endpoint: str, method: str = "GET", json_data: dict = None, timeout: int = 15):
    """
    Llama a un endpoint de la API y maneja errores comunes.

    Args:
        endpoint:   ruta del endpoint (ej: "/health", "/predict").
        method:     método HTTP ("GET" o "POST").
        json_data:  datos a enviar en el body (solo para POST).
        timeout:    segundos máximos de espera.

    Returns:
        Tupla (success: bool, data: dict | str, status_code: int).
        - success=True  → data contiene el JSON de respuesta.
        - success=False → data contiene el mensaje de error.
    """
    url = f"{st.session_state.get('api_url', 'http://localhost:8000')}{endpoint}"
    # Obtenemos la URL de la sesión (configurable en sidebar).
    # .get() con default por si aún no se inicializó.

    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=json_data, timeout=timeout)

        if resp.status_code == 200:
            return True, resp.json(), 200
        else:
            # La API devolvió un error (422, 500, 503, etc.).
            try:
                error_data = resp.json()
            except Exception:
                error_data = {"detail": resp.text}
            return False, error_data, resp.status_code

    except requests.exceptions.ConnectionError:
        return False, "No se pudo conectar con la API. ¿Está corriendo?", 0
    except requests.exceptions.Timeout:
        return False, "La API tardó demasiado en responder (timeout).", 0
    except Exception as e:
        return False, f"Error inesperado: {str(e)}", 0


def build_propiedad_payload(bedrooms, bathrooms, parking, area, lat, lon, ciudad):
    """
    Construye el diccionario JSON que espera la API a partir de los widgets.

    Convierte el nombre de ciudad (string) a one-hot encoding (3 columnas 0/1).
    """
    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spots": parking,
        "area_m2": area,
        "lat": lat,
        "lon": lon,
        "city_guayaquil": 1 if ciudad == "Guayaquil" else 0,
        "city_manta": 1 if ciudad == "Manta" else 0,
        "city_quito": 1 if ciudad == "Quito" else 0,
    }


# ── 6. SIDEBAR — CONFIGURACIÓN Y DIAGNÓSTICO ───────────────────────────

with st.sidebar:
    st.header("⚙️ Configuración")

    # ── URL de la API ────────────────────────────────────────────────────
    api_url = st.text_input(
        "URL de la API v3",
        value="http://localhost:8000",
        help="Dirección del servidor FastAPI (api_v3_avanzada.py).",
    )
    api_url = api_url.rstrip("/")
    st.session_state["api_url"] = api_url
    # Guardamos en session_state para que las funciones auxiliares
    # puedan acceder sin pasarla como parámetro cada vez.

    st.divider()

    # ── Diagnóstico de conexión ──────────────────────────────────────────
    st.subheader("🔌 Diagnóstico")

    if st.button("🔍 Verificar conexión", use_container_width=True):
        # Llama al health check de la API para verificar conectividad.
        ok, data, code = conectar_api("/health")
        if ok:
            modelo_ok = data.get("modelo_cargado", False)
            if modelo_ok:
                st.success("✅ API conectada — Modelo cargado")
            else:
                st.warning("⚠️ API conectada pero modelo NO cargado")
            st.caption(f"Timestamp: {data.get('timestamp', 'N/A')}")
        else:
            st.error(f"❌ {data}")

    if st.button("📊 Cargar info del modelo", use_container_width=True):
        # Carga /model-info y /features en paralelo.
        with st.spinner("Cargando..."):
            ok1, data1, _ = conectar_api("/model-info")
            ok2, data2, _ = conectar_api("/features")

            if ok1:
                st.session_state["model_info"] = data1
                st.success("✅ Model info cargado")
            else:
                st.error(f"❌ /model-info: {data1}")

            if ok2:
                st.session_state["features_list"] = data2
                st.success("✅ Features cargadas")
            else:
                st.error(f"❌ /features: {data2}")

    st.divider()
    st.caption("Streamlit v3 — Avanzada")
    st.caption("API v3 — Batch")


# ── 7. PESTAÑAS PRINCIPALES ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔮 Predicción Individual",
    "📦 Predicción Batch",
    "ℹ️ Info del Modelo",
])


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("🔮 Estimar Precio de Propiedad")

    # ── CIUDAD Y COORDENADAS (FUERA DEL FORM) ──────────────────────────
    # Van fuera del st.form() para que al cambiar la ciudad, Streamlit
    # re-renderice automáticamente las coordenadas. Dentro de un form,
    # los widgets no reaccionan a cambios de otros widgets.

    st.subheader("🏙️ Ciudad")
    ciudad = st.radio(
        "Selecciona la ciudad:",
        options=["Quito", "Guayaquil", "Manta"],
        index=0,
        horizontal=True,
    )

    st.subheader("📍 Coordenadas")
    st.caption("Se precargan según la ciudad. Ajustalas si conocés la ubicación exacta.")
    coords = COORDENADAS[ciudad]
    c4, c5 = st.columns(2)
    with c4:
        latitude = st.number_input("Latitud", -90.0, 90.0, coords["lat"], step=0.0001, format="%.4f", key="lat_v3")
    with c5:
        longitude = st.number_input("Longitud", -180.0, 180.0, coords["lon"], step=0.0001, format="%.4f", key="lon_v3")

    # ── CARACTERÍSTICAS Y BOTÓN (DENTRO DEL FORM) ──────────────────────
    with st.form("form_individual"):
        st.subheader("📐 Características")
        c1, c2, c3 = st.columns(3)
        with c1:
            bedrooms = st.number_input("Habitaciones", 1, 20, 3)
        with c2:
            bathrooms = st.number_input("Baños", 1, 20, 2)
        with c3:
            parking_spots = st.number_input("Estacionamientos", 0, 20, 2)

        construction_area = st.number_input(
            "Área de construcción (m²)", 10.0, 10000.0, 200.0, 10.0, format="%.1f"
        )

        # Resumen y botón
        st.divider()
        st.caption(
            f"📍 {ciudad} | 🛏️ {bedrooms} hab | 🛁 {bathrooms} baños | "
            f"🚗 {parking_spots} park | 📐 {construction_area:.0f} m²"
        )
        enviar = st.form_submit_button("💰 Estimar precio", type="primary", use_container_width=True)

    # ── Procesar envío ──────────────────────────────────────────────────
    if enviar:
        payload = build_propiedad_payload(
            bedrooms, bathrooms, parking_spots,
            construction_area, latitude, longitude, ciudad,
        )

        with st.spinner("🔍 Analizando propiedad..."):
            ok, data, code = conectar_api("/predict", method="POST", json_data=payload)

        if ok:
            precio = data["precio_usd"]
            st.session_state["ultima_prediccion"] = {
                "precio": precio, "ciudad": ciudad, "payload": payload,
                "timestamp": datetime.now().isoformat(),
            }

            # Resultado
            st.success(f"## 💵 ${precio:,.2f} USD")
            st.caption(f"Modelo: {data.get('modelo', 'Random Forest')}")

            # Métricas rápidas
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Área", f"{construction_area:,.0f} m²")
            with m2:
                precio_m2 = precio / construction_area if construction_area > 0 else 0
                st.metric("Precio / m²", f"${precio_m2:,.0f}")
            with m3:
                st.metric("Ciudad", ciudad)
        else:
            st.error(f"❌ Error {code}: {data}")

    # ── Última predicción ────────────────────────────────────────────────
    if st.session_state["ultima_prediccion"] is not None:
        with st.expander("📋 Última predicción (JSON)", expanded=False):
            st.json(st.session_state["ultima_prediccion"])


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA 2: PREDICCIÓN BATCH
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("📦 Predicción Masiva (Batch)")

    st.markdown("""
    Valuá **múltiples propiedades** en una sola llamada a la API.
    Tenés dos opciones:
    1. **Modo manual**: agrea propiedades una por una con el formulario.
    2. **Subir CSV**: carga un archivo con los datos ya preparados.
    """)

    # ── Sub-pestañas para batch manual vs CSV ───────────────────────────
    batch_tab1, batch_tab2 = st.tabs(["✏️ Modo Manual", "📁 Subir CSV"])

    # ──────────────────────────────────────────────────────────────────────
    # BATCH MANUAL
    # ──────────────────────────────────────────────────────────────────────
    with batch_tab1:
        st.subheader("✏️ Agregar propiedades manualmente")

        # Inicializar lista de propiedades en session_state
        if "batch_propiedades" not in st.session_state:
            st.session_state["batch_propiedades"] = []
            # Lista de diccionarios, cada uno = una propiedad.

        # ── Formulario para agregar una propiedad ────────────────────────
        with st.form("form_batch_add"):
            st.caption("Ingresá los datos de una propiedad y agregala a la lista:")

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                b_bedrooms = st.number_input("Hab.", 1, 20, 3, key="b_bed")
            with bc2:
                b_bathrooms = st.number_input("Baños", 1, 20, 2, key="b_bath")
            with bc3:
                b_parking = st.number_input("Park.", 0, 20, 2, key="b_park")

            b_area = st.number_input("Área (m²)", 10.0, 10000.0, 200.0, 10.0, format="%.1f", key="b_area")

            b_ciudad = st.radio(
                "Ciudad",
                options=["Quito", "Guayaquil", "Manta"],
                index=0,
                horizontal=True,
                key="b_ciudad_radio",
            )
            b_coords = COORDENADAS[b_ciudad]
            bc4, bc5 = st.columns(2)
            with bc4:
                b_lat = st.number_input("Latitud", -90.0, 90.0, b_coords["lat"], step=0.0001, format="%.4f", key="b_lat")
            with bc5:
                b_lon = st.number_input("Longitud", -180.0, 180.0, b_coords["lon"], step=0.0001, format="%.4f", key="b_lon")

            agregar = st.form_submit_button("➕ Agregar a la lista", use_container_width=True)

        if agregar:
            # Construir la propiedad y añadirla a la lista.
            prop = build_propiedad_payload(
                b_bedrooms, b_bathrooms, b_parking,
                b_area, b_lat, b_lon, b_ciudad,
            )
            st.session_state["batch_propiedades"].append(prop)
            st.success(f"✅ Propiedad agregada ({b_ciudad}, {b_area:.0f} m²)")
            st.rerun()
            # st.rerun(): fuerza la re-ejecución del script para refrescar la tabla.

        # ── Mostrar lista actual ─────────────────────────────────────────
        if st.session_state["batch_propiedades"]:
            st.subheader(f"📋 Propiedades en la lista ({len(st.session_state['batch_propiedades'])})")

            # Convertir a DataFrame para mostrar tabla.
            df_batch = pd.DataFrame(st.session_state["batch_propiedades"])
            # Renombrar columnas a nombres legibles para la UI.
            df_display = df_batch.rename(columns=FEATURE_LABELS)
            # FEATURE_LABELS: mapea nombres técnicos a nombres en español.

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
            )

            # ── Botones de acción ────────────────────────────────────────
            col_acc1, col_acc2, col_acc3 = st.columns(3)
            with col_acc1:
                if st.button("🗑️ Limpiar lista", use_container_width=True):
                    st.session_state["batch_propiedades"] = []
                    st.session_state["batch_results"] = None
                    st.rerun()

            with col_acc2:
                if st.button("🚀 Ejecutar batch", type="primary", use_container_width=True):
                    t0 = time.perf_counter()
                    with st.spinner(f"Prediciendo {len(st.session_state['batch_propiedades'])} propiedades..."):
                        ok, data, code = conectar_api(
                            "/predict/batch",
                            method="POST",
                            json_data={"propiedades": st.session_state["batch_propiedades"]},
                        )
                    elapsed = (time.perf_counter() - t0) * 1000

                    if ok:
                        st.session_state["batch_results"] = data
                        st.success(
                            f"✅ {data['total']} predicciones en "
                            f"{data.get('procesado_en_ms', elapsed):.1f} ms"
                        )
                    else:
                        st.error(f"❌ Error {code}: {data}")

            # ── Mostrar resultados del batch ─────────────────────────────
            if st.session_state["batch_results"] is not None:
                st.subheader("📊 Resultados")

                resultados = st.session_state["batch_results"]
                # resultados = {"predicciones": [...], "total": N, "procesado_en_ms": M}

                # Construir tabla combinada: datos de entrada + predicción.
                filas = []
                for i, pred in enumerate(resultados["predicciones"]):
                    prop_original = st.session_state["batch_propiedades"][i]
                    filas.append({
                        "Ciudad": (
                            "Quito" if prop_original["city_quito"] == 1
                            else "Guayaquil" if prop_original["city_guayaquil"] == 1
                            else "Manta"
                        ),
                        "Área (m²)": prop_original["area_m2"],
                        "Hab.": prop_original["bedrooms"],
                        "Baños": prop_original["bathrooms"],
                        "Park.": prop_original["parking_spots"],
                        "Precio USD": pred["precio_usd"],
                    })
                df_resultados = pd.DataFrame(filas)

                st.dataframe(df_resultados, use_container_width=True, hide_index=True)

                # ── Botón de descarga CSV ────────────────────────────────
                csv_buffer = df_resultados.to_csv(index=False).encode("utf-8")
                # encode("utf-8"): convierte el string CSV a bytes (necesario para descarga).

                st.download_button(
                    label="📥 Descargar resultados (CSV)",
                    data=csv_buffer,
                    file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    # Nombre del archivo con timestamp para evitar sobrescrituras.

                    mime="text/csv",
                    # MIME type para archivos CSV.
                )

    # ──────────────────────────────────────────────────────────────────────
    # BATCH POR CSV
    # ──────────────────────────────────────────────────────────────────────
    with batch_tab2:
        st.subheader("📁 Subir archivo CSV")

        st.markdown("""
        El CSV debe tener **exactamente estas columnas** (con estos nombres):
        ```
        bedrooms,bathrooms,parking_spots,area_m2,lat,lon,city_guayaquil,city_manta,city_quito
        ```
        - `area_m2`, `lat`, `lon`: usan los alias de la API (no construction_area_sqm).
        - Las columnas de ciudad deben ser 0 o 1. **Exactamente una debe ser 1.**
        """)

        # ── Plantilla descargable ─────────────────────────────────────────
        plantilla = pd.DataFrame([{
            "bedrooms": 3, "bathrooms": 2, "parking_spots": 2,
            "area_m2": 200.0, "lat": -0.18, "lon": -78.48,
            "city_guayaquil": 0, "city_manta": 0, "city_quito": 1,
        }])
        csv_plantilla = plantilla.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Descargar plantilla CSV",
            data=csv_plantilla,
            file_name="plantilla_batch.csv",
            mime="text/csv",
        )

        # ── Subida de archivo ────────────────────────────────────────────
        archivo = st.file_uploader(
            "Selecciona tu archivo CSV:",
            type=["csv"],
            # type=["csv"]: solo permite archivos .csv.

            help="Columnas: bedrooms, bathrooms, parking_spots, area_m2, lat, lon, city_guayaquil, city_manta, city_quito",
        )

        if archivo is not None:
            try:
                # Leer CSV subido
                df_csv = pd.read_csv(archivo)
                # archivo es un UploadedFile de Streamlit; pandas lo lee directamente.

                # Validar columnas
                columnas_esperadas = [
                    "bedrooms", "bathrooms", "parking_spots",
                    "area_m2", "lat", "lon",
                    "city_guayaquil", "city_manta", "city_quito",
                ]
                columnas_faltantes = set(columnas_esperadas) - set(df_csv.columns)
                columnas_extra = set(df_csv.columns) - set(columnas_esperadas)

                if columnas_faltantes:
                    st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                elif columnas_extra:
                    st.warning(f"⚠️ Columnas extra (se ignorarán): {', '.join(columnas_extra)}")
                else:
                    st.success(f"✅ CSV válido: {len(df_csv)} propiedades")
                    st.dataframe(df_csv, use_container_width=True, hide_index=True)

                    # Validar que exactamente una ciudad = 1 por fila
                    suma_ciudades = (
                        df_csv["city_guayaquil"] +
                        df_csv["city_manta"] +
                        df_csv["city_quito"]
                    )
                    invalidas = (suma_ciudades != 1).sum()
                    if invalidas > 0:
                        st.error(
                            f"❌ {invalidas} filas tienen más o menos de una ciudad = 1. "
                            "Corregí el archivo y volvé a subirlo."
                        )
                    else:
                        # ── Ejecutar batch desde CSV ─────────────────────
                        if st.button("🚀 Ejecutar predicción batch", type="primary", use_container_width=True):
                            # Convertir DataFrame a lista de diccionarios
                            propiedades = df_csv[columnas_esperadas].to_dict(orient="records")
                            # orient="records": cada fila → un dict.

                            t0 = time.perf_counter()
                            with st.spinner(f"Prediciendo {len(propiedades)} propiedades..."):
                                ok, data, code = conectar_api(
                                    "/predict/batch",
                                    method="POST",
                                    json_data={"propiedades": propiedades},
                                )
                            elapsed = (time.perf_counter() - t0) * 1000

                            if ok:
                                st.session_state["batch_results"] = data

                                # Combinar entrada con predicciones
                                df_preds = pd.DataFrame(data["predicciones"])
                                df_combinado = pd.concat([
                                    df_csv[columnas_esperadas].reset_index(drop=True),
                                    df_preds.reset_index(drop=True),
                                ], axis=1)

                                st.success(
                                    f"✅ {data['total']} predicciones en "
                                    f"{data.get('procesado_en_ms', elapsed):.1f} ms"
                                )
                                st.dataframe(df_combinado, use_container_width=True, hide_index=True)

                                # Descargar
                                csv_out = df_combinado.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="📥 Descargar resultados (CSV)",
                                    data=csv_out,
                                    file_name=f"predicciones_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error(f"❌ Error {code}: {data}")

            except Exception as e:
                st.error(f"❌ Error al leer el CSV: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA 3: INFO DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("ℹ️ Información del Modelo")

    # ── Si no se cargó desde la sidebar, mostrar instrucción ────────────
    if st.session_state["model_info"] is None:
        st.info("ℹ️ Usa **📊 Cargar info del modelo** en la barra lateral para ver los detalles.")
    else:
        info = st.session_state["model_info"]

        # ── Tarjetas de resumen ──────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Algoritmo", info.get("tipo", "N/A"))
        with c2:
            st.metric("Árboles", info.get("n_estimators", "N/A"))
        with c3:
            st.metric("Features", info.get("n_features_in_", "N/A"))
        with c4:
            st.metric("Carga (ms)", info.get("tiempo_carga_ms", "N/A"))

        # ── Fecha de carga ──────────────────────────────────────────────
        st.caption(f"Modelo cargado: {info.get('cargado_en', 'N/A')}")

        st.divider()

        # ── Importancias ─────────────────────────────────────────────────
        st.subheader("📊 Importancia de Variables")

        if "importancias" in info:
            # Crear DataFrame ordenado por importancia descendente.
            df_imp = pd.DataFrame(info["importancias"])
            df_imp = df_imp.sort_values("importancia", ascending=True)
            # ascending=True: para que el gráfico de barras horizontales
            # muestre la más importante arriba (Streamlit invierte el eje Y).

            # ── Gráfico de barras ────────────────────────────────────────
            st.bar_chart(
                df_imp.set_index("feature")["importancia"],
                # set_index("feature"): el eje Y muestra nombres de variables.
                # ["importancia"]: la longitud de las barras.

                use_container_width=True,
                horizontal=True,
                # Barras horizontales: más legible con nombres largos de variables.

                x_label="Importancia relativa",
                y_label="Variable",
            )

            # ── Tabla detallada ──────────────────────────────────────────
            with st.expander("📋 Ver tabla de importancias", expanded=False):
                st.dataframe(
                    info["importancias"],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "feature": "Variable",
                        "importancia": st.column_config.ProgressColumn(
                            "Importancia",
                            format="%.4f",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                )

        st.divider()

        # ── Interpretación ───────────────────────────────────────────────
        st.subheader("💡 ¿Cómo interpretar esto?")
        st.markdown("""
        | Variable | Peso | Interpretación |
        |----------|------|----------------|
        | **CONSTRUCTION_AREA_SQM** | ~61% | El tamaño es el factor más determinante del precio. |
        | **LATITUDE + LONGITUDE** | ~28% | La ubicación exacta (barrio/zona) pesa más que la ciudad. |
        | **BATHROOMS** | ~6% | Los baños son más relevantes que las habitaciones para el modelo. |
        | **PARKING_SPOTS** | ~1.5% | Impacto bajo: probablemente correlacionado con el área. |
        | **BEDROOMS** | ~1.2% | Sorprendentemente bajo: el área ya captura el tamaño. |
        | **CITY_\*** | ~2% total | Las coordenadas ya capturan la ubicación mejor que la ciudad. |
        """)

        st.info(
            "💡 **Dato curioso:** El modelo aprendió solo con latitud y longitud "
            "que dentro de una misma ciudad hay zonas caras y baratas. "
            "Por eso las columnas de ciudad (`CITY_*`) tienen poco peso."
        )
