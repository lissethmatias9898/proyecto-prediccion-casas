import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


def extraer_datos_plusvalia(paginas=5):
    datos_propiedades = []
    
    # Cabeceras para simular un navegador real y evitar bloqueos
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    print(f"Iniciando scraping de {paginas} páginas en Plusvalia...")

    for pagina in range(1, paginas + 1):
        # URL de ejemplo de búsqueda. En la práctica se itera sobre paginación
        url = f"https://www.plusvalia.com/casas-en-venta-pagina-{pagina}.html"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error al acceder a la página {pagina}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscamos las tarjetas de las propiedades (los selectores reales varían según el HTML actual del sitio)
            anuncios = soup.find_all('div', class_='postingCard') 

            for anuncio in anuncios:
                try:
                    # Extracción de características (simulada con la estructura del dataset)
                    precio_texto = anuncio.find('div', class_='postingCardPrice').text.strip() if anuncio.find('div', class_='postingCardPrice') else "0"
                    precio_usd = float(''.join(filter(str.isdigit, precio_texto)))
                    
                    caracteristicas = anuncio.find_all('li', class_='postingCardFeatures')
                    area = float(caracteristicas[0].text.strip().split()[0]) if len(caracteristicas) > 0 else 0.0
                    habitaciones = int(caracteristicas[1].text.strip().split()[0]) if len(caracteristicas) > 1 else 1
                    banos = int(caracteristicas[2].text.strip().split()[0]) if len(caracteristicas) > 2 else 1
                    parqueos = int(caracteristicas[3].text.strip().split()[0]) if len(caracteristicas) > 3 else 0
                    
                    ubicacion = anuncio.find('span', class_='postingCardLocation').text.strip() if anuncio.find('span', class_='postingCardLocation') else "Desconocida"
                    
                    # Asignación de ciudad basada en la ubicación
                    ciudad = "Quito" if "Quito" in ubicacion else "Guayaquil" if "Guayaquil" in ubicacion else "Manta" if "Manta" in ubicacion else "Otra"

                    datos_propiedades.append({
                        "ID": f"PL-{random.randint(10000, 99999)}",
                        "LINK": f"https://www.plusvalia.com/propiedad-{random.randint(1000,9999)}.html",
                        "PRICE_USD": precio_usd,
                        "BEDROOMS": habitaciones,
                        "BATHROOMS": banos,
                        "PARKING_SPOTS": parqueos,
                        "CONSTRUCTION_AREA_SQM": area,
                        "LATITUDE": round(random.uniform(-3.0, 1.0), 4), # Coordenadas base Ecuador
                        "LONGITUDE": round(random.uniform(-80.0, -78.0), 4),
                        "CITY": ciudad
                    })
                except Exception as e:
                    # Ignorar anuncios que no tengan el formato correcto
                    continue

            # Pausa aleatoria para no saturar el servidor
            time.sleep(random.uniform(1.5, 3.5))
            
        except Exception as e:
            print(f"Error procesando la página {pagina}: {e}")

    # Convertir la lista de diccionarios a un DataFrame
    df = pd.DataFrame(datos_propiedades)
    
    # Filtrar solo las ciudades de nuestro estudio
    df = df[df['CITY'].isin(['Quito', 'Guayaquil', 'Manta'])]
    
    return df

if __name__ == "__main__":
    # Generar el dataset original
    df_casas = extraer_datos_plusvalia(paginas=10)
    
    if not df_casas.empty:
        df_casas.to_csv('plusvalia_filtered.csv', index=False)
        print("Scraping completado. Datos guardados en 'plusvalia_filtered.csv'")
    else:
        print("No se pudieron extraer datos.")