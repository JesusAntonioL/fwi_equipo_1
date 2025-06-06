import time
from ipyleaflet import TileLayer
import json
import folium
import datetime
import webbrowser
import leafmap
import ee
import math
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString
import sys

if len(sys.argv) != 4:
    print("Usage: python3 fwi-map2.py <lat> <long> <maxTemp>")
    sys.exit(1)

lat = float(sys.argv[1])
long = float(sys.argv[2])
maxTemp = float(sys.argv[3])

print(f"Received coordinates: Latitude={lat}, Longitude={long}, Temperature={maxTemp}")


# ============= FUNCIONES =============

# Cálculo del índice de riesgo de incendio forestal (FWI)
def risk_score(ndvi, slope, thermal):
    """
    Para sacar el peso individual de cada variable se sumaron sus IV's resultantes del análisis con el script de woe-data2.py
        IV de NDVI = 3.8987
        IV de Temperatura = 2.3786
        IV de Pendiente = 1.3073
        3.8987 + 2.3786 + 1.3073 = 7.5846
    De ahí, se sacó su peso relativo dividiendo cada IV entre el IV resultante
        ndviWeight = 3.8987/7.5846
        slopeWeight = 1.3073/7.5846
        thermalWeight = 2.3786/7.5846
    Teniendo así los siguientes pesos
    """
    # Pesos de las variables
    ndviWeight = 0.5142
    slopeWeight = 0.1722
    thermalWeight = 0.3136
    # Conversión a °C en caso de ser necesario
    thermal = (thermal*0.2) - 273.15
    score = (ndvi*ndviWeight) + (slope*slopeWeight) + (thermal*thermalWeight)
    return score


# Niveles de riesgo de incendio forestal
def visualize_risk(ndvi, slope, temperature):
    ndviWeight = 0.5142
    slopeWeight = 0.1722
    thermalWeight = 0.3136
    # Cálculo del índice de riesgo como imagen en EE
    risk_img = ndvi.multiply(ndviWeight).add(slope.multiply(slopeWeight)).add(
        temperature.multiply(thermalWeight)).rename("Riesgo")
    return risk_img


# Clasificación del riesgo de incendio forestal
def classify_risk(score):
    if score <= 1000:
        return "Muy Bajo"
    elif score <= 1800:
        return "Bajo"
    elif score <= 2200:
        return "Moderado"
    elif score <= 2600:
        return "Alto"
    elif score <= 3000:
        return "Muy Alto"
    else:
        return "Extremo"


# Añadir capa de Earth Engine a un mapa ipyleaflet
def add_ee_layer_ipyleaflet(self, ee_object, vis_params={}, name="Layer"):
    try:
        if isinstance(ee_object, ee.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            tile_layer = TileLayer(
                url=map_id_dict['tile_fetcher'].url_format,
                name=name,
                attribution="Google Earth Engine"
            )
            self.add_layer(tile_layer)
        else:
            print("El objeto debe ser ee.Image")
    except Exception as e:
        print(f"Error: {e}")


# Fechas de análisis
def date(delay):

    #Fechas Escamilla 
    start = '2025-04-17'
    end = '2025-05-17'

    #Fechas California
    #start = '2025-04-01'
    #end = '2025-04-30'

    # gap = 31
    # today = datetime.date.today()
    # end_str = (today - datetime.timedelta(days=delay)).isoformat()
    # start_str = (today - datetime.timedelta(days=delay+gap)).isoformat()
    # print(f"Fechas de análisis: {start_str} a {end_str}")
    # start = ee.Date(start_str)
    # end = ee.Date(end_str)
    return start, end


# Descarga de datos satelitales NDVI
def api_ndvi(start, end, area):
    return ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start, end).select(
        'NDVI').sort('system:time_start', False).mean().clip(area)


# Descarga de datos satelitales de temperatura
def api_temp(start, end, point, area):
    modis = ee.ImageCollection("MODIS/061/MOD11A1").filterBounds(
        point).filterDate(start, end).sort('system:time_start', False).first()
    return modis.select('LST_Day_1km').multiply(
        0.02).subtract(273.15).rename('Temperatura_C').clip(area)


# Descarga de datos satelitales de pendiente
def api_slope(area):
    dem = ee.Image('USGS/SRTMGL1_003')
    return ee.Terrain.slope(dem).rename('Pendiente_Angulo').clip(area)


# Descarga de datos satelitales del viento
def api_wind(start, end, area):
    wind_collection = ee.ImageCollection('NOAA/GFS0P25').filterDate(start, end).sort('system:time_start', False).select([
        'u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground'])
    meanU = wind_collection.select(
        'u_component_of_wind_10m_above_ground').mean()
    meanV = wind_collection.select(
        'v_component_of_wind_10m_above_ground').mean()
    return meanU.hypot(meanV).rename('WindSpeed_m/s').clip(area)


# Descarga de datos de viento
def wind_data(url, filename, tif):
    if not os.path.exists(tif):
        leafmap.download_file(url, output=filename, overwrite=True)
        # data = leafmap.read_netcdf(filename)
        # print(data)
        # Convierte el archivo NetCDF en TIFF para las variables de viento u y v
        leafmap.netcdf_to_tif(filename, tif, variables=[
                              "u_wind", "v_wind"], shift_lon=True)


#  ============= AUTENTICACIÓN E INICIALIZACIÓN =============
ee.Authenticate()
ee.Initialize(project='light-sunup-288723')

# # Coordenadas California
#lat = float(34.21113114902449)
#long = float(-118.1138591514406)

#Coodernadas Centro deportivo escamilla
# lat = float(25.65617721063847)
# long = float(-100.28706376985032)

# Ruta al archivo HTML
html_path = "G:/My Drive/expansion_sim/map6.html"

# Leer el HTML
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Reemplazar cualquier línea que contenga "center:"
import re
html_content = re.sub(r'center: \[.*?\],', f'center: [{long},{lat}],', html_content)

# Guardar los cambios
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Coordenadas actualizadas en el HTML.")

#Coodernadas Centro deportivo escamilla
# lat = float(25.65617721063847)
# long = float(-100.28706376985032)


# Creación de punto geométrico (área 10 km alrededor)
point = ee.Geometry.Point([long, lat])
area = point.buffer(10000)

# Fecha
start, end = date(delay=35)

# ============= IMPORTAR DATOS =============
# APIs
ndvi = api_ndvi(start, end, area)
temperature = api_temp(start, end, point, area)
wind_speed = api_wind(start, end, area)
slope = api_slope(area)

# Datos del viento
url = "https://github.com/opengeos/datasets/releases/download/raster/wind_global.nc"
filename = "wind_global.nc"
tif = "wind_global.tif"
wind_data(url, filename, tif)

# Imagen compuesta: Combina las bandas de NDVI, temperatura, velocidad del viento y pendiente
full_img = ndvi.rename('NDVI').addBands(
    temperature).addBands(wind_speed).addBands(slope)

# Muestreo de la imagen compuesta en el área definida
sampling = full_img.sample(
    region=area, scale=100, numPixels=100, geometries=True)
data = sampling.getInfo()


# ============= CÁLCULO DEL RIESGO DE INCENDIO FORESTAL =============
# Validación de datos
if not data['features']:
    print("No se encontraron datos para las coordenadas y fechas especificadas.")
    risk_map = None
else:
    risk_map = visualize_risk(ndvi, slope, temperature)
    for feature in data['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        valor_ndvi = props['NDVI']
        valor_wind = props['WindSpeed_m/s']
        valor_temp = props['Temperatura_C']
        valor_pendiente = props['Pendiente_Angulo']
        valor_pendiente_rad = math.radians(valor_pendiente)

        riesgo_estimado = risk_score(
            valor_ndvi, valor_pendiente_rad, valor_temp)
        nivel = classify_risk(riesgo_estimado)

        # print(f"Índice de riesgo: {riesgo_estimado:.2f} ({nivel})")
        # print(f"Coordenadas: {coords}")
        # print(f"NDVI: {valor_ndvi}, Temp: {valor_temp} °C, Viento: {valor_wind} m/s, Pendiente: {valor_pendiente}°")
        # print(f"Índice de riesgo: {riesgo_estimado:.2f}")

# ----------------------------------------------------------------------------------------
# Step 1: Create fine-grained bins from the risk raster
step = 100
min_risk = 900
max_risk = 3300

bins = ee.Image.constant(0)
for i in range(900, 3300, step):
    bin_mask = risk_map.gte(i).And(risk_map.lt(i + step))
    bins = bins.where(bin_mask, i)

# ⬇️ Agrega estas líneas aquí, después del bucle ⬇️
# Asegura que los valores por debajo de 900 y por encima de 3300 tengan un bin válido
bins = bins.where(risk_map.lt(min_risk), min_risk - step)  # Ej. 800
bins = bins.where(risk_map.gte(max_risk), max_risk)        # Ej. 3300


# Step 2: Convert the raster bins into small vector polygons
polys = bins.rename('risk_bin').reduceToVectors(
    geometry=area,
    scale=100,  # coarser scale for stability
    geometryType='polygon',
    labelProperty='risk_bin',
    maxPixels=1e13,
    bestEffort=True,
    eightConnected=False
)

# Step 3: Add a hex color to each polygon based on risk_bin
step = int((3300 - 900) / 10)  # 240
palette = ee.List([
    '#00ff00', '#66ff00', '#99ff00', '#ccff00',
    '#ffff00', '#ffcc00', '#ff9900', '#ff6600',
    '#ff3300', '#ff0000'
])


def add_color(feature):
    bin_val = ee.Number(feature.get('risk_bin'))
    index = bin_val.subtract(min_risk).divide(step).int()

    # Clamp index to be within valid range (0 to palette.length() - 1)
    index_clamped = index.min(palette.length().subtract(1))

    color = palette.get(index_clamped)
    return feature.set('color', color)


colored_polys = polys.map(add_color)

# Step 4: Export the result to GeoJSON on Google Drive
task = ee.batch.Export.table.toDrive(
    collection=colored_polys,
    description='riesgo_vector_heatmap_geojson',
    folder='expansion_sim',  # Or any Drive folder you prefer
    fileNamePrefix='riesgo_vector_heatmapxd',
    fileFormat='GeoJSON'
)
task.start()

print("Export task started... Monitoring status:")
while task.active():
    print("Status:", task.status()['state'])
    time.sleep(10)

print("Final Status:", task.status())


# ----------------------------------------------------------------------------------------


# ============= CREAR MAPA =============
heat_map = leafmap.Map(center=(lat, long), zoom=11)
# Extiende la clase leafmap.Map para añadir imágenes de EE
leafmap.Map.add_ee_layer = add_ee_layer_ipyleaflet

# Capa de velocidad del viento
# Añadir Basemap oscuro
heat_map.add_basemap("CartoDB.DarkMatter", name="Basemap Oscuro")
heat_map.add_velocity(
    filename,
    zonal_speed="u_wind",
    meridional_speed="v_wind",
    color_scale=[
        "rgb(0,0,150)",
        "rgb(0,150,0)",
        "rgb(255,255,0)",
        "rgb(255,165,0)",
        "rgb(150,0,0)",
    ],
    name="Velocidad del Viento",
)

# Capa del indice de riesgo de incendio
risk_map = ee.Image(risk_map)
if risk_map is not None:
    riesgo_vis = {
        'min': 900,
        'max': 3300,
        'palette': ['00ff00', '66ff00', '99ff00', 'ccff00', 'ffff00', 'ffcc00', 'ff9900', 'ff6600', 'ff3300', 'ff0000']
    }
    heat_map.add_ee_layer(
        risk_map, riesgo_vis, "Índice de Riesgo de Incendio")

# Añadir geometría del área como GeoJSON
geojson_str = json.dumps(area.getInfo())
heat_map.add_geojson(geojson_str, layer_name="Área Seleccionada")

# Añadir punto central
heat_map.add_marker(location=(lat, long),
                    icon_color="blue", name="Punto Central")


# ============= GUARDAR Y ABRIR EL MAPA EN HTML =============
html_file = 'fwi-heatmap.html'  # Nombre del archivo HTML
heat_map.to_html(html_file)

# HTML que se insertará colorbars
colorbar_html = """
<style>
html, body, #map {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0; 
}

.leaflet-container {
  height: 100% !important;
  width: 100% !important;
}

/* Spinner carga */
#loading-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255,255,255,0.8);
    z-index: 2000;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: Arial, sans-serif;
    font-size: 18px;
    color: #333;
}

.spinner {
    border: 6px solid #f3f3f3;
    border-top: 6px solid #3498db;
    border-radius: 50%;
    width: 40px; height: 40px;
    animation: spin 1s linear infinite;
    margin-right: 10px;
}

@keyframes spin {
  0% { transform: rotate(0deg);}
  100% { transform: rotate(360deg);}
}

/* Contenedor colorbars */
.colorbar-stack {
    position: absolute;
    bottom: 10px;
    right: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    z-index: 1500;
    font-family: Arial, sans-serif;
}

/* Individual colorbar */
.colorbar-container {
    width: 300px;
    padding: 4px;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid black;
    font-size: 11px;
}

/* Barra de colores */
.colorbar {
    width: 100%;
    height: 15px;
    margin-top: 4px;
}

/* Etiquetas */
.labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    margin-top: 2px;
}

.labels span {
  cursor: pointer;
  transition: color 0.3s, font-weight 0.3s;
}

.labels span:hover {
  color: #000;
  font-weight: bold;
  text-shadow: 0 0 3px #555;
}

/* Gradientes */
#wind-bar {
    background: linear-gradient(to right,
        rgb(0,0,150),
        rgb(0,150,0),
        rgb(255,255,0),
        rgb(255,165,0),
        rgb(150,0,0)
    );
}

#risk-bar {
    background: linear-gradient(to right,
        #00ff00,
        #66ff00,
        #99ff00,
        #ccff00,
        #ffff00,
        #ffcc00,
        #ff9900,
        #ff6600,
        #ff3300,
        #ff0000
    );
}
</style>

<!-- Spinner Carga -->
<div id="loading-overlay">
  <div class="spinner"></div>
  Cargando mapa...
</div>

<!-- Leyendas colorbars -->
<div class="colorbar-stack">
    <!-- Velocidad del viento -->
    <div class="colorbar-container">
        <strong>Velocidad del Viento (m/s)</strong>
        <div class="colorbar" id="wind-bar"></div>
        <div class="labels">
            <span title="0 m/s">0</span>
            <span title="5 m/s">5</span>
            <span title="10 m/s">10</span>
            <span title="15 m/s">15</span>
            <span title="20+ m/s">20+</span>
        </div>
    </div>

    <!-- Índice de riesgo -->
    <div class="colorbar-container">
        <strong>Índice de Riesgo de Incendio</strong>
        <div class="colorbar" id="risk-bar"></div>
        <div class="labels">
            <span title="Muy Bajo">Muy bajo</span>
            <span title="Bajo">Bajo</span>
            <span title="Moderado">Moderado</span>
            <span title="Alto">Alto</span>
            <span title="Muy Alto">Muy alto</span>
        </div>
    </div>
</div>

<script>
document.title = "FWI Heatmap";
// Ocultar spinner después que la página cargue 
window.addEventListener('load', function() {
  setTimeout(function() {
    const loading = document.getElementById('loading-overlay');
    if (loading) loading.style.display = 'none';
  }, 2000);
});
</script>
"""

# Añadir las colorbars al archivo HTML
with open(html_file, "r+", encoding="utf-8") as file:
    content = file.read()
    if "</body>" in content:
        content = content.replace("</body>", colorbar_html + "</body>")
    file.seek(0)
    file.write(content)

# Abrir el HTML en el navegador
webbrowser.open(html_file)


import http.server
import socketserver
import webbrowser
import threading
import time
import os

PORT = 8000
HTML_FILENAME = "map6.html"

# Abre el navegador después de una pausa
def open_browser():
    time.sleep(3)  # Espera 3 segundos
    url = f"http://localhost:{8000}/{'map6.html'}"
    webbrowser.open(url)

# Usa SimpleHTTPRequestHandler para servir archivos
handler = http.server.SimpleHTTPRequestHandler

# Asegúrate de estar en la carpeta correcta
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Lanza el navegador en un hilo separado con delay
threading.Thread(target=open_browser).start()

# Inicia el servidor
with socketserver.TCPServer(("", 8000), handler) as httpd:
    print(f"Servidor iniciado en http://localhost:{8000}")
    httpd.serve_forever()
