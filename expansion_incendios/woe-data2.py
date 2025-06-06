import ee
import random
from datetime import timedelta

# Inicializar Earth Engine
ee.Initialize(project='light-sunup-288723')

# Definir el área de estudio: Condado de Los Ángeles
la_county = ee.FeatureCollection('TIGER/2018/Counties') \
    .filter(ee.Filter.eq('NAME', 'Los Angeles'))

# Definir el rango de fechas
start_date = ee.Date('2020-01-01')
end_date = ee.Date('2020-12-31')

# Cargar colección de incendios MODIS y extraer centroides de quemas
fires = ee.ImageCollection('MODIS/061/MCD64A1') \
    .filterDate(start_date, end_date) \
    .select('BurnDate') \
    .map(lambda img: img.mask(img)) \
    .map(lambda img: img.reduceToVectors(scale=500, geometryType='centroid', geometry=la_county.geometry(), labelProperty='BurnDate')) \
    .flatten()

# Extraer puntos de fuego
def extract_fire_point(feature):
    centroid = feature.geometry().centroid()
    return ee.Feature(centroid, {
        'fecha': '2020-03-01',
        'etiqueta': 1  # Incendio
    })

fire_points = fires.map(extract_fire_point)

# Generar puntos aleatorios para no-incendios
non_fire_points = ee.FeatureCollection.randomPoints(**{
    'region': la_county.geometry(),
    'points': 100,
    'seed': 42
})

def random_date_feature(feature):
    rand = ee.Number.parse(feature.id()).multiply(0.12345).sin().abs()
    rand_days = rand.multiply(end_date.difference(start_date, 'day'))
    random_date = start_date.advance(rand_days, 'day')
    return feature.set({
        'fecha': random_date.format('YYYY-MM-dd'),
        'etiqueta': 0  # No-incendio
    })

non_fire_points = non_fire_points.map(random_date_feature)

# Unir ambas colecciones
sample_points = fire_points.merge(non_fire_points)

# Agregar variables NDVI, LST, Elevación y Pendiente
def add_variables(feature):
    date = ee.Date(feature.get('fecha'))
    point = feature.geometry()

    is_valid_geom = point.bounds().coordinates().size().gt(0)

    # === NDVI ===
    ndvi_img = ee.ImageCollection('MODIS/061/MOD13Q1') \
        .filterDate(date.advance(-16, 'day'), date) \
        .filterBounds(point) \
        .first()

    ndvi = ee.Algorithms.If(
        is_valid_geom,
        ee.Algorithms.If(
            ndvi_img,
            ndvi_img.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=250
            ).get('NDVI'),
            None
        ),
        None
    )

    # === LST ===
    lst_img = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(date.advance(-5, 'day'), date) \
        .filterBounds(point) \
        .first()

    lst = ee.Algorithms.If(
        is_valid_geom,
        ee.Algorithms.If(
            lst_img,
            lst_img.select('LST_Day_1km').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).get('LST_Day_1km'),
            None
        ),
        None
    )

    # === Elevación ===
    elev = ee.Algorithms.If(
        is_valid_geom,
        ee.Image('USGS/SRTMGL1_003').select('elevation') \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('elevation'),
        None
    )

    # === Pendiente ===
    slope = ee.Algorithms.If(
        is_valid_geom,
        ee.Terrain.products(ee.Image('USGS/SRTMGL1_003')) \
            .select('slope') \
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('slope'),
        None
    )

    return feature.set({
        'ndvi': ndvi,
        'temperatura': lst,
        'elevacion': elev,
        'pendiente': slope
    })

enriched_points = sample_points.map(add_variables)

# Calcular índice de riesgo
def add_risk_index(feature):
    ndvi = feature.get('ndvi')
    lst = feature.get('temperatura')
    elev = feature.get('elevacion')
    slope = feature.get('pendiente')

    # Verifica si cada variable NO es None y convierte a Number (1 para válido, 0 para inválido)
    ndvi_valid = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(ndvi, None), 0, 1))
    lst_valid = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(lst, None), 0, 1))
    elev_valid = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(elev, None), 0, 1))
    slope_valid = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(slope, None), 0, 1))

    # Solo si todos son válidos (1), el índice de riesgo se calcula
    all_valid = ndvi_valid.multiply(lst_valid).multiply(elev_valid).multiply(slope_valid)

    lst_celsius = ee.Number(lst).multiply(0.02).subtract(273.15)

    risk = ee.Algorithms.If(
        all_valid.eq(1),
        ee.Number(lst_celsius).multiply(0.4)
            .add(ee.Number(slope).multiply(0.3))
            .subtract(ee.Number(ndvi).multiply(0.2))
            .add(ee.Number(elev).multiply(0.1)),
        None
    )

    return feature.set('riesgo', risk)
    
final_points = enriched_points.map(add_risk_index)

# Exportar
task = ee.batch.Export.table.toDrive(
    collection=final_points,
    description='datos_incendios_LA_con_riesgo',
    fileFormat='CSV'
)
task.start()
print("Exportación iniciada. Verifica el estado en la pestaña 'Tasks' de Earth Engine.")
