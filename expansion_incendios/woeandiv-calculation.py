import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar archivo
df = pd.read_csv('datos_incendios_LA_con_riesgo.csv')

# Eliminar filas con valores faltantes en variables clave
df = df[['etiqueta', 'ndvi', 'temperatura', 'pendiente']].dropna()

# Convertir etiqueta a binaria por si no está en formato numérico
df['etiqueta'] = df['etiqueta'].astype(int)

# Función para calcular WoE e IV
def calcular_woe_iv(df, feature, target, bins=10):
    # Discretización en quantiles
    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')

    # Crear tabla de frecuencias
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'eventos']
    grouped['no_eventos'] = grouped['total'] - grouped['eventos']

    # Totales
    total_eventos = grouped['eventos'].sum()
    total_no_eventos = grouped['no_eventos'].sum()

    # Proporciones
    grouped['%eventos'] = grouped['eventos'] / total_eventos
    grouped['%no_eventos'] = grouped['no_eventos'] / total_no_eventos

    # Calcular WoE
    grouped['WoE'] = np.log((grouped['%eventos'] + 1e-10) / (grouped['%no_eventos'] + 1e-10))
    
    # Calcular IV
    grouped['IV'] = (grouped['%eventos'] - grouped['%no_eventos']) * grouped['WoE']
    iv_total = grouped['IV'].sum()

    return grouped[['WoE', 'IV']], iv_total

# Calcular WoE para cada variable
for var in ['ndvi', 'temperatura', 'pendiente']:
    print(f"\nWoE para {var}:")
    tabla_woe, iv = calcular_woe_iv(df.copy(), var, 'etiqueta', bins=10)
    print(tabla_woe)
    print(f"IV total para {var}: {iv:.4f}")
