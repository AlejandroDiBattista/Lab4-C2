## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59176.streamlit.app/'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Ventas", layout="wide")

def mostrar_informacion_alumno():
    st.sidebar.markdown("### Datos del Alumno")
    st.sidebar.markdown("**Legajo:** 59.176")
    st.sidebar.markdown("**Nombre:** Argañraz Facundo Nahuel")
    st.sidebar.markdown("**Comisión:** C2")

def cargar_datos():
    st.sidebar.markdown("### Cargar Archivo CSV")
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def calcular_metricas(df):
    metricas = df.groupby('Producto').agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum',
        'Costo_total': 'sum'
    }).reset_index()
    
    # Calcular precio promedio y margen promedio
    metricas['Precio_promedio'] = metricas['Ingreso_total'] / metricas['Unidades_vendidas']
    metricas['Margen_promedio'] = (metricas['Ingreso_total'] - metricas['Costo_total']) / metricas['Ingreso_total']
    
    return metricas

def graficar_evolucion_ventas(df):
    # Crear fecha combinando año y mes
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
    ventas_mensuales = df.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], marker='o')
    
    # Añadir línea de tendencia
    x = np.arange(len(ventas_mensuales))
    z = np.polyfit(x, ventas_mensuales['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(ventas_mensuales['Fecha'], p(x), "r--", alpha=0.8, label='Tendencia')
    
    plt.title('Evolución de Ventas por Mes')
    plt.xlabel('Fecha')
    plt.ylabel('Unidades Vendidas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def main():
    st.title("Análisis de Ventas")
    
    # Sidebar con las opciones
    mostrar_informacion_alumno()
    
    # Cargar datos desde el sidebar
    df = cargar_datos()
    
    if df is not None:
        # Selector de sucursal en el sidebar
        st.sidebar.markdown("### Seleccionar Sucursal")
        sucursales = ["Todas"] + list(df['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
        
        # Filtrar datos por sucursal si es necesario
        df_filtrado = df if sucursal_seleccionada == "Todas" else df[df['Sucursal'] == sucursal_seleccionada]
        
        # Mostrar métricas
        metricas = calcular_metricas(df_filtrado)
        st.subheader(f"Métricas por Producto - Sucursal: {sucursal_seleccionada}")
        st.dataframe(metricas.style.format({
            'Precio_promedio': '${:.2f}',
            'Margen_promedio': '{:.2%}',
            'Unidades_vendidas': '{:,.0f}',
            'Ingreso_total': '${:,.2f}',
            'Costo_total': '${:,.2f}'
        }))
        
        # Mostrar gráfico
        st.subheader("Evolución de Ventas")
        fig = graficar_evolucion_ventas(df_filtrado)
        st.pyplot(fig)
    else:
        st.warning("Por favor, carga un archivo CSV para analizar los datos.")

if __name__ == "__main__":
    main()
