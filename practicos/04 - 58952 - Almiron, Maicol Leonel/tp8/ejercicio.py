import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = "https://tp8-almiron-58952.streamlit.app/"

# Configuración de la página
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

# Función para mostrar información del usuario
def mostrar_informacion_usuario():
    """
    Muestra información básica del usuario si no hay datos cargados.
    """
    with st.container():
        st.subheader("Información del Alumno")
        st.text("Legajo: 58952")
        st.text("Nombre: Maicol Leonel Almirón")
        st.text("Comisión: C2")

# Función para calcular métricas
def calcular_metricas(df_producto):
    """
    Calcula las métricas y variaciones requeridas para un producto.
    """
    df_producto['Precio_Unitario'] = df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']
    promedio_precio = df_producto['Precio_Unitario'].mean()
    variacion_precio = df_producto.groupby('Año')['Precio_Unitario'].mean().pct_change().mean() * 100

    df_producto['Margen'] = ((df_producto['Ingreso_total'] - df_producto['Costo_total']) / df_producto['Ingreso_total']) * 100
    promedio_margen = df_producto['Margen'].mean()
    variacion_margen = df_producto.groupby('Año')['Margen'].mean().pct_change().mean() * 100

    total_unidades = df_producto['Unidades_vendidas'].sum()
    variacion_unidades = df_producto.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100

    return promedio_precio, variacion_precio, promedio_margen, variacion_margen, total_unidades, variacion_unidades

# Función para crear gráfico
def crear_grafico(df_producto):
    """
    Genera un gráfico de evolución de ventas por mes para un producto.
    """
    ventas_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], marker='o', color='blue', label='Unidades Vendidas')

    if len(ventas_mensuales) > 1:
        z = np.polyfit(range(len(ventas_mensuales)), ventas_mensuales['Unidades_vendidas'], 1)
        p = np.poly1d(z)
        ax.plot(ventas_mensuales['Fecha'], p(range(len(ventas_mensuales))), "--", color="red", label="Tendencia")

    ax.set_title("Evolución de Ventas")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()
    ax.grid(True)

    return fig

# Función para mostrar métricas y gráficos
def mostrar_metricas(df_producto):
    """
    Muestra métricas y gráfico asociados a un producto.
    """
    promedio_precio, variacion_precio, promedio_margen, variacion_margen, total_unidades, variacion_unidades = calcular_metricas(df_producto)

    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.metric("Precio Promedio", f"${promedio_precio:,.2f}", f"{variacion_precio:.2f}%")
        st.metric("Margen Promedio", f"{promedio_margen:.2f}%", f"{variacion_margen:.2f}%")
        st.metric("Unidades Vendidas", f"{total_unidades:,.0f}", f"{variacion_unidades:.2f}%")
    with col2:
        fig = crear_grafico(df_producto)
        st.pyplot(fig)

# Área de carga de datos
st.sidebar.header("Carga de Datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado:
    try:
        ventas_df = pd.read_csv(archivo_cargado)

        # Verificar columnas necesarias
        columnas_requeridas = ['Año', 'Mes', 'Sucursal', 'Producto', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
        if not all(col in ventas_df.columns for col in columnas_requeridas):
            st.error("El archivo CSV no contiene las columnas necesarias.")
            st.stop()

        # Procesar datos
        ventas_df.fillna(0, inplace=True)
        ventas_df['Fecha'] = pd.to_datetime(
            ventas_df[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1)
        )

        # Selección de sucursal
        sucursales = ["Todas"] + ventas_df['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Selecciona una sucursal", sucursales)
        if sucursal_seleccionada != "Todas":
            ventas_df = ventas_df[ventas_df['Sucursal'] == sucursal_seleccionada]

        # Mostrar métricas y gráficos por producto
        productos = ventas_df['Producto'].unique()
        for producto in productos:
            with st.container():
                st.subheader(f"Producto: {producto}")
                df_producto = ventas_df[ventas_df['Producto'] == producto]
                mostrar_metricas(df_producto)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    mostrar_informacion_usuario()

