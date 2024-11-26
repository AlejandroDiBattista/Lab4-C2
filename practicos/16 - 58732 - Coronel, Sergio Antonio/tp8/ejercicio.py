import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def crear_grafico(df_producto, producto):
    ventas_mensuales = df_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ventas_mensuales.index, ventas_mensuales['Unidades_vendidas'], label=producto)

    x_vals = ventas_mensuales.index
    y_vals = ventas_mensuales['Unidades_vendidas']
    coeficientes = np.polyfit(x_vals, y_vals, 1)
    tendencia = np.poly1d(coeficientes)
    ax.plot(x_vals, tendencia(x_vals), linestyle='--', color='red', label='Tendencia')

    ax.set_title('Evolución Mensual de Ventas')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(ventas_mensuales.index)
    etiquetas = [f"{fila.Año}" if fila.Mes == 1 else "" for fila in ventas_mensuales.itertuples()]
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    return fig

def mostrar_metricas(df_producto):
    df_producto['Precio_Promedio'] = df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']
    promedio_precio = df_producto['Precio_Promedio'].mean()
    precio_anual = df_producto.groupby('Año')['Precio_Promedio'].mean()
    variacion_precio = precio_anual.pct_change().mean() * 100

    df_producto['Ganancia'] = df_producto['Ingreso_total'] - df_producto['Costo_total']
    df_producto['Margen'] = (df_producto['Ganancia'] / df_producto['Ingreso_total']) * 100
    promedio_margen = df_producto['Margen'].mean()
    margen_anual = df_producto.groupby('Año')['Margen'].mean()
    variacion_margen = margen_anual.pct_change().mean() * 100

    total_unidades = df_producto['Unidades_vendidas'].sum()
    unidades_anuales = df_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_unidades = unidades_anuales.pct_change().mean() * 100

    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.metric("Precio Promedio", f"${promedio_precio:,.0f}".replace(",", "."), f"{variacion_precio:.2f}%")
        st.metric("Margen Promedio", f"{promedio_margen:.0f}%".replace(",", "."), f"{variacion_margen:.2f}%")
        st.metric("Unidades Vendidas", f"{total_unidades:,.0f}".replace(",", "."), f"{variacion_unidades:.2f}%")
    with col2:
        fig = crear_grafico(df_producto, df_producto['Producto'].iloc[0])
        st.pyplot(fig)

st.sidebar.header("Cargar Datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    df = pd.read_csv(archivo_cargado)
    sucursales = ["Todas"] + df['Sucursal'].unique().tolist()
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]
        st.title(f"Datos de {sucursal}")
    else:
        st.title("Datos de Todas las Sucursales")

    productos = df['Producto'].unique()
    for producto in productos:
         with st.container(border=True):
            st.subheader(producto)
            df_producto = df[df['Producto'] == producto]
            mostrar_metricas(df_producto)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

# url = 'https://tp8-58732.streamlit.app/'

def mostrar_datos_alumno():
     with st.container(border=True):
        st.markdown('**Legajo:** 58.732')
        st.markdown('**Nombre:** Coronel Sergio Antonio')
        st.markdown('**Comisión:** C2')

mostrar_datos_alumno()