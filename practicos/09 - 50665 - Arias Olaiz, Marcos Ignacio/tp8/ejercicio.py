import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##URL: https://tp8-50665.streamlit.app/

st.set_page_config(page_title="Ventas por Producto", layout="wide")

# Función para cargar los datos
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Procesar datos por sucursal
def procesar_sucursal(datos, sucursal):
    if sucursal != "Todas":
        return datos[datos['Sucursal'] == sucursal]
    return datos

# Función para calcular métricas y deltas
def calcular_metricas(datos_producto):
    datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
    datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
    datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100

    precio_promedio = datos_producto['Precio_promedio'].mean()
    margen_promedio = datos_producto['Margen'].mean()
    unidades_totales = datos_producto['Unidades_vendidas'].sum()

    precio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
    margen_anual = datos_producto.groupby('Año')['Margen'].mean()
    unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()

    variacion_precio = precio_anual.pct_change().mean() * 100
    variacion_margen = margen_anual.pct_change().mean() * 100
    variacion_unidades = unidades_anuales.pct_change().mean() * 100

    return precio_promedio, variacion_precio, margen_promedio, variacion_margen, unidades_totales, variacion_unidades

# Función para crear gráfico
def crear_grafico_ventas(datos_producto, nombre_producto):
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    figura, eje = plt.subplots(figsize=(12, 6))
    eje_x = range(len(ventas_mensuales))
    eje.plot(eje_x, ventas_mensuales['Unidades_vendidas'], marker='o', linewidth=2, label=nombre_producto)

    coeficientes = np.polyfit(eje_x, ventas_mensuales['Unidades_vendidas'], 1)
    polinomio = np.poly1d(coeficientes)
    eje.plot(eje_x, polinomio(eje_x), linestyle='--', color='red', label='Tendencia')

    etiquetas = [f"{fila.Año}" if fila.Mes == 1 else "" for fila in ventas_mensuales.itertuples()]
    eje.set_xticks(eje_x)
    eje.set_xticklabels(etiquetas, rotation=45)

    eje.set_title('Evolución de Ventas Mensual', fontsize=16)
    eje.set_xlabel('Período', fontsize=14)
    eje.set_ylabel('Unidades Vendidas', fontsize=14)
    eje.legend(title='Producto')
    eje.grid(True)
    plt.tight_layout()

    return figura

# Mostrar información del estudiante
def mostrar_info_estudiante():
    st.write("""
    - **Número de Legajo:** 50665 
    - **Nombre:** Marcos Arias
    - **Comisión:** C2 
    """)

# Aplicación principal
def main():
    archivo_cargado = st.sidebar.file_uploader("Carga aquí un archivo CSV", type=["csv"])

    if archivo_cargado:
        datos = cargar_datos(archivo_cargado)
        sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Selecciona una Sucursal", sucursales)

        datos_filtrados = procesar_sucursal(datos, sucursal_seleccionada)
        st.header(f"Informe de Ventas - Sucursal: {sucursal_seleccionada}" if sucursal_seleccionada != "Todas" else "DATOS DE TODAS LAS SUCURSALES")

        productos = datos_filtrados['Producto'].drop_duplicates().values

        for producto in productos:
            st.subheader(f"Producto: {producto}")
            datos_producto = datos_filtrados[datos_filtrados['Producto'] == producto]

            # Calcular métricas
            costo_promedio, variacion_precio, margen_promedio, variacion_margen, ventas_totales, variacion_ventas = calcular_metricas(datos_producto)

            # Mostrar métricas
            col_izquierda, col_derecha = st.columns([0.25, 0.75])

            with col_izquierda:
                st.metric("Precio Promedio", f"${costo_promedio:,.0f}", f"{variacion_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.0f}%", f"{variacion_margen:.2f}%")
                st.metric("Total Unidades Vendidas", f"{ventas_totales:,.0f}", f"{variacion_ventas:.2f}%")

            # Mostrar gráfico
            with col_derecha:
                grafico = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(grafico)
    else:
        st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_info_estudiante()

if __name__ == "__main__":
    main()