import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mostrar información del alumno
def mostrar_informacion_alumno():
    st.sidebar.header("Información del Alumno")
    st.sidebar.markdown("**Legajo:** 50.665")
    st.sidebar.markdown("**Nombre:** Marcos Arias")
    st.sidebar.markdown("**Comisión:** C2")

# Cargar y procesar los datos
@st.cache_data
def cargar_datos(archivo):
    try:
        datos = pd.read_csv(archivo)
        return datos
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def calcular_estadisticas(datos):
    # Agrupar por producto y calcular las métricas
    resumen = datos.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: (x / datos.loc[x.index, "Unidades_vendidas"]).mean()),
        Margen_Promedio=("Ingreso_total", lambda x: ((x - datos.loc[x.index, "Costo_total"]) / x).mean()),
        Unidades_Vendidas=("Unidades_vendidas", "sum")
    ).reset_index()
    return resumen

def graficar_evolucion(datos, sucursal=None):
    if sucursal:
        datos = datos[datos["Sucursal"] == sucursal]

    datos["Fecha"] = pd.to_datetime(datos["Año"].astype(str) + "-" + datos["Mes"].astype(str))
    ventas_mensuales = datos.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(ventas_mensuales["Fecha"], ventas_mensuales["Unidades_vendidas"], label="Ventas Mensuales")
    z = np.polyfit(range(len(ventas_mensuales)), ventas_mensuales["Unidades_vendidas"], 1)
    p = np.poly1d(z)
    plt.plot(ventas_mensuales["Fecha"], p(range(len(ventas_mensuales))), "--", label="Tendencia", color="orange")
    plt.xlabel("Fecha")
    plt.ylabel("Unidades Vendidas")
    plt.title("Evolución de Ventas")
    plt.legend()
    st.pyplot(plt)

# Interfaz principal
def main():
    st.title("Aplicación de Análisis de Ventas")
    st.markdown("**Trabajo Práctico 8 - 2do Parcial**")
    st.write("Esta aplicación permite cargar y analizar datos de ventas, mostrando estadísticas y gráficos.")

    mostrar_informacion_alumno()

    archivo = st.file_uploader("Sube un archivo CSV con los datos de ventas", type=["csv"])

    if archivo:
        datos = cargar_datos(archivo)
        if datos is not None:
            st.write("Vista previa de los datos:", datos.head())

            sucursales = ["Todas"] + list(datos["Sucursal"].unique())
            sucursal_seleccionada = st.selectbox("Selecciona una sucursal:", sucursales)

            if sucursal_seleccionada != "Todas":
                datos = datos[datos["Sucursal"] == sucursal_seleccionada]

            resumen = calcular_estadisticas(datos)
            st.subheader("Estadísticas por Producto")
            st.dataframe(resumen)

            st.subheader("Gráfico de Evolución de Ventas")
            graficar_evolucion(datos, sucursal=None if sucursal_seleccionada == "Todas" else sucursal_seleccionada)

## Direccion en la que ha sido publicada la aplicacion
# URL = 'https://tp8-50665.streamlit.app/'

if __name__ == "__main__":
    main()
