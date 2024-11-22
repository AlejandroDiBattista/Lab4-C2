import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ventas x Sucursal", layout="wide")

# Función para calcular estadísticas por producto 
def calcular_estadisticas(df):
    estadisticas = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadisticas

# Cargar archivo CSV con datos de ventas
st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:
    # Leer datos y eliminar espacios en blanco en nombres de columnas 
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  
    
    # Comprobar que las columnas "Año" y "Mes" existan en el DataFrame 
    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener 'Año' y 'Mes'.")
    else:
        # Filtrar por sucursal y calcular estadísticas
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar una Sucursal", sucursales)
        
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        # Calcular estadísticas por producto
        estadisticas = calcular_estadisticas(df)

        # Mostrar datos por producto en columnas
        for _, row in estadisticas.iterrows():
            st.subheader(row["Producto"])
            st.metric("Precio Promedio", f"${row['Precio_Promedio']:.2f}")
            st.metric("Margen Promedio", f"{row['Margen_Promedio']:.2f}%")
            st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}")

            # Graficar evolución de ventas por producto
            datos_producto = df[df["Producto"] == row["Producto"]]
            
            # Renombrar columnas para que pd.to_datetime las reconozca automáticamente
            datos_producto = datos_producto.rename(columns={'Año': 'year', 'Mes': 'month'})
            datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month"]].assign(day=1))
            datos_producto = datos_producto.sort_values("Fecha")

            # Crear gráfico
            fig, ax = plt.subplots()
            ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label="Unidades vendidas")
            
            # Calcular y graficar línea de tendencia (regresión lineal)
            z = np.polyfit(
                np.arange(len(datos_producto)),
                datos_producto["Unidades_vendidas"],
                1
            )
            p = np.poly1d(z)
            ax.plot(datos_producto["Fecha"], p(np.arange(len(datos_producto))), "r--", label="Tendencia")
            
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Unidades vendidas")
            ax.legend()
            st.pyplot(fig)

## Direccion en la que ha sido publicada la aplicacion
# URL = 'https://tp8-59130.streamlit.app/'

def mostrar_info_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.130')
        st.markdown('**Nombre:** Luciano Gatti')
        st.markdown('**Comisión:** C2')

mostrar_info_alumno()