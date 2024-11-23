import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

# Función para calcular estadísticas por producto
def calcular_estadistica(df):
    estadistica = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadistica

# Cargar archivo CSV
st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:
    # Leer datos y eliminar espacios en blanco en nombres de columnas
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Eliminar espacios en blanco en los nombres de las columnas
    
    # Comprobar que las columnas "Año" y "Mes" existan en el DataFrame
    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Año' y 'Mes'.")
    else:
        # Filtrar por sucursal
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
        
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        # Calcular estadísticas
        estadisticas = calcular_estadistica(df)

        # Mostrar datos por producto
        for _, row in estadisticas.iterrows():
            st.subheader(row["Producto"])
            st.metric("Precio Promedio", f"${row['Precio_Promedio']:.2f}")
            st.metric("Margen Promedio", f"{row['Margen_Promedio']:.2f}%")
            st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}")

            # Graficar evolución de ventas
            datos_producto = df[df["Producto"] == row["Producto"]]
            
            # Renombrar columnas para que pd.to_datetime las reconozca
            datos_producto = datos_producto.rename(columns={'Año': 'year', 'Mes': 'month'})
            datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month"]].assign(day=1))
            datos_producto = datos_producto.sort_values("Fecha")

            # Crear gráfico
            fig, ax = plt.subplots()
            ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label="Unidades vendidas")
            
            # Calcular y graficar línea de tendencia
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

        ## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://roferesper-proyecto-laboratorioiv-ejercicio-ugi9ud.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('*Legajo:* 59.423')
        st.markdown('*Nombre:* Esper Rodrigo Fernando')
        st.markdown('*Comisión:* C2')

mostrar_informacion_alumno()