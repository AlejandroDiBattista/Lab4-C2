import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

# Función para calcular estadísticas por producto
def calculo_Estadisticas(df):
    # Calcular ingreso total
    estadisticas = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum') 
    ).reset_index()
    return estadisticas

# con esto realizo para Cargar archivo CSV
st.sidebar.title("Cargar archivo de datos")
# verifico si el archivo es csv
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
# Verifico si se subió un archivo
if uploaded_file:
    # Leeo el  archivo CSV
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Eliminar espacios en blanco en los nombres de las columnas
    
    # Comprobar que las columnas "Año" y "Mes" existan en el DataFrame
    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Año' y 'Mes'.")
    else:
        # Filtro por sucursal
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
        
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        # Calcular estadísticas
        estadisticas = calculo_Estadisticas(df)

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
# url = 'https://tp8-lab-59234.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.234')
        st.markdown('**Nombre:** Matias Sebastian Chocobar')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()
