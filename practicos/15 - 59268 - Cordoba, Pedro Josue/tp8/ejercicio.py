import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59268.streamlit.app/'

# Configuración inicial de la página
st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

#Calculo de estadistica del producto
def calcular_estadisticas(df):
    estadisticas_producto = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df.loc[x.index, 'Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df.loc[x.index, 'Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadisticas_producto

# Cargar archivo CSV
st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:
    #Lectura de datos
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Borrado de espacios en blanco
    
    # Verificamos si columnas "Año" y "Mes" existen en el DataFrame
    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Año' y 'Mes'.")
    else:
        # Filtrar por sucursal
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        # Mostrar el título dinámico según la sucursal seleccionada
        if sucursal_seleccionada == "Todas":
            st.title("Datos de Ventass Totales")
        else:
            st.title(f"Datos de Ventas en {sucursal_seleccionada}")

        # Filtrar datos según la sucursal seleccionada
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        # Calcular las estadísticas
        estadisticas = calcular_estadisticas(df)

        # Botón para descargar las estadísticas
        csv = estadisticas.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Descargar estadísticas como CSV",
            data=csv,
            file_name='estadisticas_producto.csv',
            mime='text/csv'
        )

        # Mostrar los productos y gráficos
        for _, row in estadisticas.iterrows():
            with st.container():
                # Columna 1: Mostrar información del producto
                with st.expander(f"Ver Gráfico de {row['Producto']}", expanded=True):
                    col1, col2 = st.columns([1, 3]) 
                    
                    with col1:
                        # Mostrar datos del producto
                        st.markdown(f"### {row['Producto']}")
                        st.metric("Precio Promedio", f"${row['Precio_Promedio']:.2f}")
                        st.metric("Margen Promedio", f"{row['Margen_Promedio']:.0f}%")
                        st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}")

                    with col2:
                        # Procesar los datos del producto
                        datos_producto = df[df["Producto"] == row["Producto"]]
                        datos_producto["Fecha"] = pd.to_datetime(
                            datos_producto["Año"].astype(str) + '-' + datos_producto["Mes"].astype(str) + '-01'
                        )
                        datos_producto = datos_producto.sort_values("Fecha")

                        # Crear gráfico
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=f"{row['Producto']}")
                        
                        # Calcular y graficar línea de tendencia
                        from matplotlib.dates import date2num

                        x_values = date2num(datos_producto["Fecha"])
                        z = np.polyfit(x_values, datos_producto["Unidades_vendidas"], 1)
                        p = np.poly1d(z)
                        ax.plot(datos_producto["Fecha"], p(x_values), "r--", label="Tendencia")
                        
                        ax.set_xlabel("Año-Mes")
                        ax.set_title(f"Evolución de Ventas Mensual", fontsize=16)
                        ax.set_ylabel("Unidades vendidas")
                        ax.legend()
                        plt.grid(True)
                        st.pyplot(fig)


def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59268')
        st.markdown('**Nombre:** Córdoba Pedro Josué')
        st.markdown('**Comisión:** 2')

mostrar_informacion_alumno()
