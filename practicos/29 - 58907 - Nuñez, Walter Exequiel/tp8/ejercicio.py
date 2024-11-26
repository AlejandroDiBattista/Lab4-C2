import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import numpy as np
from io import StringIO

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58907-hfgro4jrzbyjfb3cuptuj4.streamlit.app/'

st.title("Datos del Alumno")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58907')
        st.markdown('**Nombre:** Nuñez Walter Exequiel')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

# Título de la aplicación
st.subheader("Por favor, sube un archivo CSV desde la barra lateral izquierda")


# Subir archivo CSV (en la barra lateral)
uploaded_files = st.sidebar.file_uploader("Sube un archivo CSV con los datos de ventas", type=["csv"])

if uploaded_files:
    sucursal = st.sidebar.selectbox("Seleccionar sucursal", ('Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur'), index = 0)
    
    data = pd.read_csv(uploaded_files)
    
    if sucursal != "Todas":
        data = data[data['Sucursal'] == sucursal]

    # Título dinámico basado en la sucursal seleccionada
    if sucursal == "Todas":
        st.header("Información Detallada de Todas las Sucursales")
    else:
        st.title(f"Datos de {sucursal}")
        
    productos = data["Producto"].unique()

    for producto in productos:      
        datosProductos = data[data["Producto"] == producto]
        
        df_grouped = datosProductos.groupby(["Año", "Mes"]).agg({"Ingreso_total": 'sum', "Costo_total": 'sum', "Unidades_vendidas": 'sum'}).reset_index()
        df_grouped["Precio_promedio"] = df_grouped["Ingreso_total"] / df_grouped["Unidades_vendidas"]
        df_grouped["Margen_promedio"] = (df_grouped["Ingreso_total"] - df_grouped["Costo_total"]) / df_grouped["Ingreso_total"] * 100

        # Calcular los porcentajes de variación (ejemplo: respecto al mes anterior)
        df_grouped['Variacion_Precio_Promedio'] = df_grouped['Precio_promedio'].pct_change() * 100
        df_grouped['Variacion_Margen_Promedio'] = df_grouped['Margen_promedio'].pct_change() * 1000
        df_grouped['Variacion_Unidades_Vendidas'] = df_grouped['Unidades_vendidas'].pct_change() * 1000

        # Rellenar los NaN (primer mes) con 0
        df_grouped.fillna(0, inplace=True)

        precio_promedio = df_grouped["Precio_promedio"].mean()
        margen_promedio = df_grouped["Margen_promedio"]
        unidades_vendidas = datosProductos["Unidades_vendidas"].sum()

        variacion_precio_promedio = df_grouped["Variacion_Precio_Promedio"].mean()
        variacion_margen_promedio = df_grouped["Variacion_Margen_Promedio"].mean()
        variacion_unidades_vendidas = df_grouped["Variacion_Unidades_Vendidas"].mean()

        datosProductos.rename(columns = {"Año": "Year", "Mes": "Month"}, inplace = True)
        datosProductos["Fecha"] = pd.to_datetime(datosProductos[["Year", "Month"]].assign(Day=1))
        datosProductos = datosProductos.sort_values('Fecha')

        ventas_por_mes = datosProductos.groupby(["Year", "Month"])["Unidades_vendidas"].sum().reset_index()
        ventas_por_mes["Fecha"] = pd.to_datetime(ventas_por_mes[["Year", "Month"]].assign(Day=1))
        
        fechas_ordinales = ventas_por_mes["Fecha"].map(lambda x: x.toordinal())  # Convertir fechas a números ordinales
        slope, intercept, r_value, p_value, std_err = linregress(fechas_ordinales, ventas_por_mes["Unidades_vendidas"])
    
        # Línea de tendencia
        ventas_por_mes["Tendencia"] = slope * fechas_ordinales + intercept

        fig, ax = plt.subplots()

        with st.container(border=True):
            st.subheader(f"{producto}")

            col1, col2 = st.columns([1,3])

            with col1:
                st.metric("Precio Promedio",f"${precio_promedio:,.2f}".replace(",", "."), f"{variacion_precio_promedio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio.mean():.0f}%", f"{variacion_margen_promedio:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",", "."), f"{variacion_unidades_vendidas:.2f}%")

            with col2:
                fig.set_size_inches(10,5)
                ax.plot(ventas_por_mes['Fecha'], ventas_por_mes["Unidades_vendidas"],label=producto)
                ax.plot(ventas_por_mes['Fecha'], ventas_por_mes["Tendencia"], label='Tendencia', color='red', linestyle = '--' )
                
                ax.xaxis.set_minor_locator(AutoMinorLocator(15))

                ax.grid(which = "major", alpha = 1)
                ax.grid(which = "minor", alpha = 0.5)

                ax.set_xlabel("Año-Mes")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend(title="Producto")
                ax.set_ylim(bottom=0)
                ax.set_title("Evolución de Ventas Mensual")
                st.pyplot(fig)