import streamlit as st
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58943.streamlit.app/'

st.set_page_config(layout="wide")

st.title("Datos del alumno")
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58943')
        st.markdown('**Nombre:** Santiago Maza')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

st.subheader("Por favor, suba un archivo CSV desde la barra lateral izquierda")

archivo_csv = st.sidebar.file_uploader('Subir archivo CSV', type=['csv'])

if archivo_csv:
    sucursal_elegida = st.sidebar.selectbox('Seleccionar Sucursal', ('Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur'), index = 0)
    
    df = pd.read_csv(archivo_csv)

    if sucursal_elegida != "Todas":
        df = df[df["Sucursal"] == sucursal_elegida]
        st.title(f'Datos de {sucursal_elegida}')        
    else:
        st.title("Datos de Todas las Sucursales")        

    productos = df["Producto"].unique()

    for producto in productos:      
        datosProductos = df[df["Producto"] == producto]
        
        df_grouped = datosProductos.groupby("Año").agg({"Ingreso_total": 'sum', "Costo_total": 'sum', "Unidades_vendidas": 'sum'}).reset_index()
        df_grouped["Precio_promedio"] = df_grouped["Ingreso_total"] / df_grouped["Unidades_vendidas"]
        df_grouped["Margen_promedio"] = (df_grouped["Ingreso_total"] - df_grouped["Costo_total"]) / df_grouped["Ingreso_total"] * 100

        # Calcular los porcentajes de variación (ejemplo: respecto al mes anterior)
        df_grouped['Variacion_Precio_Promedio'] = df_grouped['Precio_promedio'].pct_change().mean() * 100
        df_grouped['Variacion_Margen_Promedio'] = df_grouped['Margen_promedio'].pct_change() * 100
        df_grouped['Variacion_Unidades_Vendidas'] = df_grouped['Unidades_vendidas'].pct_change() * 100

        # Rellenar los NaN (primer mes) con 0
        #df_grouped.fillna(0, inplace=True)

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
                st.metric("Precio Promedio",f"${precio_promedio:,.0f}".replace(",", "."), f"{variacion_precio_promedio:.2f}%")
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
                ax.set_ylim(0, None)
                ax.set_ylim(bottom=0)
                ax.set_title("Evolución de Ventas Mensual")
                st.pyplot(fig)