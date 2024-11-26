import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58777-tut9bi2u9wrbzbhbmkidbx.streamlit.app'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.777')
        st.markdown('**Nombre:** Alan Jassán')
        st.markdown('**Comisión:** C2')

st.sidebar.title("Carga de Archivo")
archivo_cargado = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")

if archivo_cargado is None:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container():
        mostrar_informacion_alumno()
else:
    df = pd.read_csv(archivo_cargado)
    
    if 'Sucursal' in df.columns:
        sucursales = ["Todas las Sucursales"] + df['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Selecciona una sucursal", sucursales, key="sucursal_seleccion", disabled=False)
        
        if sucursal_seleccionada:
            st.title(f"Datos cargados para {sucursal_seleccionada}")
    else:
        st.error("El archivo no contiene la columna 'Sucursal'. Verifica el formato del archivo.")
        
    productos = df['Producto'].unique()

    for producto in productos:      
        producto_data = df[df["Producto"] == producto]
        
        historical_data = producto_data.groupby(["Año", "Mes"]).agg({"Ingreso_total": 'sum', "Costo_total": 'sum', "Unidades_vendidas": 'sum'}).reset_index()
        historical_data["Precio_promedio"] = historical_data["Ingreso_total"] / historical_data["Unidades_vendidas"]
        historical_data["Margen_promedio"] = (historical_data["Ingreso_total"] - historical_data["Costo_total"]) / historical_data["Ingreso_total"] * 100

        historical_data['Historical_Precio_Promedio'] = historical_data['Precio_promedio'].pct_change() * 100
        historical_data['Historical_Margen_Promedio'] = historical_data['Margen_promedio'].pct_change() * 1000
        historical_data['Historical_Unidades_Vendidas'] = historical_data['Unidades_vendidas'].pct_change() * 1000

        precio_promedio = historical_data["Precio_promedio"].mean()
        margen_promedio = historical_data["Margen_promedio"]
        unidades_vendidas = producto_data["Unidades_vendidas"].sum()

        historical_precio_promedio = historical_data["Historical_Precio_Promedio"].mean()
        historical_margen_promedio = historical_data["Historical_Margen_Promedio"].mean()
        historical_unidades_vendidas = historical_data["Historical_Unidades_Vendidas"].mean()

        producto_data.rename(columns = {"Año": "Year", "Mes": "Month"}, inplace = True)
        producto_data["Fecha"] = pd.to_datetime(producto_data[["Year", "Month"]].assign(Day=1))
        producto_data = producto_data.sort_values('Fecha')
        
        historical_mensual = producto_data.groupby(["Year", "Month"])["Unidades_vendidas"].sum().reset_index()
        historical_mensual["Fecha"] = pd.to_datetime(historical_mensual[["Year", "Month"]].assign(Day=1))
        
        fechas_ordinales = historical_mensual["Fecha"].map(lambda x: x.toordinal())  # Convertir fechas a números ordinales
        slope, intercept, r_value, p_value, std_err = linregress(fechas_ordinales, historical_mensual["Unidades_vendidas"])

        # Línea de tendencia
        historical_mensual["Tendencia"] = slope * fechas_ordinales + intercept

        fig, ax = plt.subplots()
        
        with st.container(border=True):
            col1, col2 = st.columns([1,3])

            with col1:
                st.markdown(f"### {producto}")
                st.metric("Precio Promedio",f"${precio_promedio:,.2f}".replace(",", "."), f"{historical_precio_promedio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio.mean():.0f}%", f"{historical_margen_promedio:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",", "."), f"{historical_unidades_vendidas:.2f}%")

            with col2:
                fig.set_size_inches(10,5)

                ax.plot(historical_mensual['Fecha'], historical_mensual["Unidades_vendidas"],label=producto)
                ax.plot(historical_mensual['Fecha'], historical_mensual["Tendencia"], label='Tendencia', color='red', linestyle = '--' )
                
                ax.xaxis.set_minor_locator(AutoMinorLocator(15))

                ax.grid(which = "major", alpha = 1)
                ax.grid(which = "minor", alpha = 0.5)

                ax.set_xlabel("Año-Mes")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend(title="Producto")
                ax.set_ylim(bottom=0)
                ax.set_title("Evolución de Ventas Mensual")
                st.pyplot(fig)