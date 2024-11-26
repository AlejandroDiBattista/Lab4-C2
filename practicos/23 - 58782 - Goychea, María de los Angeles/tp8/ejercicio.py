import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://trabajopractico.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58782')
        st.markdown('**Nombre:** Maria de los Angeles Goyechea')
        st.markdown('**Comisión:** C2')


st.set_page_config(page_title="Datos de Ventas", layout="wide")
st.title("Datos de Ventas")

mostrar_informacion_alumno()

st.markdown("### Por favor, sube un archivo CSV desde la barra lateral.")
st.sidebar.header("Cargar archivo de datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])

if archivo:
    datos = pd.read_csv(archivo)

    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]
        st.title(f"Datos de {sucursal}")
    else:
        st.title(f"Datos de Todas las sucursales")    
    
    productos = datos["Producto"].unique()

    for producto in productos:
        with st.container(border=True):
        
            productoActual = datos[datos["Producto"] == producto]            
            margen_Promedio = np.mean((productoActual["Ingreso_total"] - productoActual["Costo_total"]) / productoActual["Ingreso_total"])
            
            
            resumen_anual = productoActual.groupby("Año").agg(
                Unidades_vendidas=("Unidades_vendidas", "sum"),
                Ingreso_total=("Ingreso_total", "sum"),
                Costo_total=("Costo_total", "sum")
            ).reset_index()

            resumen_anual["Precio_promedio"] = resumen_anual["Ingreso_total"] / resumen_anual["Unidades_vendidas"]
            resumen_anual["Margen_Promedio"] = (resumen_anual["Ingreso_total"] - resumen_anual["Costo_total"]) / resumen_anual["Ingreso_total"]
            resumen_anual["Unidades_Promedio"] = resumen_anual["Unidades_vendidas"]           
            if len(resumen_anual) > 1:              
                
                variacion_precio = resumen_anual["Precio_promedio"].diff().iloc[1:] / resumen_anual["Precio_promedio"].iloc[:-1].values
                variacion_precio_Promedio = variacion_precio.mean() * 100

                variacion_margen = resumen_anual["Margen_Promedio"].diff().iloc[1:] / resumen_anual["Margen_Promedio"].iloc[:-1].values
                variacion_margen_Promedio = variacion_margen.mean() * 100

                variacion_unidades = resumen_anual["Unidades_Promedio"].diff().iloc[1:] / resumen_anual["Unidades_Promedio"].iloc[:-1].values
                variacion_unidades_vendidas_Promedio = variacion_unidades.mean() *100
            else:                
                variacion_precio_Promedio = variacion_margen_Promedio = variacion_unidades_vendidas_Promedio = 0

            metricaA , metricaB = st.columns([1, 3])

            with metricaA :
                st.subheader(f" {producto}")
                st.metric("Precio Promedio", f"${resumen_anual["Precio_promedio"].mean():,.0f}", f"{variacion_precio_Promedio:.2f}%")
                st.metric("Margen Promedio", f"{margen_Promedio*100:.0f} %", f"{variacion_margen_Promedio:.2f}%")
                st.metric("Unidades Vendidas", f"{resumen_anual["Unidades_Promedio"].sum():,}", f"{variacion_unidades_vendidas_Promedio:.2f}%")
            
            ventas_mensuales = productoActual.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
            x = np.arange(len(ventas_mensuales))
            y = ventas_mensuales['Unidades_vendidas']

            fig, ax = plt.subplots(figsize=(12, 7))

            ax.plot(x, y, label=producto)

            Coeficiente = np.polyfit(x, y, 1)
            p = np.poly1d(Coeficiente)
            ax.plot(x, p(x), linestyle='--', color='red',label='Tendencia')
            ax.set_xticks(x)   
            ax.set_xticklabels([f"{row.Año}" if row.Mes == 1 else "" for row in ventas_mensuales.itertuples()])
            ax.set_ylim(0, None)   
            ax.set_title("Evolución de Ventas")            
            ax.set_ylabel("Unidades Vendidas")
            ax.set_xlabel("Año-Mes")
            ax.grid(linestyle='--')
            ax.legend(loc='best', frameon=True,facecolor='white', edgecolor='lightgray')

            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
            
            with metricaB:                
                st.pyplot(fig)
