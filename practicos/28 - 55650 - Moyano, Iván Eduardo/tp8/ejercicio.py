import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress

#Link de streamlit: https://tp8-lab-55650.streamlit.app/


st.set_page_config(layout="wide")

st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown("**Legajo:** 55650")
        st.markdown("**Nombre:** Moyano Ivan Eduardo")
        st.markdown("**Comisión:** C2")

mostrar_informacion_alumno()

archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_csv:
    sucursal = st.sidebar.selectbox(
        "Seleccionar Sucursal", 
        options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"], 
        index=0
    )
    
    datos = pd.read_csv(archivo_csv)
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]
        st.title(f"Datos de {sucursal}")
    else:
        st.title("Datos de Todas las Sucursales")
    

    for producto in datos["Producto"].unique():
        producto_datos = datos[datos["Producto"] == producto]
        

        resumen = producto_datos.groupby("Año").agg(
            {"Ingreso_total": "sum", "Costo_total": "sum", "Unidades_vendidas": "sum"}
        ).reset_index()
        resumen["Precio_promedio"] = resumen["Ingreso_total"] / resumen["Unidades_vendidas"]
        resumen["Margen_promedio"] = (resumen["Ingreso_total"] - resumen["Costo_total"]) / resumen["Ingreso_total"] * 100

        resumen["Cambio_Precio"] = resumen["Precio_promedio"].pct_change().mean() * 100
        resumen["Cambio_Margen"] = resumen["Margen_promedio"].pct_change() * 100
        resumen["Cambio_Unidades"] = resumen["Unidades_vendidas"].pct_change() * 100

        promedio_precio = resumen["Precio_promedio"].mean()
        promedio_margen = resumen["Margen_promedio"].mean()
        total_unidades = producto_datos["Unidades_vendidas"].sum()
        cambio_precio = resumen["Cambio_Precio"].mean()
        cambio_margen = resumen["Cambio_Margen"].mean()
        cambio_unidades = resumen["Cambio_Unidades"].mean()


        producto_datos = producto_datos.rename(columns={"Año": "Year", "Mes": "Month"})
        producto_datos["Fecha"] = pd.to_datetime(producto_datos[["Year", "Month"]].assign(Day=1))
        producto_datos = producto_datos.sort_values("Fecha")

        ventas_mes = producto_datos.groupby(["Year", "Month"])["Unidades_vendidas"].sum().reset_index()
        ventas_mes["Fecha"] = pd.to_datetime(ventas_mes[["Year", "Month"]].assign(Day=1))
        fechas_ordinales = ventas_mes["Fecha"].map(lambda x: x.toordinal())
        slope, intercept, *_ = linregress(fechas_ordinales, ventas_mes["Unidades_vendidas"])


        ventas_mes["Tendencia"] = slope * fechas_ordinales + intercept


        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ventas_mes["Fecha"], ventas_mes["Unidades_vendidas"], label=producto)
        ax.plot(ventas_mes["Fecha"], ventas_mes["Tendencia"], label="Tendencia", color="red", linestyle="--")
        ax.xaxis.set_minor_locator(AutoMinorLocator(15))
        ax.grid(which="major", alpha=1)
        ax.grid(which="minor", alpha=0.5)
        ax.set_xlabel("Año-Mes")
        ax.set_ylabel("Unidades Vendidas")
        ax.set_ylim(0, None)
        ax.legend(title="Producto")
        ax.set_title("Evolución de Ventas Mensual")

        with st.container(border=True):
            st.subheader(f"{producto}")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Precio Promedio", f"${promedio_precio:,.0f}".replace(",", "."), f"{cambio_precio:.2f}%")
                st.metric("Margen Promedio", f"{promedio_margen:.0f}%", f"{cambio_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{total_unidades:,.0f}".replace(",", "."), f"{cambio_unidades:.2f}%")
            with col2:
                st.pyplot(fig)
