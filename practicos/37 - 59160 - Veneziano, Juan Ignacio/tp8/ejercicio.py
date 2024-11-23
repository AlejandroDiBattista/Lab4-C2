import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# link del deploy de el Trabajo Practico Numero 8: 
# URL = 'https://venezianojuantp8.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 59.160')
        st.markdown('**Nombre:** Juan Ignacio Veneziano')
        st.markdown('**Comisión:** C2')

def cargar_datos():
    archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
    if archivo_csv:
        contenido = pd.read_csv(archivo_csv)
        return contenido
    return None

def analizar_datos(datos_cargados, seleccion_sucursal):
    if seleccion_sucursal != "Todas":
        datos_cargados = datos_cargados[datos_cargados["Sucursal"] == seleccion_sucursal]

    lista_productos = datos_cargados["Producto"].unique()
    for item_producto in lista_productos:
        datos_filtro = datos_cargados[datos_cargados["Producto"] == item_producto]

        if datos_filtro["Ingreso_total"].isnull().any():
            st.error(f"El producto '{item_producto}' tiene valores nulos en la columna 'Ingreso_total'.")
            continue
        if datos_filtro["Unidades_vendidas"].isnull().any():
            st.error(f"El producto '{item_producto}' tiene valores nulos en la columna 'Unidades_vendidas'.")
            continue
        if (datos_filtro["Ingreso_total"] < 0).any():
            st.error(f"El producto '{item_producto}' tiene valores negativos en 'Ingreso_total'.")
            continue
        if (datos_filtro["Unidades_vendidas"] <= 0).any():
            st.error(f"El producto '{item_producto}' tiene valores no positivos en 'Unidades_vendidas'.")
            continue

        total_unidades = datos_filtro["Unidades_vendidas"].sum()
        total_ingresos = datos_filtro["Ingreso_total"].sum()
        total_costos = datos_filtro["Costo_total"].sum()

        precio_avg = total_ingresos / total_unidades
        margen_avg = (total_ingresos - total_costos) / total_ingresos * 100

        precio_promedio_global = datos_cargados["Ingreso_total"].sum() / datos_cargados["Unidades_vendidas"].sum()
        delta_precio = precio_avg - precio_promedio_global
        precio_avg_2024 = datos_filtro[datos_filtro["Año"] == 2024]["Ingreso_total"].sum() / datos_filtro[datos_filtro["Año"] == 2024]["Unidades_vendidas"].sum()
        precio_avg_2023 = datos_filtro[datos_filtro["Año"] == 2023]["Ingreso_total"].sum() / datos_filtro[datos_filtro["Año"] == 2023]["Unidades_vendidas"].sum()
        margen_avg_2024 = ((datos_filtro[datos_filtro["Año"] == 2024]["Ingreso_total"].sum() - datos_filtro[datos_filtro["Año"] == 2024]["Costo_total"].sum()) / datos_filtro[datos_filtro["Año"] == 2024]["Ingreso_total"].sum()) * 100
        margen_avg_2023 = ((datos_filtro[datos_filtro["Año"] == 2023]["Ingreso_total"].sum() - datos_filtro[datos_filtro["Año"] == 2023]["Costo_total"].sum()) / datos_filtro[datos_filtro["Año"] == 2023]["Ingreso_total"].sum()) * 100
        unidades_2024 = datos_filtro[datos_filtro["Año"] == 2024]["Unidades_vendidas"].sum()
        unidades_2023 = datos_filtro[datos_filtro["Año"] == 2023]["Unidades_vendidas"].sum()

        st.header(item_producto)
        st.metric("Precio Promedio", f"${precio_avg:,.2f}", delta=f"{((precio_avg_2024 / precio_avg_2023) - 1) * 100:.2f}%")
        st.metric("Margen Promedio", f"{margen_avg:.2f}%", delta=f"{((margen_avg_2024 / margen_avg_2023) - 1) * 100:.2f}%")
        st.metric("Unidades Vendidas", f"{total_unidades:,}", delta=f"{((unidades_2024 / unidades_2023) - 1) * 100:.2f}%")

        datos_filtro['Año'] = datos_filtro['Año'].astype(int)
        datos_filtro['Mes'] = datos_filtro['Mes'].astype(int)
        datos_filtro['Fecha'] = pd.to_datetime(
            datos_filtro['Año'].astype(str) + '-' + datos_filtro['Mes'].astype(str) + '-01'
        )

        datos_filtro.sort_values('Fecha', inplace=True)

        fig, ax = plt.subplots()
        ax.plot(datos_filtro["Fecha"], datos_filtro["Unidades_vendidas"], label=item_producto)

        x = np.arange(len(datos_filtro))
        y = datos_filtro["Unidades_vendidas"].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        tendencia = slope * x + intercept
        ax.plot(datos_filtro["Fecha"], tendencia, label="Tendencia", color="red")

        ax.set_title("Evolución de Ventas Mensual")
        ax.set_xlabel("Año-Mes")
        ax.set_ylabel("Unidades vendidas")
        ax.legend()
        st.pyplot(fig)

def inicio():
    st.sidebar.title("Cargar archivo de datos")
    mostrar_informacion_alumno()

    dataset = cargar_datos()
    if dataset is not None:
        lista_sucursales = ["Todas"] + dataset["Sucursal"].unique().tolist()
        sucursal_elegida = st.sidebar.selectbox("Seleccionar Sucursal", lista_sucursales)

        st.header(f"Datos de {'Todas las Sucursales' if sucursal_elegida == 'Todas' else sucursal_elegida}")
        analizar_datos(dataset, sucursal_elegida)
    else:
        st.write("Sube un archivo CSV desde la barra lateral izquierda.")

if __name__ == "__main__":
    inicio()
