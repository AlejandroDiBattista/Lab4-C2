import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# Evaluación: 5 | Recuperar para promocionar
# 1. Calcula mal precio y margenes promedio (-3)
# 2. No muestra las metricas a la par de los graficos (-1) 
# 3. No indica si esta mostrando todo o solo los datos de una sucursal (-1)


st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

## URL https://tp8-59277.streamlit.app/

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 59.277')
        st.markdown('**Nombre:** Emanuel Aguirre')
        st.markdown('**Comisión:** C2')

def calcular_estadisticas(df):
    estadisticas = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadisticas

st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener 'Año' y 'Mes'.")
    else:
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar una Sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        estadisticas = calcular_estadisticas(df)

        for _, row in estadisticas.iterrows():
            st.subheader(row["Producto"])
            st.metric("Precio Promedio",   f"${row['Precio_Promedio']:.2f}")
            st.metric("Margen Promedio",   f"{row['Margen_Promedio']:.2f}%")
            st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}")

            datos_producto = df[df["Producto"] == row["Producto"]]

            datos_producto = datos_producto.rename(columns={'Año': 'year', 'Mes': 'month'})
            datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month"]].assign(day=1))
            datos_producto = datos_producto.sort_values("Fecha")

            fig, ax = plt.subplots()
            ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label="Unidades vendidas")

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

def mostrar_info_util():
    st.markdown("### Consejos útiles:")
    st.markdown("Para un análisis más preciso, asegúrate de que los datos estén completos y correctamente formateados en el archivo CSV.")
    st.markdown("Puedes filtrar las ventas por sucursal para ver estadísticas específicas de cada una.")
    st.markdown("Si no ves las gráficas o los cálculos, revisa si las columnas 'Año', 'Mes', 'Producto' y 'Unidades_vendidas' están presentes en el archivo.")
    st.markdown("Recuerda que los valores de las columnas 'Ingreso_total' y 'Costo_total' deben ser numéricos para el cálculo correcto de las estadísticas.")

mostrar_informacion_alumno()
mostrar_info_util()
