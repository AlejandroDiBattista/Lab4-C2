import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# url: https://tp-58909.streamlit.app/

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.909')
        st.markdown('**Nombre:** Argañaraz Leonardo Ramiro')
        st.markdown('**Comisión:** C2')

st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

def calcular_estadisticas(df):
    estadisticas_producto = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadisticas_producto

st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
mostrar_informacion_alumno()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip() 

    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Año' y 'Mes'.")
    else:
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada == "Todas":
            st.title("Datos de Ventas Totales")
        else:
            st.title(f"Datos de Ventas en {sucursal_seleccionada}")

        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        estadisticas = calcular_estadisticas(df)

        for _, row in estadisticas.iterrows():
            with st.container():

                container_style = """
                <style>
                .resize-container {
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #f4f4f4;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    resize: both;
                    overflow: auto;
                    width: 80%;
                    min-width: 600px;
                    height: 600px;
                }
                .resize-handle {
                    cursor: ew-resize;
                    width: 10px;
                    height: 100%;
                    position: absolute;
                    right: 0;
                    top: 0;
                }
                </style>
                """
                st.markdown(container_style, unsafe_allow_html=True)

                with st.expander(f"Ver Gráfico de {row['Producto']}", expanded=True):

                    col1, col2 = st.columns([1, 3]) 

                    with col1:                       
                        st.markdown(f"### {row['Producto']}")
                        st.metric("Precio Promedio", f"${row['Precio_Promedio']:.2f}")
                        st.metric("Margen Promedio", f"{row['Margen_Promedio']:.0f}%")
                        st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}")

                    with col2:  
                        datos_producto = df[df["Producto"] == row["Producto"]]                       
                        datos_producto = datos_producto.rename(columns={'Año': 'year', 'Mes': 'month'})
                        datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month"]].assign(day=1))
                        datos_producto = datos_producto.sort_values("Fecha")

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=f"{row['Producto']}")

                        z = np.polyfit(
                            np.arange(len(datos_producto)),
                            datos_producto["Unidades_vendidas"],
                            1
                        )
                        p = np.poly1d(z)
                        ax.plot(datos_producto["Fecha"], p(np.arange(len(datos_producto))), "r--", label="Tendencia")

                        ax.set_xlabel("Año-Mes")
                        ax.set_title(f"Evolución de Ventas Mensual", fontsize=16)
                        ax.set_ylabel("Unidades vendidas")
                        ax.legend()
                        plt.grid(True)
                        st.pyplot(fig)