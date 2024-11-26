import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58865.streamlit.app/'


# EJERCICIO RESUELTO TP8

# Configuración inicial de la página
st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

# Función para calcular las estadísticas por producto
def calcular_estadisticas(df):
    estadisticas_producto = df.groupby('Producto').agg(
        Precio_Promedio=('Ingreso_total', lambda x: x.sum() / df['Unidades_vendidas'].sum()),
        Margen_Promedio=('Ingreso_total', lambda x: ((x.sum() - df['Costo_total'].sum()) / x.sum()) * 100),
        Unidades_Vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    return estadisticas_producto

# Cargar archivo CSV
st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:
    # Leer datos y eliminar espacios en blanco en nombres de columnas
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Eliminar espacios en blanco en los nombres de las columnas
    
    # Comprobar que las columnas "Año" y "Mes" existan en el DataFrame
    if 'Año' not in df.columns or 'Mes' not in df.columns:
        st.error("El archivo CSV debe contener las columnas 'Año' y 'Mes'.")
    else:
        # Filtrar por sucursal
        sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        # Mostrar el título dinámico según la sucursal seleccionada
        if sucursal_seleccionada == "Todas":
            st.title("Datos de Ventas Totales")
        else:
            st.title(f"Datos de Ventas en {sucursal_seleccionada}")

        # Filtrar datos según la sucursal seleccionada
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]

        # Calcular las estadísticas
        estadisticas = calcular_estadisticas(df)

        # Mostrar los productos y gráficos
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
                        # Renombrar columnas para que `pd.to_datetime` las reconozca
                        datos_producto = datos_producto.rename(columns={'Año': 'year', 'Mes': 'month'})
                        datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month"]].assign(day=1))
                        datos_producto = datos_producto.sort_values("Fecha")

                        # Crear gráfico
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=f"{row['Producto']}")
                        
                        # Calcular y graficar línea de tendencia
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