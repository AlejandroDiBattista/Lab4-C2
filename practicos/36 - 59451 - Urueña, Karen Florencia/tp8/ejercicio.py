import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://modificacionparcial.streamlit.app/'
# Configuración de la página
st.set_page_config(page_title="Análisis de Ventas", layout="wide")

st.markdown('<h3>Por favor, sube un archivo CSV desde la barra lateral.</h3>', unsafe_allow_html=True)

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59451')
        st.markdown('**Nombre:** Karen Florencia Urueña')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

# Panel lateral
st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])
sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])

if uploaded_file: 
    # Cargar archivo CSV
    df = pd.read_csv(uploaded_file)

    # Validar columnas necesarias
    columnas_esperadas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
    if not all(col in df.columns for col in columnas_esperadas):
        st.error("El archivo cargado no contiene las columnas esperadas.")
    else:
        # Validar y convertir columnas Año y Mes a enteros
        try:
            df["Año"] = df["Año"].astype(int)
            df["Mes"] = df["Mes"].astype(int)
            # Crear columna de fechas manualmente, asignando el primer día de cada mes
            df["Fecha"] = pd.to_datetime(df["Año"].astype(str) + "-" + df["Mes"].astype(str) + "-01")
        except ValueError as e:
            st.error(f"Error al convertir 'Año' y 'Mes' en fechas: {e}")
            st.stop()

        # Filtrar por sucursal
        if sucursal_seleccionada != "Todas":
            df = df[df["Sucursal"] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
        
        # Calcular métricas para cada producto
        productos = df["Producto"].unique()
        for producto in productos:
            with st.container(border=True):
                df_producto = df[df["Producto"] == producto]
                
                # Calcular el precio promedio
                df_producto["Precio_promedio"] = df_producto["Ingreso_total"] / df_producto["Unidades_vendidas"]
                precio_promedio = df_producto["Precio_promedio"].mean()
                
                # Calcular el margen promedio
                df_producto["Ganancia"] = df_producto["Ingreso_total"] - df_producto["Costo_total"]
                df_producto["Margen"] = (df_producto["Ganancia"] / df_producto["Ingreso_total"]) * 100
                margen_promedio = df_producto["Margen"].mean()
                
                # Calcular unidades totales vendidas
                unidades_totales = df_producto["Unidades_vendidas"].sum()
                
                # Agrupar por año-mes y calcular métricas mensuales
                df_producto["Año-Mes"] = df_producto["Fecha"].dt.to_period("M")
                resumen_mensual = df_producto.groupby("Año-Mes").agg(
                    Unidades_vendidas=("Unidades_vendidas", "sum"),
                    Ingreso_total=("Ingreso_total", "sum"),
                    Costo_total=("Costo_total", "sum")
                ).reset_index()

                # Calcular variaciones porcentuales anuales
                metricas_anuales = df_producto.groupby('Año').agg({
                    'Precio_promedio': 'mean',
                    'Margen': 'mean',
                    'Unidades_vendidas': 'sum'
                })
                variacion_precio = metricas_anuales['Precio_promedio'].pct_change().mean() * 100
                variacion_margen = metricas_anuales['Margen'].pct_change().mean() * 100
                variacion_unidades = metricas_anuales['Unidades_vendidas'].pct_change().mean() * 100
                
                # Gráfico de evolución de ventas mensual
                ventas_mensuales = df_producto.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()
                
                # Crear diseño con columnas
                col_izquierda, col_derecha = st.columns([1, 3])
                
                # Mostrar métricas
                with col_izquierda:
                    st.subheader(f"{producto}")
                    st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{variacion_precio:.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{variacion_margen:.2f}%")
                    st.metric("Unidades Vendidas", f"{unidades_totales:,.0f}", f"{variacion_unidades:.2f}%")
                
                # Mostrar gráfico en la columna derecha
                with col_derecha:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(
                        ventas_mensuales["Fecha"],
                        ventas_mensuales["Unidades_vendidas"],
                        label=producto
                    )
                    
                    # Línea de tendencia
                    x_vals = np.arange(len(ventas_mensuales))
                    y_vals = ventas_mensuales["Unidades_vendidas"].values
                    coef = np.polyfit(x_vals, y_vals, 1)
                    tendencia = np.poly1d(coef)(x_vals)
                    ax.plot(
                        ventas_mensuales["Fecha"],
                        tendencia,
                        label="Tendencia",
                        linestyle="--",
                        color="red"
                    )
                    
                    ax.set_title("Evolución de Ventas Mensual")
                    ax.set_xlabel("Año-Mes")
                    ax.set_ylabel("Unidades Vendidas")
                    ax.legend()
                    ax.grid(True)
                    ax.set_ylim(0, None)  # Ajustar el eje Y para comenzar desde 0
                    st.pyplot(fig)
                
