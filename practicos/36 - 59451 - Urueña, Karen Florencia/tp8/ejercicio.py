import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Configuración de la página
st.set_page_config(page_title="Análisis de Ventas", layout="wide")

st.markdown('<h3>Por favor, sube un archivo CSV desde la barra lateral.</h3>', unsafe_allow_html=True)

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59451')
        st.markdown('**Nombre:** Karen Florencia Urueña')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

# Título principal
st.title("Datos de Todas las Sucursales")

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

        # Agrupar datos por producto
        productos = df["Producto"].unique()
        for producto in productos:
            df_producto = df[df["Producto"] == producto]

            # Calcular métricas actuales
            precio_promedio = np.sum(df_producto["Ingreso_total"]) / np.sum(df_producto["Unidades_vendidas"])
            margen_promedio = np.mean(
                (df_producto["Ingreso_total"] - df_producto["Costo_total"]) / df_producto["Ingreso_total"]
            )
            unidades_totales = np.sum(df_producto["Unidades_vendidas"])

            # Calcular métricas de cambio (último mes vs penúltimo mes)
            df_producto["Año-Mes"] = df_producto["Fecha"].dt.to_period("M")
            resumen_mensual = df_producto.groupby("Año-Mes").agg(
                Unidades_vendidas=("Unidades_vendidas", "sum"),
                Ingreso_total=("Ingreso_total", "sum"),
                Costo_total=("Costo_total", "sum")
            ).reset_index()
            
            if len(resumen_mensual) > 1:
                # Variación porcentual entre los dos últimos periodos
                delta_precio = (
                    (resumen_mensual["Ingreso_total"].iloc[-1] / resumen_mensual["Unidades_vendidas"].iloc[-1])
                    - (resumen_mensual["Ingreso_total"].iloc[-2] / resumen_mensual["Unidades_vendidas"].iloc[-2])
                ) / (resumen_mensual["Ingreso_total"].iloc[-2] / resumen_mensual["Unidades_vendidas"].iloc[-2]) * 100
                
                delta_margen = (
                    (resumen_mensual["Ingreso_total"].iloc[-1] - resumen_mensual["Costo_total"].iloc[-1])
                    / resumen_mensual["Ingreso_total"].iloc[-1]
                    - (resumen_mensual["Ingreso_total"].iloc[-2] - resumen_mensual["Costo_total"].iloc[-2])
                    / resumen_mensual["Ingreso_total"].iloc[-2]
                ) * 100
                
                delta_unidades = (
                    (resumen_mensual["Unidades_vendidas"].iloc[-1] - resumen_mensual["Unidades_vendidas"].iloc[-2])
                    / resumen_mensual["Unidades_vendidas"].iloc[-2]
                ) * 100
            else:
                # No hay suficientes datos para calcular cambios
                delta_precio = delta_margen = delta_unidades = 0

            # Evolución de ventas mensuales
            ventas_mensuales = df_producto.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()

            # Crear diseño con columnas
            col_izquierda, col_derecha = st.columns([1, 3])  # Proporción de espacio: 1 para métricas, 3 para gráfico

            # Mostrar métricas en la columna izquierda
            with col_izquierda:
                st.subheader(f" {producto}")
                st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{delta_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio * 100:.2f}%", f"{delta_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_totales:,.0f}", f"{delta_unidades:.2f}%")

            # Mostrar gráfico en la columna derecha
            with col_derecha:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(
                    ventas_mensuales["Fecha"],
                    ventas_mensuales["Unidades_vendidas"],
                    label=producto,
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
                    color="red",
                )

                # Configurar gráfico
                ax.set_title("Evolución de Ventas Mensual")
                ax.set_xlabel("Año-Mes")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

