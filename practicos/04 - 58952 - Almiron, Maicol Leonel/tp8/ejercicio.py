import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Evaluación: 6 | Recuperar para promocionar 
# 1. Muestra los datos en tabla que no se pide (-1)
# 2. No usa el sidebar correctamente (carga datos y seleccionar sucursal) (-1)
# 3. No muestra las métricas ni los gráficos por productos (-1)
# 4. Usa un estilo personalizado en lugar del pedido (sacar css)
# 5. No muestra línea de tendencia (-1)


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = "https://tp8-almiron.streamlit.app/"

# Configuración básica de la app
st.set_page_config(
    page_title="Análisis de Ventas",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
        .main {background-color: #f7f7f7;}
        h1 {color: #2C3E50; text-align: center;}
        h2 {color: #2980B9;}
        .sidebar .sidebar-content {background-color: #D6EAF8;}
        .block-container {padding: 1rem 3rem;}
        table {margin: 0 auto;}
        .stButton>button {background-color: #2980B9; color: white; font-size: 14px;}
        .stMarkdown {color: #2C3E50;}
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("📊 Análisis de Ventas")
st.markdown("""
    Bienvenido a la aplicación de análisis de ventas. Aquí podrás cargar un archivo CSV con los datos 
    y explorar métricas clave, gráficas y análisis por sucursal.
""")

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.sidebar:
        st.markdown("### 🎓 Información del Alumno", unsafe_allow_html=True)
        st.markdown("- **Legajo:** 58952")
        st.markdown("- **Nombre:** Maicol Leonel Almirón")
        st.markdown("- **Comisión:** C2")
        st.markdown("---")
     

mostrar_informacion_alumno()

# Cargar datos
uploaded_file = st.file_uploader("📂 Cargue el archivo CSV", type=["csv"])

def calcular_resumen(df):
    """
    Calcula el resumen por producto con las métricas requeridas.
    """
    resumen = df.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: (x / np.where(df.loc[x.index, "Unidades_vendidas"] > 0, 
                                                                 df.loc[x.index, "Unidades_vendidas"], 1)).mean()),
        Margen_Promedio=("Ingreso_total", lambda x: ((x - df.loc[x.index, "Costo_total"]) / np.where(x > 0, x, 1)).mean()),
        Unidades_Vendidas=("Unidades_vendidas", "sum")
    ).reset_index()
    return resumen

def graficar_evolucion_ventas(df):
    """
    Genera un gráfico de evolución de ventas por mes.
    """
    ventas_mensuales = df.groupby("Fecha")['Unidades_vendidas'].sum().reset_index()
    ventas_mensuales = ventas_mensuales.set_index("Fecha").asfreq("MS", fill_value=0).reset_index()
    fig = px.line(
        ventas_mensuales, 
        x="Fecha", 
        y="Unidades_vendidas", 
        title="📈 Evolución de Ventas por Mes",
        labels={"Unidades_vendidas": "Unidades Vendidas", "Fecha": "Fecha"},
        template="plotly_white"
    )
    return fig

if uploaded_file:
    try:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Datos cargados:")
        st.dataframe(df, use_container_width=True)

        # Verificar que las columnas necesarias existan
        columnas_requeridas = ['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
        if not all(col in df.columns for col in columnas_requeridas):
            st.error(f"El archivo CSV debe contener las columnas: {', '.join(columnas_requeridas)}")
        else:
            # Verificar y convertir "Año" y "Mes" a numérico
            df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
            df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')

            # Detectar valores nulos en "Año" y "Mes"
            if df[['Año', 'Mes']].isnull().any().any():
                st.warning("Algunas filas contienen valores nulos o no válidos en las columnas 'Año' o 'Mes'. Estas filas serán eliminadas.")
                df.dropna(subset=['Año', 'Mes'], inplace=True)

            # Crear la columna "Fecha"
            df['Fecha'] = pd.to_datetime({
                'year': df['Año'].astype(int),
                'month': df['Mes'].astype(int),
                'day': 1
            }, errors='coerce')

            if df['Fecha'].isnull().all():
                st.error("No se pudieron generar fechas válidas a partir de 'Año' y 'Mes'.")
                st.stop()

            # Seleccionar sucursal
            sucursales = df['Sucursal'].unique()
            sucursal_seleccionada = st.selectbox("🏢 Seleccione una sucursal:", ["Todas"] + list(sucursales))

            if sucursal_seleccionada != "Todas":
                df = df[df["Sucursal"] == sucursal_seleccionada]

            # Mostrar resumen por producto
            st.markdown("### 📊 Resumen por Producto", unsafe_allow_html=True)
            resumen = calcular_resumen(df)
            st.dataframe(resumen.style.format({
                "Precio_Promedio": "${:.2f}",
                "Margen_Promedio": "{:.2%}",
                "Unidades_Vendidas": "{:.0f}"
            }), use_container_width=True)

            # Mostrar gráfico de evolución de ventas
            st.plotly_chart(graficar_evolucion_ventas(df), use_container_width=True)
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

# Pie de página
st.markdown("---")
st.markdown("**📊 Aplicación desarrollada con Streamlit** - Estilo personalizado para mayor claridad.")
