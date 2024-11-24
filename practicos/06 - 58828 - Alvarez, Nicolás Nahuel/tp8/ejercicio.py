import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from streamlit_lottie import st_lottie

# Evaluación: 7 | Recuperar para promocionar 
# 1. Usa el mismo codigo que C2-10
# 2. No respeta el diseño (métricas y gráfico a la par) (-1)
# 3. Calcula mal el precio promedio (-1)

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58828.streamlit.app/'

def cargar_animacion(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def validar_columnas(df: pd.DataFrame) -> bool:
    columnas_necesarias = ['Año', 'Mes', 'Sucursal', 'Producto', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        st.error(f"❌ Columnas faltantes: {', '.join(columnas_faltantes)}")
        return False
    return True

def obtener_metricas(df_prod: pd.DataFrame) -> dict:
    try:
        ingreso_total = df_prod['Ingreso_total'].sum()
        unidades_vendidas = df_prod['Unidades_vendidas'].sum()
        costo_total = df_prod['Costo_total'].sum()
        return {
            'precio_promedio': ingreso_total / unidades_vendidas if unidades_vendidas else 0,
            'margen_promedio': ((ingreso_total - costo_total) / ingreso_total * 100) if ingreso_total else 0,
            'unidades_vendidas': unidades_vendidas,
            'ingreso_total': ingreso_total,
            'costo_total': costo_total
        }
    except ZeroDivisionError:
        st.error("Error de cálculo (división por cero)")
        return None

def crear_grafico(fecha, unidades_vendidas, tendencia) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fecha, y=unidades_vendidas, name='Ventas', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=fecha, y=tendencia, name='Tendencia', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Evolución de Ventas', xaxis_title='Fecha', yaxis_title='Unidades Vendidas', template='plotly_white', height=400)
    return fig

def main():
    st.set_page_config(page_title="Dashboard de Ventas 📊", page_icon="📈", layout="wide")

    with st.sidebar:
        st.title("📊 Panel de Control")
        lottie_json = cargar_animacion("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
        if lottie_json:
            st_lottie(lottie_json, height=200)
        archivo = st.file_uploader("📁 Subir archivo CSV", type=['csv'])

    if archivo is None:
        st.title("¡Bienvenido al Dashboard de Ventas! 🚀")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🌟 Características principales: - 📊 Análisis detallado de ventas - 📈 Gráficos interactivos")
        with col2:
            lottie_json = cargar_animacion("https://assets3.lottiefiles.com/packages/lf20_dyZfuR.json")
            if lottie_json:
                st_lottie(lottie_json, height=300)
        return

    try:
        df = pd.read_csv(archivo)
        if not validar_columnas(df):
            return
        
        with st.sidebar:
            sucursal = st.selectbox("🏢 Seleccionar Sucursal", ["Todas"] + sorted(df['Sucursal'].unique()))
            producto = st.selectbox("🛍️ Seleccionar Producto", ["Todos"] + sorted(df['Producto'].unique()))

        st.title(f"Datos de {sucursal if sucursal != 'Todas' else 'Todas las Sucursales'}")
        df_filtrado = df[(df['Sucursal'] == sucursal) | (sucursal == "Todas")]
        df_filtrado = df_filtrado[(df_filtrado['Producto'] == producto) | (producto == "Todos")]
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Año'].astype(str) + '-' + df_filtrado['Mes'].astype(str) + '-01')

        for producto in df_filtrado['Producto'].unique():
            df_producto = df_filtrado[df_filtrado['Producto'] == producto]
            metricas = obtener_metricas(df_producto)
            if metricas is None:
                continue

            datos_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
            tendencia = np.polyfit(np.arange(len(datos_mensuales['Fecha'])), datos_mensuales['Unidades_vendidas'], 1)
            
            st.subheader(f"Análisis de {producto}")
            
            # Mostrar las métricas en una sola fila
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Unidades Vendidas", f"{metricas['unidades_vendidas']}")
            with col2:
                st.metric("💰 Precio Promedio", f"${metricas['precio_promedio']:.2f}")
            with col3:
                st.metric("📈 Margen Promedio", f"{metricas['margen_promedio']:.1f}%")
            
            st.plotly_chart(crear_grafico(datos_mensuales['Fecha'], datos_mensuales['Unidades_vendidas'], tendencia), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.exception(e)

if __name__ == "__main__":
    main()
