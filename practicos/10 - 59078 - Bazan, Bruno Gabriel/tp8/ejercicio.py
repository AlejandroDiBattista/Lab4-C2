import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from typing import Dict, Any


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://bruno-python.streamlit.app/'


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def validar_datos(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame contenga las columnas necesarias"""
    columnas_requeridas = [
        'Año', 'Mes', 'Sucursal', 'Producto',
        'Unidades_vendidas', 'Ingreso_total', 'Costo_total'
    ]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if columnas_faltantes:
        st.error(f"❌ Faltan las columnas: {', '.join(columnas_faltantes)}")
        return False
    return True

def calcular_tendencia(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula la línea de tendencia usando numpy"""
    x_num = np.arange(len(x))
    coefficients = np.polyfit(x_num, y, 1)
    return coefficients[0] * x_num + coefficients[1]

def calcular_metricas(df_producto: pd.DataFrame) -> Dict[str, float]:
    """Calcula las métricas principales para un producto"""
    try:
        ingreso_total = df_producto['Ingreso_total'].sum()
        unidades_vendidas = df_producto['Unidades_vendidas'].sum()
        costo_total = df_producto['Costo_total'].sum()
        
        return {
            'precio_promedio': ingreso_total / unidades_vendidas if unidades_vendidas > 0 else 0,
            'margen_promedio': ((ingreso_total - costo_total) / ingreso_total * 100) if ingreso_total > 0 else 0,
            'unidades_vendidas': unidades_vendidas,
            'ingreso_total': ingreso_total,
            'costo_total': costo_total
        }
    except ZeroDivisionError:
        st.error("Error: División por cero al calcular métricas")
        return None

def procesar_datos(df: pd.DataFrame, sucursal: str, producto: str) -> Dict[str, Any]:
    """Procesa los datos y calcula todas las métricas necesarias"""
    df_filtrado = df.copy()
    if sucursal != "Todas":
        df_filtrado = df_filtrado[df_filtrado['Sucursal'] == sucursal]
    if producto != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Producto'] == producto]
    
    df_filtrado['Fecha'] = pd.to_datetime(df_filtrado.apply(lambda x: f"{x['Año']}-{x['Mes']}-01", axis=1))
    
    resultados = {}
    for prod in df_filtrado['Producto'].unique():
        df_producto = df_filtrado[df_filtrado['Producto'] == prod]
        
        metricas = calcular_metricas(df_producto)
        if metricas is None:
            continue
            
        datos_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
        tendencia = calcular_tendencia(datos_mensuales['Fecha'], datos_mensuales['Unidades_vendidas'])
        
        resultados[prod] = {
            **metricas,
            'datos_mensuales': datos_mensuales,
            'tendencia': tendencia,
            'fechas': datos_mensuales['Fecha']
        }
    
    return resultados

def crear_grafico(resultados: Dict[str, Any], producto: str) -> go.Figure:
    """Crea el gráfico para un producto específico"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=resultados[producto]['fechas'],
        y=resultados[producto]['datos_mensuales']['Unidades_vendidas'],
        name='Ventas',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=resultados[producto]['fechas'],
        y=resultados[producto]['tendencia'],
        name='Tendencia',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Evolución de Ventas - {producto}',
        xaxis_title='Fecha',
        yaxis_title='Unidades Vendidas',
        template='plotly_white',
        showlegend=True,
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Dashboard de Ventas 📊",
        page_icon="📈",
        layout="wide"
    )

    # Sidebar
    with st.sidebar:
        st.title("📊 Panel de Control")
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=200)
        
        uploaded_file = st.file_uploader(
            "📁 Subir archivo CSV",
            type=['csv'],
            help="Limit 200MB per file • CSV"
        )

    if uploaded_file is None:
        st.title("¡Bienvenido al Dashboard de Ventas! 🚀")
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("""
                ### 🌟 Características principales:
                - 📊 Análisis detallado de ventas
                - 📈 Gráficos interactivos
                - 🎯 Métricas en tiempo real
                - 📱 Diseño responsive
            """)
            
            # Usando container con borde nativo
            with st.container(border=True):
                st.markdown("### 🎓 Información del Desarrollador")
                st.markdown("*🔢 Legajo:* 59.078")
                st.markdown("*👨‍💻 Nombre:* Bruno Gabriel Bazán")
                st.markdown("*📚 Comisión:* C2")
        
        with col2:
            lottie_url_welcome = "https://assets3.lottiefiles.com/packages/lf20_dyZfuR.json"
            lottie_welcome = load_lottieurl(lottie_url_welcome)
            if lottie_welcome:
                st_lottie(lottie_welcome, height=300)
        return

    try:
        df = pd.read_csv(uploaded_file)
        
        if not validar_datos(df):
            return
            
        with st.sidebar:
            sucursales = ["Todas"] + sorted(df['Sucursal'].unique().tolist())
            sucursal_seleccionada = st.selectbox("🏢 Seleccionar Sucursal", sucursales)
            
            productos = ["Todos"] + sorted(df['Producto'].unique().tolist())
            producto_seleccionado = st.selectbox("🛍️ Seleccionar Producto", productos)
            
            vista = st.radio("📊 Tipo de Vista", ["Detallada", "Resumen"])

        if sucursal_seleccionada == "Todas":
            st.title("Datos de Todas las Sucursales")
        else:
            st.title(f"Datos de {sucursal_seleccionada}")
        
        resultados = procesar_datos(df, sucursal_seleccionada, producto_seleccionado)
        productos_a_mostrar = [producto_seleccionado] if producto_seleccionado != "Todos" else resultados.keys()
        
        if vista == "Resumen":
            datos_resumen = pd.DataFrame([
                {
                    'Producto': prod,
                    'Unidades Vendidas': resultados[prod]['unidades_vendidas'],
                    'Precio Promedio': resultados[prod]['precio_promedio'],
                    'Margen Promedio': resultados[prod]['margen_promedio']
                }
                for prod in productos_a_mostrar
            ])
            
            st.dataframe(datos_resumen, use_container_width=True)
        else:
            for producto in productos_a_mostrar:
                with st.container(border=True):
                    st.subheader(producto)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric(
                            "📦 Unidades Vendidas",
                            f"{resultados[producto]['unidades_vendidas']:,.0f}",
                            f"{((resultados[producto]['unidades_vendidas'] / 1000000) - 1) * 100:.1f}%"
                        )
                    with cols[1]:
                        st.metric(
                            "💰 Precio Promedio",
                            f"${resultados[producto]['precio_promedio']:.2f}",
                            f"{((resultados[producto]['precio_promedio'] / 1000) - 1) * 100:.1f}%"
                        )
                    with cols[2]:
                        st.metric(
                            "📈 Margen Promedio",
                            f"{resultados[producto]['margen_promedio']:.1f}%",
                            f"{(resultados[producto]['margen_promedio'] - 30):.1f}%"
                        )
                    
                    st.plotly_chart(crear_grafico(resultados, producto), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Ver detalles del error"):
            st.exception(e)

if _name_ == "_main_":
    main()