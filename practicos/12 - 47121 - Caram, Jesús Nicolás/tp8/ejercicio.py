import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .panel-datos {
        border: 2px solid black;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f0f0;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Cargar archivo de datos")
archivoCsv = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

def mostrarInformacionAlumno():
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    st.markdown("""
        <div class="panel-datos">
            <p><strong>Legajo:</strong> 47121</p>
            <p><strong>Apellido y Nombre:</strong> Caram Jesús Nicolás</p>
            <p><strong>Comisión:</strong> 2</p>
        </div>
    """, unsafe_allow_html=True)

def crear_grafico_ventas(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    fig, gr = plt.subplots(figsize=(10, 6))
    gr.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto)
    
    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    tendencia = np.poly1d(z)

    gr.plot(x, tendencia(x), linestyle='--', color='red', label='Tendencia')
    
    gr.set_title('Evolución de Ventas Mensual', fontsize=16)
    gr.set_xlabel('Año-Mes')
    gr.set_xticks(range(len(ventas_por_producto)))
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_por_producto.itertuples()]
    gr.set_xticklabels(etiquetas)
    gr.set_ylabel('Unidades Vendidas')
    gr.set_ylim(0, None)
    gr.legend(title='Producto')
    gr.grid(True)

    return fig

if archivoCsv is None:
    mostrarInformacionAlumno()
else:
    df = pd.read_csv(archivoCsv)
    sucursales = ["Todas"] + df['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    if sucursal_seleccionada != "Todas":
        df = df[df['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de Ventas en {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")

    productos = sorted(df['Producto'].unique(), key=lambda x: (not str(x)[0].isalpha(), str(x).lower()))

    for producto in productos:
        with st.container(border=True):
            datos_producto = df[df['Producto'] == producto]
            
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_promedio'].mean()
            
            precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100
            
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = datos_producto['Margen'].mean()
            
            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100
            
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

            st.subheader(f"{producto}")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(label="Precio Promedio", 
                         value=f"${precio_promedio:,.0f}".replace(",", "."), 
                         delta=f"{variacion_precio_promedio_anual:.2f}%")
                
                st.metric(label="Margen Promedio", 
                         value=f"{margen_promedio:.0f}%".replace(",", "."), 
                         delta=f"{variacion_margen_promedio_anual:.2f}%")
                
                st.metric(label="Unidades Vendidas", 
                         value=f"{unidades_vendidas:,.0f}".replace(",", "."), 
                         delta=f"{variacion_anual_unidades:.2f}%")
            
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)
                plt.close(fig)