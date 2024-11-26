import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def agrupar_ventas(info_producto):
    return info_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

def agregar_tendencia(ax, ventas_producto):
    x = np.arange(len(ventas_producto))
    y = ventas_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='--', color='red', label='Tendencia')

def configurar_grafico(ax, ventas_producto):
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(range(len(ventas_producto)))
    etiquetas = [f"{row.Año}" if row.Mes == 1 else "" for row in ventas_producto.itertuples()]
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)

def calcular_precio_promedio(info_producto):
    info_producto['Precio_promedio'] = info_producto['Ingreso_total'] / info_producto['Unidades_vendidas']
    precio_promedio = info_producto['Precio_promedio'].mean()
    precio_promedio_anual = info_producto.groupby('Año')['Precio_promedio'].mean()
    variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100
    return precio_promedio, variacion_precio_promedio_anual

def Venta_Graficar(info_producto, producto):
    ventas_producto = agrupar_ventas(info_producto)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_producto)), ventas_producto['Unidades_vendidas'], label=producto)
    agregar_tendencia(ax, ventas_producto)
    configurar_grafico(ax, ventas_producto)
    return fig

def calcular_margen_promedio(info_producto):
    info_producto['Ganancia'] = info_producto['Ingreso_total'] - info_producto['Costo_total']
    info_producto['Margen'] = (info_producto['Ganancia'] / info_producto['Ingreso_total']) * 100
    margen_promedio = info_producto['Margen'].mean()
    margen_promedio_anual = info_producto.groupby('Año')['Margen'].mean()
    variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100
    return margen_promedio, variacion_margen_promedio_anual

def calcular_unidades_vendidas(info_producto):
    unidades_vendidas = info_producto['Unidades_vendidas'].sum()
    unidades_por_año = info_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100
    return unidades_vendidas, variacion_anual_unidades

def mostrar_metricas(col, precio_promedio, variacion_precio_promedio_anual, margen_promedio, variacion_margen_promedio_anual, unidades_vendidas, variacion_anual_unidades):
    col.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio_promedio_anual:.2f}%")
    col.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen_promedio_anual:.2f}%")
    col.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_anual_unidades:.2f}%")

st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    productos = datos['Producto'].unique()
    for producto in productos:
        
        with st.container(border=True):
            st.subheader(f"{producto}")
            info_producto = datos[datos['Producto'] == producto]
            precio_promedio, variacion_precio_promedio_anual = calcular_precio_promedio(info_producto)
            margen_promedio, variacion_margen_promedio_anual = calcular_margen_promedio(info_producto)
            unidades_vendidas, variacion_anual_unidades = calcular_unidades_vendidas(info_producto)
            
            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                mostrar_metricas(col1, precio_promedio, variacion_precio_promedio_anual, margen_promedio, variacion_margen_promedio_anual, unidades_vendidas, variacion_anual_unidades)
            with col2:
                fig = Venta_Graficar(info_producto, producto)
                st.pyplot(fig)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-lab-59234.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.234')
        st.markdown('**Nombre:** Matias Sebastian Chocobar')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()
