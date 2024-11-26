import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL de la aplicación en Streamlit
# https://tp-58909.streamlit.app/

st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.909')
        st.markdown('**Nombre:** Argañaraz Leonardo Ramiro')
        st.markdown('**Comisión:** C2')  

def crear_grafico_ventas(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, gr = plt.subplots(figsize=(10, 6))
    gr.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto)

    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    ten = np.poly1d(z)

    gr.plot(x, ten(x), linestyle='--', color='red', label='Tendencia')

    gr.set_title('Evolución de Ventas Mensual', fontsize=16)
    gr.set_xlabel('Año-Mes')
    gr.set_xticks(range(len(ventas_por_producto)))

    etiquetas = []
    for i, row in enumerate(ventas_por_producto.itertuples()):
        if row.Mes == 1:
            etiquetas.append(f"{row.Año}")
        else:
            etiquetas.append("")
    gr.set_xticklabels(etiquetas)
    gr.set_ylabel('Unidades Vendidas')
    gr.set_ylim(0, None)  
    gr.legend(title='Producto')
    gr.grid(True)

    return fig

st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()

    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de Ventas en {sucursal_seleccionada}")
    else:
        st.title("Datos de Ventas Totales")

    productos = datos['Producto'].unique()

    for producto in productos:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)

        st.subheader(f"{producto}")
        datos_producto = datos[datos['Producto'] == producto]

        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        precio_promedio = datos_producto['Precio_promedio'].mean()

        precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
        variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()

        margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
        variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

        unidades_promedio = datos_producto['Unidades_vendidas'].mean()
        unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

        unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

        with st.expander(f"Ver estadísticas y gráfico de {producto}", expanded=True):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="Precio Promedioo", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio_promedio_anual:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen_promedio_anual:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_anual_unidades:.2f}%")

            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)     

mostrar_informacion_alumno()