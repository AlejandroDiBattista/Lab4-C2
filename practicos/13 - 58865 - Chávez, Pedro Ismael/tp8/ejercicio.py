import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58865.streamlit.app/'


# EJERCICIO RESUELTO TP8

# Configuración inicial de la página
st.set_page_config(page_title="Ventas por Sucursal", layout="wide")

def crear_grafico_ventas(datos_producto, producto):
    # Agrupar las ventas por año y mes
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    # Creación del gráfico de evolución de las ventas
    fig, gr = plt.subplots(figsize=(10, 6))
    gr.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto)
    
    # Calcular la línea de tendencia
    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    ten = np.poly1d(z)
    
    # Línea de tendencia al gráfico
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

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 55.555')
        st.markdown('**Nombre:** Juan Pérez')
        st.markdown('**Comisión:** C1')

# Estilos personalizados
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

# Carga de los datos
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)

    # Obtener lista de sucursales
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()

    # Selección de sucursal
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    # Filtrar datos por sucursal si no se selecciona "Todas"
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de Ventas en {sucursal_seleccionada}")
    else:
        st.title("Datos de Ventas Totales")

    # Calcular las métricas y gráficos por producto
    productos = datos['Producto'].unique()


    for producto in productos:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)

        # Filtrar datos por producto
        st.subheader(f"{producto}")
        datos_producto = datos[datos['Producto'] == producto]

        # Calcular precio promedio
        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        precio_promedio = datos_producto['Precio_promedio'].mean()

        # Calcula la variación anual
        precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
        variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

        # Calcula las ganancias promedio y el margen
        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()

        # Calcula la variación anual del margen promedio
        margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
        variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

        # Calcula las unidades vendidas promedio
        unidades_promedio = datos_producto['Unidades_vendidas'].mean()
        unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

        # Calcula la variación anual de las unidades vendidas
        unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

        # Mostrar y ocultar estadísticas y gráfico
        with st.expander(f"Ver estadísticas y gráfico de {producto}", expanded=True):
            col1, col2 = st.columns([1, 2])

            # Métricas en la primera columna
            with col1:
                st.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{variacion_precio_promedio_anual:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{variacion_margen_promedio_anual:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_anual_unidades:.2f}%")

            # Mostrar el gráfico en la segunda columna ocupando todo el ancho disponible
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.865')
        st.markdown('**Nombre:** Pedro Ismael Chávez')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()