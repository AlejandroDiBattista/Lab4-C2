# Importar las librerías necesarias
import streamlit as st  # Para crear interfaces web interactivas
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones matemáticas y trabajo con arrays
import matplotlib.pyplot as plt  # Para crear gráficos

# Configurar la página de Streamlit con título y diseño amplio
st.set_page_config(page_title="Ventas por sucursal", layout="wide")

# Función para crear gráficos de evolución de ventas
def crear_grafico_ventas(datos_producto, producto):
    # Agrupar las ventas por año y mes, sumando las unidades vendidas de todas las sucursales
    ventas_agrupadas = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    # Crear el gráfico de evolución de ventas
    fig, grafico_ejes = plt.subplots(figsize=(10, 6))  # Establecer tamaño del gráfico
    grafico_ejes.plot(range(len(ventas_agrupadas)), ventas_agrupadas['Unidades_vendidas'], label=producto)  # Graficar las unidades vendidas

    # Calcular la línea de tendencia
    x = np.arange(len(ventas_agrupadas))  # Crear un array para el eje X
    y = ventas_agrupadas['Unidades_vendidas']  # Datos del eje Y
    valores_ajuste = np.polyfit(x, y, 1)  # Calcular la tendencia (ajuste lineal)
    curva_tendencia = np.poly1d(valores_ajuste)  # Crear la función de tendencia

    # Agregar la línea de tendencia al gráfico
    grafico_ejes.plot(x, curva_tendencia(x), linestyle='--', color='red', label='Tendencia')

    # Ajustar el título y etiquetas
    grafico_ejes.set_title(f'Evolución de Ventas de {producto}' )  # Título del gráfico
    grafico_ejes.set_xticks(range(len(ventas_agrupadas)))  # Configurar las posiciones en el eje X

    # Crear etiquetas para el eje X que solo muestren los años en enero
    etiquetas = []
    for i, row in enumerate(ventas_agrupadas.itertuples()):
        if row.Mes == 1:
            etiquetas.append(f"{row.Año}")  # Mostrar el año si es enero
        else:
            etiquetas.append("")  # Dejar vacío para otros meses
    grafico_ejes.set_xticklabels(etiquetas)

    # Ajustar el eje Y para mostrar valores más claros
    max_y = ventas_agrupadas['Unidades_vendidas'].max()  # Obtener el valor máximo en Y
    intervalo_y = max_y // 8 if max_y > 0 else 1  # Determinar pasos dinámicos
    ticks_y = range(0, max_y + intervalo_y, intervalo_y)  # Crear ticks en el eje Y
    grafico_ejes.set_yticks(ticks_y)
    grafico_ejes.set_ylim(0, max_y + intervalo_y)  # Asegurar que el eje Y comience en 0

    # Agregar leyenda y cuadrícula
    grafico_ejes.legend(title= 'Producto' )
    grafico_ejes.grid(True, linewidth=2)

    return fig  # Retornar el gráfico creado

# Función para calcular métricas de un producto
def calcular_metricas(datos_producto):
    """Calcula las métricas principales para un producto."""
    # Calcular el precio promedio
    datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
    promedio_precio = datos_producto['Precio_promedio'].mean()

    # Calcular la variación anual del precio promedio
    precio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
    variacion_anual_precio = precio_anual.pct_change().mean() * 100  # Variación porcentual

    # Calcular las ganancias promedio y el margen
    datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
    datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
    promedio_margen = datos_producto['Margen'].mean()

    # Calcular la variación anual del margen promedio
    margen_anual = datos_producto.groupby('Año')['Margen'].mean()
    variacion_anual_margen = margen_anual.pct_change().mean() * 100

    # Calcular las unidades vendidas promedio y totales
    promedio_unidades = datos_producto['Unidades_vendidas'].mean()
    total_unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

    # Calcular la variación anual de las unidades vendidas
    unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_anual_unidades = unidades_anuales.pct_change().mean() * 100

    return {
        "promedio_precio": promedio_precio,
        "variacion_anual_precio": variacion_anual_precio,
        "promedio_margen": promedio_margen,
        "variacion_anual_margen": variacion_anual_margen,
        "promedio_unidades": promedio_unidades,
        "total_unidades_vendidas": total_unidades_vendidas,
        "variacion_anual_unidades": variacion_anual_unidades,
    }

# Función para mostrar métricas en la interfaz de Streamlit
def mostrar_metricas_en_ui(metricas, columna_metrica):
    """Muestra las métricas calculadas en la interfaz."""
    with columna_metrica:
        st.metric(label="Precio Promedio", value=f"${metricas['promedio_precio']:,.2f}", delta=f"{metricas['variacion_anual_precio']:.2f}%")
        st.metric(label="Margen Promedio", value=f"{metricas['promedio_margen']:,.2f}%", delta=f"{metricas['variacion_anual_margen']:.2f}%")
        st.metric(label="Unidades Vendidas", value=f"{metricas['total_unidades_vendidas']:,.0f}", delta=f"{metricas['variacion_anual_unidades']:.2f}%")

# Función para mostrar información del alumno en la interfaz
def informacion_alumno():
    datos_alumno = {
        "Legajo": "59.423",
        "Nombre": "Esper Rodrigo Fernando",
        "Comisión": "C2"
    }
    st.markdown(f"**Legajo:** {datos_alumno['Legajo']}")
    st.markdown(f"**Nombre:** {datos_alumno['Nombre']}")
    st.markdown(f"**Comisión:** {datos_alumno['Comisión']}")

# Sección de carga de archivo en la barra lateral
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

# Validar si se cargó un archivo
if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)  # Leer el archivo CSV cargado

    # Obtener la lista de sucursales disponibles
    sucursales_disponibles = ["Todas"] + datos['Sucursal'].unique().tolist()

    # Seleccionar una sucursal desde el sidebar
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales_disponibles)

    # Filtrar los datos según la sucursal seleccionada
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")

    # Iterar por cada producto para calcular métricas y gráficos
    lista_productos = datos['Producto'].unique()
    for producto in lista_productos:
        st.subheader(f"{producto}")
        datos_por_producto = datos[datos['Producto'] == producto]

        # Calcular métricas usando la función auxiliar
        metricas = calcular_metricas(datos_por_producto)

        # Crear columnas para mostrar métricas y gráficos
        columna_metrica, columna_grafico = st.columns([0.25, 0.75])

        # Mostrar las métricas en la interfaz
        mostrar_metricas_en_ui(metricas, columna_metrica)

        # Mostrar el gráfico en la segunda columna
        with columna_grafico:
            figura = crear_grafico_ventas(datos_por_producto, producto)
            st.pyplot(figura)
else:
    # Mostrar mensaje si no se cargó ningún archivo
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59423.streamlit.app/' 

# Mostrar información del alumno al final de la página
informacion_alumno()
