import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59176.streamlit.app/'

# TP8 CORREGIDO 

# Configuración de la página
st.set_page_config(page_title="Ventas por Producto", layout="wide")

# Función para generar gráficos
def crear_grafico(datos_producto, producto):
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    # Configuración del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_mensuales)), ventas_mensuales['Unidades_vendidas'], label=producto, color="#4CAF50")

    # Línea de tendencia
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas']
    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x, tendencia(x), linestyle='--', color='red', label="Tendencia")

    # Etiquetas y formato
    ax.set_title(f"Evolución Mensual de Ventas: {producto}", fontsize=16)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.set_xticks(range(len(ventas_mensuales)))
    etiquetas = [
        f"{row.Año}" if row.Mes == 1 else "" for row in ventas_mensuales.itertuples()
    ]
    ax.set_xticklabels(etiquetas)
    ax.legend()
    ax.grid()

    return fig

# Función para mostrar métricas
def mostrar_metricas(datos_producto, producto):
    # Cálculos de métricas
    datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
    precio_promedio = datos_producto['Precio_promedio'].mean()

    precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
    variacion_precio = precio_promedio_anual.pct_change().mean() * 100

    datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
    datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
    margen_promedio = datos_producto['Margen'].mean()

    margen_anual = datos_producto.groupby('Año')['Margen'].mean()
    variacion_margen = margen_anual.pct_change().mean() * 100

    unidades_totales = datos_producto['Unidades_vendidas'].sum()
    unidades_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_unidades = unidades_anual.pct_change().mean() * 100

    # Métricas en Streamlit
    with st.expander(f"Estadísticas de {producto}", expanded=True):
        col1, col2 = st.columns([1, 2])

        # Métricas en texto
        with col1:
            st.metric(
                label="Precio Promedio",
                value=f"${precio_promedio:,.0f}".replace(",", "."),
                delta=f"{variacion_precio:.2f}%"
            )
            st.metric(
                label="Margen Promedio",
                value=f"{margen_promedio:.0f}%".replace(",", "."),
                delta=f"{variacion_margen:.2f}%"
            )
            st.metric(
                label="Unidades Vendidas",
                value=f"{unidades_totales:,.0f}".replace(",", "."),
                delta=f"{variacion_unidades:.2f}%"
            )

        # Gráfico de ventas
        with col2:
            fig = crear_grafico(datos_producto, producto)
            st.pyplot(fig)

# Función para cargar y procesar datos
def cargar_datos():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)

        # Selección de sucursales
        sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
        sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
        if sucursal != "Todas":
            datos = datos[datos['Sucursal'] == sucursal]
            st.title(f"Ventas en {sucursal}")
        else:
            st.title("Ventas Totales")

        return datos
    else:
        st.warning("Por favor, sube un archivo CSV.")
        return None

# Mostrar datos del alumno
def mostrar_datos_alumno():
    st.sidebar.markdown("### Datos del Alumno")
    st.sidebar.markdown("Legajo: 59.176")
    st.sidebar.markdown("Nombre: Facundo Nahuel Argañaraz")
    st.sidebar.markdown("Comisión: C2")

# Flujo principal
def main():
    mostrar_datos_alumno()
    datos = cargar_datos()

    if datos is not None:
        productos = datos['Producto'].unique()
        for producto in productos:
            datos_producto = datos[datos['Producto'] == producto]
            mostrar_metricas(datos_producto, producto)

# Iniciar aplicación
if __name__ == "__main__":
    main()
