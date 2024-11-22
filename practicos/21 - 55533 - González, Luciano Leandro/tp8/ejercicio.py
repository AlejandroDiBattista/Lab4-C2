import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
import base64

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-55533.streamlit.app/'

def mostrar_informacion_alumno():
    st.markdown("# Por favor, sube un archivo CSV desde la barra lateral.")
    st.markdown(
        """
        <div style="border: 1px solid #d3d3d3; padding: 15px; width: 300px; border-radius: 10px; margin-left: 0;">
            <p><strong>Legajo:</strong> 55533</p>
            <p><strong>Nombre:</strong> Gonzalez Luciano</p>
            <p><strong>Comisión:</strong> C2</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def cargar_datos():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.sidebar.success("Archivo cargado correctamente")
        return datos
    else:
        return None

def calcular_estadisticas(datos, sucursal=None):
    if sucursal:
        datos = datos[datos["Sucursal"] == sucursal]

    # Calcular métricas
    datos['Precio_promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    datos['Margen_promedio'] = (datos['Ingreso_total'] - datos['Costo_total']) / datos['Ingreso_total']

    # Resumir por producto
    resumen = datos.groupby('Producto').agg({
        'Unidades_vendidas': 'sum',
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean'
    }).reset_index()

    return datos, resumen

def graficar_evolucion(datos, producto=None):
    # Agrupar por año y mes
    datos['Fecha'] = pd.to_datetime(datos[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1))
    evolucion = datos.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()

    # Crear gráfico
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evolucion['Fecha'], evolucion['Unidades_vendidas'], color='blue', label=f"Unidades vendidas - {producto}")

    # Línea de tendencia
    z = np.polyfit(range(len(evolucion)), evolucion['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(evolucion['Fecha'], p(range(len(evolucion))), linestyle='--', color='red', label="Tendencia")

    ax.set_title("Evolución de Ventas Mensual")
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()
    plt.grid(True)
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    # Mostrar información a la izquierda
    col1, col2 = st.columns([1, 3])
    with col1:
        if producto:
            st.markdown(f"**Producto:** {producto}")
        st.markdown(f"**Precio promedio:** {datos['Ingreso_total'].sum() / datos['Unidades_vendidas'].sum():.2f}")
        st.markdown(f"**Margen promedio:** {(datos['Ingreso_total'].sum() - datos['Costo_total'].sum()) / datos['Ingreso_total'].sum():.2%}")
        st.markdown(f"**Unidades vendidas:** {datos['Unidades_vendidas'].sum()}")
    with col2:
        st.markdown(f"<img src='data:image/png;base64,{encoded_image}' style='max-width: 100%; height: auto; border-radius: 10px;'/>", unsafe_allow_html=True)

def main():
    datos = cargar_datos()
    if datos is None:
        mostrar_informacion_alumno()
    else:
        st.title("Análisis de Ventas")
        sucursal = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(datos["Sucursal"].unique()))
        if sucursal != "Todas":
            datos_filtrados, resumen = calcular_estadisticas(datos, sucursal)
        else:
            datos_filtrados, resumen = calcular_estadisticas(datos)

        for producto in resumen['Producto']:
            datos_producto = datos_filtrados[datos_filtrados['Producto'] == producto]
            graficar_evolucion(datos_producto, producto)

if __name__ == "__main__":
    main()
