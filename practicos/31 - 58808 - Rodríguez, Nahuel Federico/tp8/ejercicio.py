import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58808.streamlit.app/'

# Mostrar información del alumno 
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.808')
        st.markdown('**Nombre:** Nahuel Federico Rodríguez')
        st.markdown('**Comisión:** C2')

# Función para calcular estadísticas y graficar
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

# Función para graficar la evolución de ventas
def graficar_evolucion(datos):
    # Agrupar por año y mes
    datos['Fecha'] = pd.to_datetime(datos[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1))
    evolucion = datos.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evolucion['Fecha'], evolucion['Unidades_vendidas'], marker='o', label="Unidades vendidas")
    
    # Línea de tendencia
    z = np.polyfit(range(len(evolucion)), evolucion['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(evolucion['Fecha'], p(range(len(evolucion))), linestyle='--', color='orange', label="Tendencia")
    
    ax.set_title("Evolución de Ventas")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()
    
    # Mostrar información a la izquierda
    precio_promedio = datos['Ingreso_total'].sum() / datos['Unidades_vendidas'].sum()
    margen_promedio = (datos['Ingreso_total'].sum() - datos['Costo_total'].sum()) / datos['Ingreso_total'].sum()
    unidades_vendidas = datos['Unidades_vendidas'].sum()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**Precio promedio:** {precio_promedio:.2f}")
        st.markdown(f"**Margen promedio:** {margen_promedio:.2f}")
        st.markdown(f"**Unidades vendidas:** {unidades_vendidas}")
    with col2:
        st.pyplot(fig)
    
    # Graficar evolución por producto
    productos = datos['Producto'].unique()
    for producto in productos:
        datos_producto = datos[datos['Producto'] == producto]
        evolucion_producto = datos_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evolucion_producto['Fecha'], evolucion_producto['Unidades_vendidas'], marker='o', label=f"Unidades vendidas - {producto}")
        
        # Línea de tendencia
        z_producto = np.polyfit(range(len(evolucion_producto)), evolucion_producto['Unidades_vendidas'], 1)
        p_producto = np.poly1d(z_producto)
        ax.plot(evolucion_producto['Fecha'], p_producto(range(len(evolucion_producto))), linestyle='--', color='orange', label="Tendencia")
        
        ax.set_title(f"Evolución de Ventas - {producto}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades Vendidas")
        ax.legend()
        
        # Mostrar información a la izquierda
        precio_promedio_producto = datos_producto['Ingreso_total'].sum() / datos_producto['Unidades_vendidas'].sum()
        margen_promedio_producto = (datos_producto['Ingreso_total'].sum() - datos_producto['Costo_total'].sum()) / datos_producto['Ingreso_total'].sum()
        unidades_vendidas_producto = datos_producto['Unidades_vendidas'].sum()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Producto:** {producto}")
            st.markdown(f"**Precio promedio:** {precio_promedio_producto:.2f}")
            st.markdown(f"**Margen promedio:** {margen_promedio_producto:.2f}")
            st.markdown(f"**Unidades vendidas:** {unidades_vendidas_producto}")
        with col2:
            st.pyplot(fig)

# Carga de datos
def main():
    st.title("Análisis de Ventas")
    mostrar_informacion_alumno()
    
    st.sidebar.title("Opciones")
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type="csv")
    
    if uploaded_file is not None:
        datos = pd.read_csv(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(datos)

        sucursal = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(datos["Sucursal"].unique()))
        if sucursal != "Todas":
            datos_filtrados, resumen = calcular_estadisticas(datos, sucursal)
        else:
            datos_filtrados, resumen = calcular_estadisticas(datos)

        st.subheader("Resumen de Datos")
        st.dataframe(resumen)
        
        st.subheader("Gráfico de Ventas")
        graficar_evolucion(datos_filtrados)

if __name__ == "__main__":
    main()
