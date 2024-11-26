import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://55751-franciscojerez.streamlit.app/'


st.set_page_config(page_title="FranciscoJerez-TP8", layout="wide")

def crear_grafico_ventas(datos_ventas, nombre_producto):
    datos_agrupados = datos_ventas.pivot_table(index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum').reset_index()

    figura, eje = plt.subplots(figsize=(10, 6)) 

    x = np.arange(len(datos_agrupados))
    y = datos_agrupados['Unidades_vendidas']
    eje.plot(x, y, linewidth=3, label=nombre_producto)

    indices = np.arange(len(datos_agrupados))
    valores = datos_agrupados['Unidades_vendidas']
    coef_pol = np.polyfit(indices, valores, 2)
    curva_tendencia = np.poly1d(coef_pol)

    eje.plot(indices, curva_tendencia(indices), linestyle='--', color='red', linewidth=1.5, label='Curva de Tendencia')

    eje.set_title('Evolución de Ventas (mensual)', fontsize=16)
    eje.set_xlabel('Año-Mes', fontsize=15)
    eje.set_ylabel('Unidades Vendidas', fontsize=15)
    eje.set_ylim(0)
    eje.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    eje.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.5)

    etiquetas = []
    posiciones = []
    for i, fila in enumerate(datos_agrupados.itertuples()):
        if fila.Mes == 1:
            etiquetas.append(str(fila.Año))
            posiciones.append(i)
            eje.axvline(x=i, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

    eje.set_xticks(posiciones)
    eje.set_xticklabels(etiquetas, fontsize=10)

    eje.xaxis.set_minor_locator(plt.MultipleLocator(1))
    eje.grid(which='minor', axis='x', linestyle='-', color='gray', alpha=0.3)
    eje.legend(title='Producto Destacado')

    return figura

def cargar_datos(archivo):
    return pd.read_csv(archivo)

def procesar_sucursal(datos, sucursal):
    if sucursal != "Todas":
        return datos[datos['Sucursal'] == sucursal]
    return datos

def calcular_metricas(datos_producto):
    datos_producto['Costo_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
    costo_promedio = datos_producto['Costo_promedio'].mean()

    precios_anuales = datos_producto.groupby('Año')['Costo_promedio'].mean()
    variacion_precio = precios_anuales.pct_change().mean() * 100

    datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
    datos_producto['Margen_ganancia'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
    margen_promedio = datos_producto['Margen_ganancia'].mean()

    margen_anual = datos_producto.groupby('Año')['Margen_ganancia'].mean()
    variacion_margen = margen_anual.pct_change().mean() * 100

    ventas_totales = datos_producto['Unidades_vendidas'].sum()
    ventas_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_ventas = ventas_anuales.pct_change().mean() * 100

    return costo_promedio, variacion_precio, margen_promedio, variacion_margen, ventas_totales, variacion_ventas

def mostrar_info_estudiante():
    st.write("""
    - Numero de Legajo: 55751 
    - Nombre: Francisco Jerez 
    - Comision: C2 
    """)

archivo_cargado = st.sidebar.file_uploader("Carga aquí un archivo CSV", type=["csv"])

if archivo_cargado:
    datos = cargar_datos(archivo_cargado)

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Elegi una Sucursal", sucursales)

    datos_filtrados = procesar_sucursal(datos, sucursal_seleccionada)
    st.header(f"Informe de Ventas - Sucursal: {sucursal_seleccionada}" if sucursal_seleccionada != "Todas" else "DATOS DE TODAS LAS SUCURSALES")

    productos = datos_filtrados['Producto'].drop_duplicates().values

    for producto in productos:
        st.subheader(producto)
        datos_producto = datos_filtrados[datos_filtrados['Producto'] == producto]

        costo_promedio, variacion_precio, margen_promedio, variacion_margen, ventas_totales, variacion_ventas = calcular_metricas(datos_producto)

        col_izquierda, col_derecha = st.columns([0.25, 0.75])

        with col_izquierda:
            st.metric("Precio Promedio", f"${costo_promedio:,.0f}", f"{variacion_precio:.2f}%")
            st.metric("Margen Promedio", f"{margen_promedio:.0f}%", f"{variacion_margen:.2f}%")
            st.metric("Total Unidades Vendidas", f"{ventas_totales:,.0f}", f"{variacion_ventas:.2f}%")

        with col_derecha:
            grafico = crear_grafico_ventas(datos_producto, producto)
            st.pyplot(grafico)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    mostrar_info_estudiante()