import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-55530.streamlit.app/'

st.set_page_config( layout="wide")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 55.530')
        st.markdown('**Nombre:** Fernández, Agustín Alejo')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

def generar_grafico_evolutivo(datos_venta, nombre_articulo):
  
    datos_combinados = datos_venta.pivot_table(index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum').reset_index()

    grafico_figura, grafico_objeto = plt.subplots(figsize=(10, 6))

    eje_x = np.arange(len(datos_combinados)) 
    eje_y = datos_combinados['Unidades_vendidas']
    grafico_objeto.plot(eje_x, eje_y, linewidth=2, label=nombre_articulo)

    coef_pol = np.polyfit(eje_x, eje_y, 2)
    curva_ajustada = np.poly1d(coef_pol)

    grafico_objeto.plot(eje_x, curva_ajustada(eje_x), linestyle='--', color='red', linewidth=1.5, label='Tendencia')

    grafico_objeto.set_title('Evolución de Ventas Mensual', fontsize=16)
    grafico_objeto.set_xlabel('Año-Mes', fontsize=12)
    grafico_objeto.set_ylabel('Unidades Vendidas', fontsize=12)

    grafico_objeto.set_ylim(0)
    grafico_objeto.yaxis.set_major_locator(MaxNLocator(integer=True))

    grafico_objeto.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.5)

    etiquetas = []
    posiciones = []
    for i, fila in enumerate(datos_combinados.itertuples()):
        if fila.Mes == 1: 
            etiquetas.append(str(fila.Año))
            posiciones.append(i)
            grafico_objeto.axvline(x=i, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)


    grafico_objeto.set_xticks(posiciones)
    grafico_objeto.set_xticklabels(etiquetas, fontsize=10)

    grafico_objeto.xaxis.set_minor_locator(MultipleLocator(1))  
    grafico_objeto.grid(which='minor', axis='x', linestyle='-', color='gray', alpha=0.3)

    grafico_objeto.legend(title='Producto')

    return grafico_figura

st.sidebar.title("Subida de Datos")
archivo_csv = st.sidebar.file_uploader("Carga un archivo CSV", type=["csv"])

if archivo_csv:
    tabla_datos = pd.read_csv(archivo_csv)

    opciones_sucursal = ["Todas"] + tabla_datos['Sucursal'].unique().tolist()
    sucursal_actual = st.sidebar.selectbox("Elige una sucursal", opciones_sucursal)

    if sucursal_actual != "Todas":
        datos_filtrados = tabla_datos[tabla_datos['Sucursal'] == sucursal_actual]
        st.header(f"Datos de la {sucursal_actual}")
    else:
        datos_filtrados = tabla_datos
        st.header("Datos de todas las sucursales")

    lista_productos = datos_filtrados['Producto'].unique()
    for item in lista_productos:
        st.subheader(item)
        datos_producto = datos_filtrados[datos_filtrados['Producto'] == item]
        
        datos_producto['Precio_medio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        promedio_precio = datos_producto['Precio_medio'].mean()
        precios_anuales = datos_producto.groupby('Año')['Precio_medio'].mean()
        cambio_precio_anual = precios_anuales.pct_change().mean() * 100

        datos_producto['Utilidad'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Porcentaje_ganancia'] = (datos_producto['Utilidad'] / datos_producto['Ingreso_total']) * 100
        promedio_ganancia = datos_producto['Porcentaje_ganancia'].mean()
        ganancia_anual = datos_producto.groupby('Año')['Porcentaje_ganancia'].mean()
        cambio_ganancia_anual = ganancia_anual.pct_change().mean() * 100

        total_ventas = datos_producto['Unidades_vendidas'].sum()
        ventas_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        cambio_ventas_anuales = ventas_anuales.pct_change().mean() * 100

        col_izq, col_der = st.columns([0.25, 0.75])
        with col_izq:
            st.metric("Precio Medio", f"${int(promedio_precio):,}".replace(",", "."), f"{cambio_precio_anual:.2f}%")
            st.metric("Margen Medio", f"{promedio_ganancia:.0f}%", f"{cambio_ganancia_anual:.2f}%")
            st.metric("Total Ventas", f"{total_ventas:,.0f}".replace(",", "."), f"{cambio_ventas_anuales:.2f}%")
        with col_der:
            grafico = generar_grafico_evolutivo(datos_producto, item)
            st.pyplot(grafico)
else:
    st.info("Por favor, sube un archivo CSV para continuar.")