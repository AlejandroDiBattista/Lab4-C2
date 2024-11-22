import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58777-tut9bi2u9wrbzbhbmkidbx.streamlit.app'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.777')
        st.markdown('**Nombre:** Alan Jassán')
        st.markdown('**Comisión:** C2')

def mostrar_detalles(df, sucursal_seleccionada, producto_seleccionado):
    if sucursal_seleccionada == "Todas las Sucursales":
        producto_data = df[df['Producto'] == producto_seleccionado]
    else:
        producto_data = df[(df['Sucursal'] == sucursal_seleccionada) & (df['Producto'] == producto_seleccionado)]
    
    if producto_data.empty:
        precio_promedio = 0
        margen_promedio = 0
        unidades_vendidas = 0
    else:
        total_ingreso = producto_data['Ingreso_total'].sum()
        total_costo = producto_data['Costo_total'].sum()
        unidades_vendidas = producto_data['Unidades_vendidas'].sum()
        precio_promedio = total_ingreso / unidades_vendidas if unidades_vendidas > 0 else 0
        margen_promedio = ((total_ingreso - total_costo) / total_ingreso) * 100 if total_ingreso > 0 else 0
    
    # Obtener datos históricos para el mismo producto y sucursal (si se seleccionó una)
    if sucursal_seleccionada == "Todas las Sucursales":
        historical_data = df[df['Producto'] == producto_seleccionado]
    else:
        historical_data = df[(df['Sucursal'] == sucursal_seleccionada) & (df['Producto'] == producto_seleccionado)]

    # Calcular el margen promedio histórico
    if not historical_data.empty:
        # Excluir el período actual del cálculo de historical_data
        historical_data = historical_data[historical_data['Año'] < producto_data['Año'].max()]

        total_ingreso_historico = historical_data['Ingreso_total'].sum()
        total_costo_historico = historical_data['Costo_total'].sum()
        historical_margen_promedio = ((total_ingreso_historico - total_costo_historico) / total_ingreso_historico) * 100 if total_ingreso_historico > 0 else 0
        historical_unidades_vendidas = historical_data['Unidades_vendidas'].sum()
    else:
        historical_margen_promedio = 0
        historical_unidades_vendidas = 0

    delta_margen = ((margen_promedio - historical_margen_promedio) / historical_margen_promedio) * 100 if historical_margen_promedio != 0 else 0
    delta_unidades = ((unidades_vendidas - historical_unidades_vendidas) / historical_unidades_vendidas) * 100 if historical_unidades_vendidas > 0 else 0

    # Visualización
    st.markdown(f"### {producto_seleccionado}")
    st.metric(label="Precio Promedio", value=f"${precio_promedio:,.2f}", delta=f"{((precio_promedio / 3000) - 1) * 100:.2f}%")
    st.metric(label="Margen Promedio", value=f"{margen_promedio:.2f}%", delta=f"{delta_margen:.2f}%")
    st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}", delta=f"{delta_unidades:.2f}%")

    return producto_data

def graficar_ventas(df, sucursal_seleccionada):
    productos = df['Producto'].unique()
    
    for producto in productos:
        data = df[(df['Producto'] == producto)]
        if sucursal_seleccionada != "Todas las Sucursales":
            data = data[data['Sucursal'] == sucursal_seleccionada]
        
        data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str).str.zfill(2)
        data = data.sort_values(by='Periodo')
        
        if data.empty:
            st.write(f"No hay datos disponibles para {producto} en {sucursal_seleccionada}.")
            continue

        # Cálculo de tendencia
        x = np.arange(len(data))
        if len(x) == 0:
            st.write(f"No hay datos suficientes para calcular la tendencia de {producto}.")
            continue
        z = np.polyfit(x, data['Unidades_vendidas'], 1)
        p = np.poly1d(z)
        
        # Mostrar el año en el mes 1 (enero)
        data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str)
        data = data.sort_values(by='Periodo')
        
        with st.container(border=True):
            col1, col2 = st.columns([1,3])

            with col1:
                mostrar_detalles(df, sucursal_seleccionada, producto)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(data['Periodo'], data['Unidades_vendidas'], label=f'{producto}', color='blue')
                ax.plot(data['Periodo'], p(x), '--r', label='Tendencia', alpha=0.7)
                ax.set_title(f'Evolución de Ventas: {producto}')
                ax.set_xlabel('Período (Año-Mes)')
                ax.set_ylabel('Unidades Vendidas')

                # Agregar líneas verticales por cada mes
                for i in range(len(data)):
                    ax.axvline(x=i, color='black', linestyle='-', alpha=0.3)

                # Generar etiquetas de los años solo para el primer mes de cada año
                años = data['Año'].unique()
                ax.set_xticks(ticks=np.arange(len(años)) * 12)  
                ax.set_xticklabels(labels=años)

                # Agregar líneas horizontales
                ax.grid(axis='y', linestyle='-', alpha=0.7)

                # Ajustar el rango del eje x
                ax.set_xlim(data['Periodo'].min(), data['Periodo'].max())

                ax.legend()
                st.pyplot(fig)
            
st.sidebar.title("Carga de Archivo")
archivo_cargado = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")

if archivo_cargado is None:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container():
        mostrar_informacion_alumno()
else:
    df = pd.read_csv(archivo_cargado)
    
    if 'Sucursal' in df.columns:
        sucursales = ["Todas las Sucursales"] + df['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Selecciona una sucursal", sucursales, key="sucursal_seleccion", disabled=False)
        
        if sucursal_seleccionada:
            st.title(f"Datos cargados para {sucursal_seleccionada}")
            graficar_ventas(df, sucursal_seleccionada)
    else:
        st.error("El archivo no contiene la columna 'Sucursal'. Verifica el formato del archivo.")