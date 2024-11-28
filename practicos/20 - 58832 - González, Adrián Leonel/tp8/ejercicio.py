import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58832.streamlit.app/'

st.title("Análisis de Ventas por Producto y Sucursal")

st.sidebar.header("Cargar datos y seleccionar sucursal")

uploaded_file = st.sidebar.file_uploader("Sube el archivo CSV de ventas", type="csv")

sucursal = None
if uploaded_file is not None:
    Datos = pd.read_csv(uploaded_file)
    
    sucursal = st.sidebar.selectbox("Selecciona una sucursal", ['Todas'] + Datos['Sucursal'].unique().tolist())

    años_disponibles = sorted(Datos['Año'].unique()) 
    año_seleccionado = st.sidebar.selectbox("Selecciona el año o 'Todos'", ['Todos'] + años_disponibles)

if uploaded_file is not None:
    if sucursal != 'Todas':
        Datos = Datos[Datos['Sucursal'] == sucursal]

    if año_seleccionado != 'Todos':
        Datos = Datos[Datos['Año'] == año_seleccionado]

    Datos['Precio_promedio'] = Datos['Ingreso_total'] / Datos['Unidades_vendidas']
    Datos['Margen_promedio'] = (Datos['Ingreso_total'] - Datos['Costo_total']) / Datos['Ingreso_total']
    
    resumen = Datos.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()

    st.header(f"Resumen de Productos en {año_seleccionado}" if año_seleccionado != 'Todos' else "Resumen de Productos en Todos los Años")
    
    for producto in resumen['Producto']:
        with st.container():
            st.subheader(f"Producto: {producto}")
            
            producto_data = Datos[Datos['Producto'] == producto]
            
            precio_promedio = producto_data['Precio_promedio'].mean()
            margen_promedio = producto_data['Margen_promedio'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Precio Promedio:** {precio_promedio:.2f}")
                st.write(f"**Margen Promedio:** {margen_promedio:.2f}")
                st.write(f"**Unidades Vendidas:** {resumen[resumen['Producto'] == producto]['Unidades_vendidas'].values[0]}")
            
            with col2:
                producto_ventas = producto_data.groupby(['Año', 'Mes']).agg(
                    Ventas_totales=('Unidades_vendidas', 'sum')
                ).reset_index()

                if año_seleccionado == 'Todos':
                    producto_ventas = producto_data.groupby(['Año']).agg(
                        Ventas_totales=('Unidades_vendidas', 'sum')
                    ).reset_index()
                    producto_ventas['Año'] = producto_ventas['Año'].astype(str)

                if año_seleccionado != 'Todos':
                    mes_abreviado = {
                        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
                    }
                    producto_ventas['Mes_abreviado'] = producto_ventas['Mes'].map(mes_abreviado)
                else:
                    producto_ventas['Año'] = producto_ventas['Año'].astype(str)

                fig, ax = plt.subplots(figsize=(10, 6))

                if año_seleccionado != 'Todos':
                    ax.plot(producto_ventas['Mes_abreviado'], producto_ventas['Ventas_totales'], label='Ventas Totales', color='blue')
                else:
                    ax.plot(producto_ventas['Año'], producto_ventas['Ventas_totales'], label='Ventas Totales', color='blue')

                x_values = np.arange(len(producto_ventas))  
                y_values = producto_ventas['Ventas_totales'].values

                m, b = np.polyfit(x_values, y_values, 1) 

                trendline = m * x_values + b

                ax.plot(producto_ventas['Mes_abreviado'] if año_seleccionado != 'Todos' else producto_ventas['Año'], trendline, label='Línea de Tendencia', color='red', linestyle='--')

                ax.set_title(f'Evolución de Ventas de {producto} en {año_seleccionado} con Línea de Tendencia' if año_seleccionado != 'Todos' else f'Evolución de Ventas de {producto} en Todos los Años con Línea de Tendencia')
                ax.set_xlabel('Mes' if año_seleccionado != 'Todos' else 'Año')
                ax.set_ylabel('Unidades Vendidas')

                ax.set_ylim(bottom=0)

                ax.legend()
                
                st.pyplot(fig)

    st.write(f"Análisis completado para el año {año_seleccionado}. Puedes ver la evolución de ventas y los datos resumidos por producto.")

def mostrar_informacion_alumno():
    with st.sidebar.container(border=True):
        st.sidebar.markdown('**Legajo:** 58.832')
        st.sidebar.markdown('**Nombre:** Adrian Leonel Gonzalez')
        st.sidebar.markdown('**Comisión:** C2')

mostrar_informacion_alumno()


