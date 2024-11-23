import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Análisis Detallado de Ventas</h1>", unsafe_allow_html=True)

    st.sidebar.header('Cargar archivo de datos')
    uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        sucursales = ['Todas'] + df['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada != 'Todas':
            df = df[df['Sucursal'] == sucursal_seleccionada]

        st.markdown(f"<h2 style='color: #FF4B4B;'>Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}</h2>", unsafe_allow_html=True)

        productos = df['Producto'].unique()

        for producto in productos:
            st.markdown(f"<h3 style='color: #5F9EA0;'>{producto}</h3>", unsafe_allow_html=True)
            
            df_producto = df[df['Producto'] == producto]
            
            ingreso_total = df_producto['Ingreso_total'].sum()
            unidades_vendidas = df_producto['Unidades_vendidas'].sum()
            costo_total = df_producto['Costo_total'].sum()
            precio_promedio = ingreso_total / unidades_vendidas if unidades_vendidas != 0 else 0
            margen_promedio = ((ingreso_total - costo_total) / ingreso_total) * 100 if ingreso_total != 0 else 0
            delta_precio = np.random.uniform(-5, 5) 
            delta_unidades = np.random.uniform(-10, 10)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Precio Promedio", f"${precio_promedio:.2f}", f"{delta_precio:.2f}%")
            col2.metric("Margen Promedio", f"{margen_promedio:.2f}%", "0.00%")
            col3.metric("Unidades Vendidas", f"{int(unidades_vendidas):,}", f"{delta_unidades:.2f}%")
            
            df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str))
            df_ventas = df_producto.groupby('Fecha').agg({'Unidades_vendidas': 'sum', 'Ingreso_total': 'sum'}).reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_ventas['Fecha'], df_ventas['Unidades_vendidas'], marker='o', label='Unidades Vendidas', color='blue')
            
            z = np.polyfit(df_ventas.index, df_ventas['Unidades_vendidas'], 1)
            p = np.poly1d(z)
            ax.plot(df_ventas['Fecha'], p(df_ventas.index), "r--", label='Tendencia Unidades')

            ax.set_title(f"Evolución Mensual de Ventas - {producto}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Unidades Vendidas")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
                
            st.dataframe(df_producto[['Fecha', 'Unidades_vendidas', 'Ingreso_total']].sort_values('Fecha'))

            st.markdown("<hr>", unsafe_allow_html=True)
                
    else:
        st.info("Por favor, suba un archivo CSV para comenzar.")

if __name__ == '__main__':
    main()

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-58732.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.732')
        st.markdown('**Nombre:** Sergio Antonio Coronel')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()