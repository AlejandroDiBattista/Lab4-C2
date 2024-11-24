import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58828.streamlit.app/'


def mostrar_informacion_alumno():
    st.title("Información del Alumno:")
    st.markdown('*Legajo:* 58.828')
    st.markdown('*Nombre:* Nicolas Nahuel Alvarez')
    st.markdown('*Comisión:* C2')
    st.markdown("---") 


mostrar_informacion_alumno()


st.sidebar.title("Panel de Control de Ventas")
st.sidebar.markdown("Cargar archivo CSV para análisis de ventas de productos.")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])


if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    
    st.title("Análisis de Ventas")
    st.markdown("Selecciona una sucursal para visualizar las tendencias de ventas de todos los productos.")

    
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ['Todas'] + list(df['Sucursal'].unique()))

    
    if sucursal != 'Todas':
        df_sucursal = df[df['Sucursal'] == sucursal]
    else:
        df_sucursal = df

    
    productos = df_sucursal['Producto'].unique()

    for producto in productos:
        
        df_producto = df_sucursal[df_sucursal['Producto'] == producto]

        
        df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01', format='%Y-%m-%d')

      
        precio_promedio = df_producto['Ingreso_total'].sum() / df_producto['Unidades_vendidas'].sum()
        margen_promedio = ((df_producto['Ingreso_total'] - df_producto['Costo_total']).sum() / df_producto['Ingreso_total'].sum()) * 100
        unidades_vendidas = df_producto['Unidades_vendidas'].sum()

      
        with st.container():
          
            st.markdown("""
                <div style="border: 2px solid #ccc; border-radius: 10px; padding: 20px; margin-bottom: 30px; background-color: #f9f9f9;">
                    <h3 style="text-align:center; font-weight: bold; color: #333;">{}</h3>
                    """.format(producto), unsafe_allow_html=True)

          
            col1, col2, col3 = st.columns(3)
            col1.metric("Precio Promedio", f"${precio_promedio:,.2f}")
            col2.metric("Margen Promedio", f"{margen_promedio:.2f}%")
            col3.metric("Unidades Vendidas", f"{int(unidades_vendidas):,}")

           
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], label=producto, color='b')

           
            z = np.polyfit(df_producto['Fecha'].astype(int), df_producto['Unidades_vendidas'], 1)
            p = np.poly1d(z)
            ax.plot(df_producto['Fecha'], p(df_producto['Fecha'].astype(int)), label="Tendencia", color='r', linestyle='--')

            ax.set_xlabel('Fecha')
            ax.set_ylabel('Unidades Vendidas')
            ax.set_title(f'Evolución de Ventas Mensuales')
            ax.legend()

            
            st.pyplot(fig)

            
            st.markdown("</div>", unsafe_allow_html=True)
