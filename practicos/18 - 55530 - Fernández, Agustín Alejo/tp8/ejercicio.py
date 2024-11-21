import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://agustin-fernandez-tp8.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 55.555')
        st.markdown('**Nombre:** Juan Pérez')
        st.markdown('**Comisión:** C1')

mostrar_informacion_alumno()

st.sidebar.title('Subir archivo')
archivo_cargado = st.sidebar.file_uploader('Seleccione un archivo CSV', type='csv')
sucursal = st.sidebar.selectbox('Seleccione un valor', options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])

def promedio(ingreso_total, unidades_vendidas):
    return ingreso_total / unidades_vendidas

def mostrar_informacion_producto(df, producto, sucursal=None):
  
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            if sucursal:
                df_producto = df[(df['Sucursal'] == sucursal) & (df['Producto'] == producto)]
            else:
                df_producto = df[df['Producto'] == producto]

            unidades_vendidas = df_producto['Unidades_vendidas'].sum()
            ingreso_total = df_producto['Ingreso_total'].sum()
            costo_total = df_producto['Costo_total'].sum()
            
            precio_promedio = ingreso_total / unidades_vendidas
            margen_promedio = ((ingreso_total - costo_total) / ingreso_total) * 100
            
            precio_promedio_formateado = f"${precio_promedio:,.2f}".replace(",", ".")
            unidades_vendidas_formateado = f"{unidades_vendidas:,}".replace(",", ".")
            
            st.markdown(f"### {producto}")
            st.metric("Precio Promedio", precio_promedio_formateado)
            st.metric("Margen Promedio", f"{margen_promedio:.0f}%")
            st.metric("Unidades Vendidas", unidades_vendidas_formateado)

        with col2:
            df_producto['Año-Mes'] = df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str).str.zfill(2)
            df_producto = df_producto.sort_values(by='Año-Mes')

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(df_producto["Año-Mes"], df_producto["Unidades_vendidas"], label={producto})

            z = np.polyfit(np.arange(len(df_producto)), df_producto["Unidades_vendidas"], 1)
            p = np.poly1d(z)
            ax.plot(df_producto["Año-Mes"], p(np.arange(len(df_producto))), "r--", label="Tendencia") 

            ax.set_xlabel("Año-Mes")
            ax.set_ylabel("Unidades vendidas")
            ax.set_title(f"Evolución de Ventas Mensuales - {producto}", fontsize=14)
            ax.legend()
            plt.xticks(rotation=45, ha='right',fontsize=5)
            plt.grid(True)

            st.pyplot(fig)
        
if archivo_cargado is not None:
    df = pd.read_csv(archivo_cargado)
    
    if archivo_cargado.name == "gaseosas.csv":
        if sucursal == "Todas":
            st.title('Datos de todas las sucursales')
            mostrar_informacion_producto(df, 'Fanta')
            mostrar_informacion_producto(df, 'Coca Cola')
            mostrar_informacion_producto(df, 'Sprite')
            mostrar_informacion_producto(df, '7 Up')
            mostrar_informacion_producto(df, 'Pepsi')
        else:
            st.title(f'Datos de la {sucursal}')
            mostrar_informacion_producto(df, 'Fanta', sucursal)
            mostrar_informacion_producto(df, 'Coca Cola', sucursal)
            mostrar_informacion_producto(df, 'Sprite', sucursal)
            mostrar_informacion_producto(df, '7 Up', sucursal)
            mostrar_informacion_producto(df, 'Pepsi', sucursal)
    
    elif archivo_cargado.name == "vinos.csv":
        if sucursal == "Todas":
            st.title('Datos de todas las sucursales')
            mostrar_informacion_producto(df, 'Cabernet Sauvignon')
            mostrar_informacion_producto(df, 'Chardonnay')
            mostrar_informacion_producto(df, 'Merlot')
            mostrar_informacion_producto(df, 'Pinot Noir')
            mostrar_informacion_producto(df, 'Sauvignon Blanc')
        else:
            st.title(f'Datos de la {sucursal}')
            mostrar_informacion_producto(df, 'Cabernet Sauvignon', sucursal)
            mostrar_informacion_producto(df, 'Chardonnay', sucursal)
            mostrar_informacion_producto(df, 'Merlot', sucursal)
            mostrar_informacion_producto(df, 'Pinot Noir', sucursal)
            mostrar_informacion_producto(df, 'Sauvignon Blanc', sucursal)