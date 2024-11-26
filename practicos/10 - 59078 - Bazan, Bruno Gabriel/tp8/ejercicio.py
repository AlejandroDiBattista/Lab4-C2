import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59078.streamlit.app/'

@st.cache_data
def cargar_datos(archivo):
    datos = pd.read_csv(archivo)
    return datos

st.sidebar.title("Cargar Archivo")
archivo_subido = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
st.session_state['archivo_subido'] = archivo_subido

if archivo_subido is not None:
    datos = cargar_datos(archivo_subido)
    
    sucursal = st.sidebar.selectbox("Selecciona la Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])
    
    if sucursal != "Todas":
        st.write(f"### Sucursal seleccionada: {sucursal}")
        datos = datos[datos["Sucursal"] == sucursal]
    else:
        st.write("#### Sucursal seleccionada: Todas")

    columnas_requeridas = ['Ingreso_total', 'Costo_total', 'Producto', 'Unidades_vendidas', 'Año', 'Mes']
    if all(col in datos.columns for col in columnas_requeridas):
        datos['Precio_unitario'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
        datos['Margen'] = (datos['Ingreso_total'] - datos['Costo_total']) / datos['Ingreso_total']
        
        productos = datos["Producto"].unique()
        
        for producto in productos:
            datos_producto = datos[datos["Producto"] == producto]
            
            resumen = datos_producto.groupby('Producto').agg(
                Total_Unidades=('Unidades_vendidas', 'sum'),
                Precio_Prom=('Precio_unitario', 'mean'),
                Margen_Prom=('Margen', 'mean')
            ).reset_index()
            
            datos_producto = datos_producto.sort_values(by=['Año', 'Mes'])
            datos_producto['Unidades_Ant'] = datos_producto['Unidades_vendidas'].shift(1)
            datos_producto['Cambio_Unidades'] = ((datos_producto['Unidades_vendidas'] - datos_producto['Unidades_Ant']) / datos_producto['Unidades_Ant']) * 100
            datos_producto['Precio_Ant'] = datos_producto['Precio_unitario'].shift(1)
            datos_producto['Cambio_Precio'] = ((datos_producto['Precio_unitario'] - datos_producto['Precio_Ant']) / datos_producto['Precio_Ant']) * 100
            datos_producto['Margen_Ant'] = datos_producto['Margen'].shift(1)
            datos_producto['Cambio_Margen'] = ((datos_producto['Margen'] - datos_producto['Margen_Ant']) / datos_producto['Margen_Ant']) * 100
            
            prom_precio = datos_producto.groupby('Año')['Precio_unitario'].mean()
            var_prom_precio = ((prom_precio.diff()) / prom_precio.shift(1)).mean() * 100

            prom_margen = datos_producto.groupby('Año')['Margen'].mean()
            var_prom_margen = ((prom_margen.diff()) / prom_margen.shift(1)).mean() * 100

            total_unidades = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            var_total_unidades = ((total_unidades.diff()) / total_unidades.shift(1)).mean() * 100

            
            with st.container(border=True):
                st.subheader(producto)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Precio Promedio", f"${resumen['Precio_Prom'].values[0]:.2f}", f"{var_prom_precio:.2f}%")
                    st.metric("Margen Promedio", f"{resumen['Margen_Prom'].values[0] * 100:.2f}%", f"{var_prom_margen:.2f}%")
                    st.metric("Unidades Vendidas", f"{resumen['Total_Unidades'].values[0]:.0f}", f"{var_total_unidades:.2f}%")
                    
                with col2:
                    datos_producto['Mes_Año'] = datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str)
                    ventas_mensuales = datos_producto.groupby(['Mes_Año'])['Unidades_vendidas'].sum()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(ventas_mensuales.index, ventas_mensuales.values, label='Ventas', color='blue', linestyle='-')
                    
                    
                    tendencia = np.polyfit(range(len(ventas_mensuales)), ventas_mensuales.values, 1)
                    linea_tendencia = np.poly1d(tendencia)
                    ax.plot(ventas_mensuales.index, linea_tendencia(range(len(ventas_mensuales))), label='Tendencia', color='red', linestyle='--')
                    
                    
                    ax.set_ylim(bottom=0)
                    
                    ax.set_xticks([f'{a}-01' for a in range(2020, 2025)])  
                    ax.set_xticklabels([str(a) for a in range(2020, 2025)])  
                    ax.set_title(f"Evolución de Ventas de {producto}", fontsize=16)
                    ax.set_xlabel("Año-Mes", fontsize=14)
                    ax.set_ylabel("Unidades Vendidas", fontsize=14)
                    ax.set_xticks(range(len(ventas_mensuales)))
                    ax.set_xticklabels([date.split('-')[0] if i == 0 or date.split('-')[0] != ventas_mensuales.index[i-1].split('-')[0] else '' for i, date in enumerate(ventas_mensuales.index)])
                    ax.grid(True)                    
                    ax.legend()
                    st.pyplot(fig)
    else:
        st.error(f"El archivo debe contener las columnas {', '.join(columnas_requeridas)}.")
else:
    st.header("Carga el archivo CSV para comenzar.")
    with st.container(border=True):
        st.write("Legajo: 59.078")
        st.write("Nombre: Bruno Gabriel Bazan")
        st.write("Comisión: C2")