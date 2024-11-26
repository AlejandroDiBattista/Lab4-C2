import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mostrar_informacion_alumno():
    
    with st.container(border=True):
        st.markdown('**Legajo:** 59.160.')
        st.markdown('**Nombre:** Veneziano Juan Ignacio')
        st.markdown('**Comisión:** C2')



def graficar_ventas(datos_producto, nombre_producto):
  
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    figura, eje = plt.subplots(figsize=(12, 6))

    eje_x = range(len(ventas_mensuales))
    eje.plot(eje_x, ventas_mensuales['Unidades_vendidas'], 
             marker='o', linewidth=2, markersize=4, 
             label=nombre_producto, color='#1E90FF')

    coeficientes = np.polyfit(eje_x, ventas_mensuales['Unidades_vendidas'], 1)
    polinomio = np.poly1d(coeficientes)
    eje.plot(eje_x, polinomio(eje_x), linestyle='--', color='#800000', 
             label='Tendencia', linewidth=1.5)
    
    
    eje.set_title('Evolución de Ventas Mensual', pad=20, fontsize=20)
    eje.set_xlabel('Período', fontsize=20)
    eje.set_ylabel('Unidades Vendidas', fontsize=20)
    
    eje.set_xticks(eje_x)
    etiquetas = [f"{fila.Año}" if fila.Mes == 1 else "" 
                 for fila in ventas_mensuales.itertuples()]
    eje.set_xticklabels(etiquetas, rotation=45)
    

    eje.set_ylim(0, None)
    eje.legend(title='Producto', bbox_to_anchor=(1.05, 1), loc='upper left')
    eje.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return figura


def calcular_estadisticas(datos_producto):

    datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
    precio_promedio = datos_producto['Precio_promedio'].mean()
    precio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
    variacion_precio = precio_anual.pct_change().mean() * 100

    datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
    datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
    margen_promedio = datos_producto['Margen'].mean()
    margen_anual = datos_producto.groupby('Año')['Margen'].mean()
    variacion_margen = margen_anual.pct_change().mean() * 100

    unidades_totales = datos_producto['Unidades_vendidas'].sum()
    unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
    variacion_unidades = unidades_anuales.pct_change().mean() * 100
    
    return (precio_promedio, variacion_precio, margen_promedio, variacion_margen, 
            unidades_totales, variacion_unidades)

def main():

    st.sidebar.title("Configuración")
    
    archivo_csv = st.sidebar.file_uploader(
        "Cargar archivo CSV",
        type=["csv"],
        help="Seleccione un archivo CSV con los datos de ventas"
    )    
    if archivo_csv:
        datos = pd.read_csv(archivo_csv) 
        sucursales = ["Todas"] + (datos['Sucursal'].unique().tolist())
        sucursal_seleccionada = st.sidebar.selectbox(
            "Seleccionar Sucursal",
            sucursales,
            help="Filtre los datos por sucursal"
        )
        
        if sucursal_seleccionada != "Todas":
            datos = datos[datos['Sucursal'] == sucursal_seleccionada]
            st.title(f"Ventas-{sucursal_seleccionada}")
        else:
            st.title("Datos de todas las sucursales")
            
   
        for producto in (datos['Producto'].unique()):
            with st.container(border=True):
                st.subheader(f" {producto}")
                
                datos_producto = datos[datos['Producto'] == producto]
                estadisticas = calcular_estadisticas(datos_producto)
           
                col_izq, col_der = st.columns([0.25, 0.75])
                
                with col_izq:
                    st.metric(
                        "Precio Promedio",
                        f"${estadisticas[0]:,.0f}",
                        f"{estadisticas[1]:.2f}%"
                    )
                    st.metric(
                        "Margen Promedio",
                        f"{estadisticas[2]:.0f}%",
                        f"{estadisticas[3]:.2f}%"
                    )
                    st.metric( 
                        "Unidades Vendidas",
                        f"{estadisticas[4]:,.0f}",
                        f"{estadisticas[5]:.2f}%"
                        
                    )                
                with col_der:
                    figura = graficar_ventas(datos_producto, producto)
                    st.pyplot(figura)
    else:
        mostrar_informacion_alumno()

main()
 ## Direccion en la que ha sido publicada la aplicacion
    # URL = 'https://venezianojuantp8.streamlit.app/'