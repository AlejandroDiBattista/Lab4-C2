import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

URL = 'https://parcial2tp8-eappjybq5qchyqdfdnson6d.streamlit.app/'

st.set_page_config(page_title="TP8", layout="wide")

def construir_grafico_evolucion(ventas_producto, titulo_producto):
    ventas_agrupadas = ventas_producto.pivot_table(index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum').reset_index()

   
    figura, grafico = plt.subplots(figsize=(10, 6)) 
    
    
    x = np.arange(len(ventas_agrupadas))  
    y = ventas_agrupadas['Unidades_vendidas']
    grafico.plot(x, y, linewidth=2, label=titulo_producto)
    
    indices = np.arange(len(ventas_agrupadas))
    valores = ventas_agrupadas['Unidades_vendidas']
    coeficientes_pol = np.polyfit(indices, valores, 2)
    curva_tendencia = np.poly1d(coeficientes_pol)
    
    grafico.plot(indices, curva_tendencia(indices), linestyle='--', color='red', linewidth=1.5, label='Curva de Tendencia')
    
    grafico.set_title('Evolución de Ventas Mensual', fontsize=16)
    grafico.set_xlabel('Año-Mes', fontsize=12)
    grafico.set_ylabel('Unidades Vendidas', fontsize=12)

    grafico.set_ylim(0)  
    grafico.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  

    grafico.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.5)  
    
    etiquetas = []
    posiciones = []
    for i, fila in enumerate(ventas_agrupadas.itertuples()):
        if fila.Mes == 1:  
            etiquetas.append(str(fila.Año))
            posiciones.append(i)
            
            grafico.axvline(x=i, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)  
     
    grafico.set_xticks(posiciones)
    grafico.set_xticklabels(etiquetas, fontsize=10)

    
    grafico.xaxis.set_minor_locator(plt.MultipleLocator(1))  
    grafico.grid(which='minor', axis='x', linestyle='-', color='gray', alpha=0.3)  

    grafico.legend(title='Producto Destacado')

    return figura


st.sidebar.title("Importación de Archivos")
archivo_cargado = st.sidebar.file_uploader("Carga aquí un archivo CSV", type=["csv"])


if archivo_cargado:
    datos = pd.read_csv(archivo_cargado)
    
    sucursales_disponibles = ["Todas"] + datos['Sucursal'].unique().tolist()
    
    sucursal_seleccionada = st.sidebar.selectbox("Elige una Sucursal", sucursales_disponibles)
    
    if sucursal_seleccionada != "Todas":
        datos_filtrados = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.header(f"Informe de Ventas - Sucursal: {sucursal_seleccionada}")
    else:
        datos_filtrados = datos
        st.header("Informe Consolidado de Ventas")

    lista_productos = datos_filtrados['Producto'].drop_duplicates().values

    for producto in lista_productos:
        st.subheader(producto)
        producto_datos = datos_filtrados[datos_filtrados['Producto'] == producto]
    
        producto_datos['Costo_promedio'] = producto_datos['Ingreso_total'] / producto_datos['Unidades_vendidas']
        costo_medio = producto_datos['Costo_promedio'].mean()

        precios_agrupados = producto_datos.groupby('Año')['Costo_promedio'].mean()
        variacion_precio_anual = precios_agrupados.pct_change().mean() * 100
         
        producto_datos['Ganancia'] = producto_datos['Ingreso_total'] - producto_datos['Costo_total']
        producto_datos['Margen_ganancia'] = (producto_datos['Ganancia'] / producto_datos['Ingreso_total']) * 100
        margen_ganancia_medio = producto_datos['Margen_ganancia'].mean()

        margen_anual = producto_datos.groupby('Año')['Margen_ganancia'].mean()
        variacion_margen_anual = margen_anual.pct_change().mean() * 100
        
        ventas_totales = producto_datos['Unidades_vendidas'].sum()
        ventas_promedio = producto_datos['Unidades_vendidas'].mean()

        ventas_anuales = producto_datos.groupby('Año')['Unidades_vendidas'].sum()
        variacion_ventas_anuales = ventas_anuales.pct_change().mean() * 100
         
        col_izquierda, col_derecha = st.columns([0.25, 0.75])
        
        with col_izquierda:
             st.metric("Precio Promedio", f"${costo_medio:,.0f}", f"{variacion_precio_anual:.2f}%")           
             st.metric("Margen Promedio", f"{margen_ganancia_medio:.0f}%", f"{variacion_margen_anual:.2f}%")
             st.metric("Total Unidades Vendidas", f"{ventas_totales:,.0f}", f"{variacion_ventas_anuales:.2f}%")
         
        with col_derecha:
            grafico = construir_grafico_evolucion(producto_datos, producto)
            st.pyplot(grafico)
else:
    
    def mostrar_info_estudiante():
      st.write("""
    *Información del Usuario Registrado*  
    - Legajo:47417   
    - Nombre: Rosales José Ignacio  
    - Comision: C2 
    """)
   
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

    mostrar_info_estudiante()