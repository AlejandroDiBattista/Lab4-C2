import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="TP8", layout="wide")

def construir_grafico_evolucion(ventas_producto, titulo_producto):
    # Organiza los datos de ventas sumando las unidades por cada combinación de año y mes
    ventas_agrupadas = ventas_producto.pivot_table(index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum').reset_index()

    # Crear una figura para el gráfico y un espacio para dibujar
    figura, grafico = plt.subplots(figsize=(10, 6)) 
    
    # Añade una línea principal al gráfico basada en las ventas acumuladas por mes
    x = np.arange(len(ventas_agrupadas))  # Generar índices para el eje X
    y = ventas_agrupadas['Unidades_vendidas']
    grafico.plot(x, y, linewidth=2, label=titulo_producto)
    
    # Generar un ajuste polinómico cuadrático para modelar la tendencia de los datos
    indices = np.arange(len(ventas_agrupadas))
    valores = ventas_agrupadas['Unidades_vendidas']
    coeficientes_pol = np.polyfit(indices, valores, 2)
    curva_tendencia = np.poly1d(coeficientes_pol)
    
    # Incorporar la curva de tendencia en el gráfico como una línea segmentada
    grafico.plot(indices, curva_tendencia(indices), linestyle='--', color='red', linewidth=1.5, label='Curva de Tendencia')
    
    # Configuración del título, etiquetas de ejes y leyenda del gráfico
    grafico.set_title('Evolución de Ventas Mensual', fontsize=16)
    grafico.set_xlabel('Año-Mes', fontsize=12)
    grafico.set_ylabel('Unidades Vendidas', fontsize=12)

    # Configurar límites e intervalos del eje Y automáticamente
    grafico.set_ylim(0)  # Asegurar que el eje Y comienza desde 0
    grafico.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ajustar automáticamente los valores del eje Y

    # Configurar grilla para que solo se muestre en los valores principales del eje Y
    grafico.grid(which='major', axis='y', linestyle='-', color='gray', alpha=0.5)  # Grilla solo en valores principales del eje Y
    
    # Crear etiquetas para el eje X, mostrando únicamente el año en enero
    etiquetas = []
    posiciones = []
    for i, fila in enumerate(ventas_agrupadas.itertuples()):
        if fila.Mes == 1:  # Condición para capturar únicamente el inicio de cada año
            etiquetas.append(str(fila.Año))
            posiciones.append(i)
            # Añadir línea sólida vertical en cada inicio de año
            grafico.axvline(x=i, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)  # Línea sólida para los años
    
    # Configurar el gráfico para mostrar etiquetas específicas del eje X
    grafico.set_xticks(posiciones)
    grafico.set_xticklabels(etiquetas, fontsize=10)

    # Configurar líneas menores (12 columnas por año, sólidas)
    grafico.xaxis.set_minor_locator(plt.MultipleLocator(1))  # Cada índice representa un mes
    grafico.grid(which='minor', axis='x', linestyle='-', color='gray', alpha=0.3)  # Líneas sólidas para los meses

    grafico.legend(title='Producto Destacado')

    return figura

# Configurar la carga de archivos desde la barra lateral
st.sidebar.title("Importación de Archivos")
archivo_cargado = st.sidebar.file_uploader("Carga aquí un archivo CSV", type=["csv"])

# Comprobar si el archivo CSV fue cargado
if archivo_cargado:
    datos = pd.read_csv(archivo_cargado)
    
    # Identificar las sucursales disponibles para filtrar
    sucursales_disponibles = ["Todas"] + datos['Sucursal'].unique().tolist()
    
    # Crear un menú desplegable para seleccionar una sucursal
    sucursal_seleccionada = st.sidebar.selectbox("Elige una Sucursal", sucursales_disponibles)
    
    # Aplicar un filtro para seleccionar los datos de la sucursal elegida
    if sucursal_seleccionada != "Todas":
        datos_filtrados = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.header(f"Informe de Ventas - Sucursal: {sucursal_seleccionada}")
    else:
        datos_filtrados = datos
        st.header("Informe Consolidado de Ventas")

    # Listar todos los productos disponibles en los datos filtrados
    lista_productos = datos_filtrados['Producto'].drop_duplicates().values

    for producto in lista_productos:
        st.subheader(producto)
        producto_datos = datos_filtrados[datos_filtrados['Producto'] == producto]
        
        # Calcular el precio promedio basado en los ingresos y unidades vendidas
        producto_datos['Costo_promedio'] = producto_datos['Ingreso_total'] / producto_datos['Unidades_vendidas']
        costo_medio = producto_datos['Costo_promedio'].mean()

        # Calcular la variación porcentual anual del precio promedio
        precios_agrupados = producto_datos.groupby('Año')['Costo_promedio'].mean()
        variacion_precio_anual = precios_agrupados.pct_change().mean() * 100
        
        # Calcular las ganancias y márgenes para evaluar rentabilidad
        producto_datos['Ganancia'] = producto_datos['Ingreso_total'] - producto_datos['Costo_total']
        producto_datos['Margen_ganancia'] = (producto_datos['Ganancia'] / producto_datos['Ingreso_total']) * 100
        margen_ganancia_medio = producto_datos['Margen_ganancia'].mean()

        margen_anual = producto_datos.groupby('Año')['Margen_ganancia'].mean()
        variacion_margen_anual = margen_anual.pct_change().mean() * 100
        
        # Sumar las ventas totales y calcular la variación anual
        ventas_totales = producto_datos['Unidades_vendidas'].sum()
        ventas_promedio = producto_datos['Unidades_vendidas'].mean()

        ventas_anuales = producto_datos.groupby('Año')['Unidades_vendidas'].sum()
        variacion_ventas_anuales = ventas_anuales.pct_change().mean() * 100
        
        # Dividir el espacio en dos columnas para mostrar gráficos y estadísticas
        col_izquierda, col_derecha = st.columns([0.25, 0.75])
        
        # Mostrar métricas clave en la columna izquierda
        with col_izquierda:
             st.metric("Precio Promedio", f"${costo_medio:,.0f}", f"{variacion_precio_anual:.2f}%")           
             st.metric("Margen Promedio", f"{margen_ganancia_medio:.0f}%", f"{variacion_margen_anual:.2f}%")
             st.metric("Total Unidades Vendidas", f"{ventas_totales:,.0f}", f"{variacion_ventas_anuales:.2f}%")
        
        # Mostrar el gráfico de ventas en la columna derecha
        with col_derecha:
            grafico = construir_grafico_evolucion(producto_datos, producto)
            st.pyplot(grafico)
else:
    # Mostrar información adicional sobre el usuario
    def mostrar_info_estudiante():
      st.write("""
    **Información del Usuario Registrado**  
    - Legajo: 59130  
    - Nombre: Luciano Gatti  
    - Comision: C2 
    """)
    ## Direccion en la que ha sido publicada la aplicacion
    # URL = 'https://tp8-59130.streamlit.app/'

    # Mostrar mensaje cuando no se cargó ningún archivo
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")

    mostrar_info_estudiante()
