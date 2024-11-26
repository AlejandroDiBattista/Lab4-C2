import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="TP8", layout="wide")

#URL : https://tp8-59268.streamlit.app/

# Estilos CSS para los contenedores y texto
st.markdown("""
<style>
    .border-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        font-size: 18px; /* Aumentar el tamaño de la letra */
    }
    .metric-container div {
        font-size: 20px; /* Tamaño específico para las métricas */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def generar_grafico_ventas(datos_producto, nombre_producto):
    # Agrupar las ventas sumando las unidades vendidas por año y mes
    ventas_mensuales = datos_producto.pivot_table(index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum').reset_index()

    # Crear la figura y el eje para el gráfico
    figura, eje = plt.subplots(figsize=(8, 4))

    # Dibujar la línea principal con las ventas mensuales
    indice_x = np.arange(len(ventas_mensuales))
    valores_y = ventas_mensuales['Unidades_vendidas']
    eje.plot(indice_x, valores_y, linewidth=1, label=nombre_producto, color='blue')

    # Ajuste polinómico cuadrático para tendencia
    coeficientes = np.polyfit(indice_x, valores_y, 2)
    curva_ajustada = np.poly1d(coeficientes)
    eje.plot(indice_x, curva_ajustada(indice_x), linestyle='--', color='red', linewidth=1, label='Tendencia')

    # Configuración del gráfico
    eje.set_title('Evolución de Ventas Mensual', fontsize=12)
    eje.set_xlabel('Año-Mes', fontsize=10)
    eje.set_ylabel('Unidades Vendidas', fontsize=10)
    eje.set_ylim(0)
    eje.grid(which='major', axis='both', linestyle='--', color='gray', alpha=0.7)

    # Líneas verticales continuas para dividir cada mes
    etiquetas_x = []
    posiciones_x = []
    for idx, fila in enumerate(ventas_mensuales.itertuples()):
        etiquetas_x.append(f"{fila.Año}-{fila.Mes:02d}")
        posiciones_x.append(idx)
        eje.axvline(x=idx, color='black', linestyle='-', linewidth=0.1, alpha=0.6)  # Línea negra para cada mes

    # Líneas horizontales continuas para cada 10,000 unidades
    max_y = int(max(valores_y)) + 10000  # Escala máxima ajustada
    for y_value in range(0, max_y, 10000):  # Intervalos de 10,000 unidades
        eje.axhline(y=y_value, color='black', linestyle='-', linewidth=0.1, alpha=0.6)

    # Configurar etiquetas del eje X
    eje.set_xticks(posiciones_x[::12])  # Mostrar solo una etiqueta por año (cada enero)
    eje.set_xticklabels([etiqueta.split("-")[0] for etiqueta in etiquetas_x[::12]], fontsize=10)
    plt.tight_layout()

    return figura


# Configurar la carga de archivos desde la barra lateral
st.sidebar.title("Carga archivo de datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo:
    datos = pd.read_csv(archivo)

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_actual = st.sidebar.selectbox("Selecciona Sucursal", sucursales)

    if sucursal_actual != "Todas":
        datos_filtrados = datos[datos['Sucursal'] == sucursal_actual]
        st.header(f"Datos de la Sucursal: {sucursal_actual}")
    else:
        datos_filtrados = datos
        st.header("Datos de Todas las Sucursales")

    productos_unicos = datos_filtrados['Producto'].drop_duplicates().values

    for producto in productos_unicos:

        st.subheader(producto)
        datos_producto = datos_filtrados[datos_filtrados['Producto'] == producto]

        datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        promedio_precio = datos_producto['Precio_promedio'].mean()

        precios_por_año = datos_producto.groupby('Año')['Precio_promedio'].mean()
        variacion_precio = precios_por_año.pct_change().mean() * 100

        datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
        datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
        margen_promedio = datos_producto['Margen'].mean()

        margen_anual = datos_producto.groupby('Año')['Margen'].mean()
        variacion_margen = margen_anual.pct_change().mean() * 100

        total_ventas = datos_producto['Unidades_vendidas'].sum()
        promedio_ventas = datos_producto['Unidades_vendidas'].mean()

        ventas_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_ventas = ventas_por_año.pct_change().mean() * 100

        # Crear columnas para las métricas y el gráfico
        col_izq, col_der = st.columns([0.10, 0.75])

        with col_izq:
            # Ajustar el tamaño de letra usando una clase CSS
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Precio Promedio", f"${promedio_precio:,.0f}", f"{variacion_precio:.2f}%")
            st.metric("Margen Promedio", f"{margen_promedio:.0f}%", f"{variacion_margen:.2f}%")
            st.metric("Unidades Vendidas", f"{total_ventas:,.0f}", f"{variacion_ventas:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_der:
            grafico = generar_grafico_ventas(datos_producto, producto)
            st.pyplot(grafico)

else:
    def mostrar_informacion_alumno():
        st.write("**Legajo:** 59268")
        st.write("**Nombre:** Cordoba Pedro")
        st.write("**Comisión:** C2")

    st.subheader("Sube un archivo CSV desde la barra lateral.")
    mostrar_informacion_alumno()
