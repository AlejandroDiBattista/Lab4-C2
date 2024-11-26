import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL de la aplicación en Streamlit
# https://aalejoo12-parcial2lab4-ejercicio-jgiiuc.streamlit.app/


# Configuración de la página
st.set_page_config(page_title="Cargar archivo de datos", layout="wide")

def mostrar_informacion_alumno():
    st.markdown(
        """
        <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 5px;">
            <p><strong>Legajo:</strong> 58.829</p>
            <p><strong>Nombre:</strong> Albornoz Silva, Alejo Miguel</p>
            <p><strong>Comisión:</strong> C2</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("Información del Alumno")
mostrar_informacion_alumno()

# Instrucciones y área de carga del archivo
st.sidebar.header("Opciones")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv", help="Límite 200MB por archivo")

# Procesar archivo subido
if uploaded_file:
    # Leer el archivo
    ventas_df = pd.read_csv(uploaded_file)

    # Crear columna de fecha
    ventas_df['Fecha'] = pd.to_datetime(
        ventas_df[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1)
    )

    # Seleccionar sucursal en la barra lateral
    sucursales = ventas_df["Sucursal"].unique()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccione una sucursal", ["Todas"] + list(sucursales))

    # Mostrar título dependiendo de la sucursal seleccionada
    if sucursal_seleccionada == "Todas":
        st.title("Datos de todas las sucursales")
    else:
        st.title(f"{sucursal_seleccionada}")

    # Filtrar datos por sucursal si se seleccionó una
    if sucursal_seleccionada != "Todas":
        ventas_df = ventas_df[ventas_df["Sucursal"] == sucursal_seleccionada]

    # Obtener lista de productos
    productos = ventas_df["Producto"].unique()

    def calcular_metricas(producto_df):
        # Calcular precio promedio
        producto_df['Precio_promedio'] = producto_df['Ingreso_total'] / producto_df['Unidades_vendidas']
        precio_promedio = producto_df['Precio_promedio'].mean()

        # Calcular margen promedio
        producto_df['Ganancia'] = producto_df['Ingreso_total'] - producto_df['Costo_total']
        producto_df['Margen'] = (producto_df['Ganancia'] / producto_df['Ingreso_total']) * 100
        margen_promedio = producto_df['Margen'].mean()

        # Calcular unidades vendidas totales
        unidades_vendidas = producto_df['Unidades_vendidas'].sum()

        # Calcular variaciones anuales
        metricas_anuales = producto_df.groupby('Año').agg({
            'Precio_promedio': 'mean',
            'Margen': 'mean',
            'Unidades_vendidas': 'sum'
        })

        # Calcular variaciones porcentuales
        variacion_precio = metricas_anuales['Precio_promedio'].pct_change().mean() * 100
        variacion_margen = metricas_anuales['Margen'].pct_change().mean() * 100
        variacion_unidades = metricas_anuales['Unidades_vendidas'].pct_change().mean() * 100

        return (precio_promedio, margen_promedio, unidades_vendidas,
                variacion_precio, variacion_margen, variacion_unidades)

    def mostrar_producto(producto_df, producto):
        with st.container(border=True):
            st.subheader(f"{producto}")
            
            # Crear dos columnas: métricas (25%) y gráfico (75%)
            col1, col2 = st.columns([0.25, 0.75])
            
            # Calcular métricas
            Precio_promedio, Margen_promedio, Unidades_vendidas, cambio_precio, cambio_margen, cambio_unidades = calcular_metricas(producto_df)
            
            # Columna izquierda - Métricas
            with col1:
                st.metric(
                    label="Precio Promedio",
                    value=f"${Precio_promedio:,.0f}".replace(",", "."),
                    delta=f"{cambio_precio:.2f}%"
                )
                st.metric(
                    label="Margen Promedio",
                    value=f"{Margen_promedio:.0f}%",
                    delta=f"{cambio_margen:.2f}%"
                )
                st.metric(
                    label="Unidades Vendidas", 
                    value=f"{Unidades_vendidas:,.0f}".replace(",", "."),
                    delta=f"{cambio_unidades:.2f}%"
                )
            
            # Columna derecha - Gráfico
            with col2:
                # Gráfico de evolución de ventas
                ventas_mensual = producto_df.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(ventas_mensual["Fecha"], ventas_mensual["Unidades_vendidas"], 
                       marker="o", label=producto)
                
                # Línea de tendencia
                z = np.polyfit(range(len(ventas_mensual)), ventas_mensual["Unidades_vendidas"], 1)
                p = np.poly1d(z)
                ax.plot(ventas_mensual["Fecha"], p(range(len(ventas_mensual))), 
                       "--", color="red", label="Tendencia")
                
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Unidades Vendidas")
                ax.set_ylim(bottom=0)
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                plt.close()

    # Recorrer cada producto y mostrar información
    for producto in productos:
        mostrar_producto(ventas_df[ventas_df["Producto"] == producto], producto)
else:
    st.info("Por favor, sube un archivo CSV desde la barra lateral.")
