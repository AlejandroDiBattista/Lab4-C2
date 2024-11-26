import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = "https://tp8-58952-almiron.streamlit.app/"

# Configuración de la página
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

# Función para mostrar información del usuario
def mostrar_informacion_usuario():
    st.markdown(
        """
        <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 5px;">
            <p><strong>Legajo:</strong> 58952</p>
            <p><strong>Nombre:</strong> Maicol Leonel Almirón</p>
            <p><strong>Comisión:</strong> C2</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Mostrar información del usuario
st.title("Información del alumno")
mostrar_informacion_usuario()

# Instrucciones y área de carga del archivo
st.sidebar.header("Opciones")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv", help="Límite 200MB por archivo")

def calcular_resumen(df):
    """
    Calcula el resumen por producto con las métricas requeridas.
    """
    resumen = df.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: (x / np.where(df.loc[x.index, "Unidades_vendidas"] > 0, 
                                                                 df.loc[x.index, "Unidades_vendidas"], 1)).mean()),
        Margen_Promedio=("Ingreso_total", lambda x: ((x - df.loc[x.index, "Costo_total"]) / np.where(x > 0, x, 1)).mean()),
        Unidades_Vendidas=("Unidades_vendidas", "sum")
    ).reset_index()
    return resumen

def graficar_evolucion_ventas(df):
    """
    Genera un gráfico de evolución de ventas por mes.
    """
    ventas_mensuales = df.groupby("Fecha")['Unidades_vendidas'].sum().reset_index()
    ventas_mensuales = ventas_mensuales.set_index("Fecha").asfreq("MS", fill_value=0).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ventas_mensuales["Fecha"], ventas_mensuales["Unidades_vendidas"], marker="o", color="blue", label="Unidades Vendidas")

    # Agregar línea de tendencia
    if len(ventas_mensuales) > 1:
        z = np.polyfit(range(len(ventas_mensuales)), ventas_mensuales["Unidades_vendidas"], 1)
        p = np.poly1d(z)
        ax.plot(ventas_mensuales["Fecha"], p(range(len(ventas_mensuales))), "--", color="red", label="Tendencia")

    # Configuración del gráfico
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.set_title("Evolución de Ventas")
    ax.legend()
    ax.grid(True)

    return fig

# Procesar archivo subido
if uploaded_file:
    try:
        # Leer el archivo
        ventas_df = pd.read_csv(uploaded_file)

        # Verificar que las columnas necesarias existan
        required_columns = ['Año', 'Mes', 'Sucursal', 'Producto', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
        if not all(column in ventas_df.columns for column in required_columns):
            st.error("El archivo CSV no contiene las columnas necesarias.")
            st.stop()

        # Manejar valores nulos
        ventas_df.fillna(0, inplace=True)

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
            st.title(f"Sucursal: {sucursal_seleccionada}")

        # Filtrar datos por sucursal si se seleccionó una
        if sucursal_seleccionada != "Todas":
            ventas_df = ventas_df[ventas_df["Sucursal"] == sucursal_seleccionada]

        # Obtener lista de productos
        productos = ventas_df["Producto"].unique()

        # Recorrer cada producto y mostrar información
        for producto in productos:
            st.markdown(f"### Producto: {producto}")
            producto_df = ventas_df[ventas_df["Producto"] == producto]

            # Calcular métricas globales
            unidades_vendidas = producto_df["Unidades_vendidas"].sum()
            ingreso_total = producto_df["Ingreso_total"].sum()
            costo_total = producto_df["Costo_total"].sum()
            precio_promedio = ingreso_total / unidades_vendidas if unidades_vendidas > 0 else 0
            margen_promedio = (ingreso_total - costo_total) / ingreso_total * 100 if ingreso_total > 0 else 0

            # Calcular precio unitario
            producto_df['Precio_unitario'] = producto_df.apply(
                lambda row: row['Ingreso_total'] / row['Unidades_vendidas'] if row['Unidades_vendidas'] > 0 else 0, axis=1
            )

            # Agrupar por fecha y calcular métricas mensuales
            precio_mensual = producto_df.groupby('Fecha').agg({
                'Ingreso_total': 'sum',
                'Unidades_vendidas': 'sum',
                'Costo_total': 'sum'
            }).reset_index()

            # Calcular precio promedio y margen promedio
            precio_mensual['Precio_promedio'] = precio_mensual.apply(
                lambda row: row['Ingreso_total'] / row['Unidades_vendidas'] if row['Unidades_vendidas'] > 0 else 0, axis=1
            )
            precio_mensual['Margen_promedio'] = precio_mensual.apply(
                lambda row: (row['Ingreso_total'] - row['Costo_total']) / row['Ingreso_total'] * 100 if row['Ingreso_total'] > 0 else 0, axis=1
            )

            # Calcular cambios
            if len(precio_mensual) >= 2:
                cambio_precio = precio_mensual['Precio_promedio'].iloc[-1] - precio_mensual['Precio_promedio'].iloc[-2]
                cambio_margen = precio_mensual['Margen_promedio'].iloc[-1] - precio_mensual['Margen_promedio'].iloc[-2]
                cambio_unidades = precio_mensual['Unidades_vendidas'].iloc[-1] - precio_mensual['Unidades_vendidas'].iloc[-2]
            else:
                cambio_precio = 0
                cambio_margen = 0
                cambio_unidades = 0

            # Diseño de columnas
            col1, col2 = st.columns([1, 3])

            # Columna 1: Mostrar métricas
            with col1:
                st.metric("Precio Promedio", f"${precio_promedio:.2f}", f"{cambio_precio:.2f}")
                st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{cambio_margen:.2f}")
                st.metric("Unidades Vendidas", f"{unidades_vendidas}", f"{cambio_unidades:.2f}")

            # Columna 2: Gráfico de evolución de ventas
            with col2:
                fig = graficar_evolucion_ventas(producto_df)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV desde la barra lateral.")
# Pie de página
st.markdown("---")
st.markdown("**📊 Aplicación desarrollada con Streamlit** - Estilo personalizado para mayor claridad.")
