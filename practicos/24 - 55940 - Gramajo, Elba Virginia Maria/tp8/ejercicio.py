
# url = ' http://192.168.100.31:8501'
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache
def cargar_datos(file):
    return pd.read_csv(file)

# Título principal
st.title("Cargar archivo de datos")

# Información del alumno
def mostrar_informacion_alumno():
    st.markdown("### Por favor, sube un archivo CSV desde la barra lateral.")
    st.text("Legajo: 55.940")
    st.text("Nombre: Gramajo Elba Virginia Mailen")
    st.text("Comisión: C2")

mostrar_informacion_alumno()

st.sidebar.header("Cargar archivo de datos")
archivo_gaseosas = st.sidebar.file_uploader("Sube el archivo de Gaseosas", type=['csv'])
archivo_vinos = st.sidebar.file_uploader("Sube el archivo de Vinos", type=['csv'])

if archivo_gaseosas and archivo_vinos:
    # Cargar los datos
    gaseosas = cargar_datos(archivo_gaseosas)
    vinos = cargar_datos(archivo_vinos)

    
    # Selección de tipo de producto
    tipo_producto = st.sidebar.selectbox("Selecciona el tipo de producto:", ["Gaseosas", "Vinos"])

    # Selección de sucursal
    sucursales = ['Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
    sucursal_seleccionada = st.sidebar.selectbox("Selecciona la sucursal:", ['Todas'] + sucursales)

    # Filtrar los datos
    df = gaseosas if tipo_producto == "Gaseosas" else vinos
    if sucursal_seleccionada != 'Todas':
        df = df[df['Sucursal'] == sucursal_seleccionada]


    # Crear columna de fecha y cálculos con NumPy
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + "-" + df['Mes'].astype(str) + "-01")
    df['Precio_promedio'] = np.where(df['Unidades_vendidas'] > 0, df['Ingreso_total'] / df['Unidades_vendidas'], 0)
    df['Margen_promedio'] = np.where(
        df['Ingreso_total'] > 0,
        (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total'] * 100,
        0
    )
    df['Variacion_precio'] = df.groupby('Producto')['Precio_promedio'].pct_change() * 100
    df['Variacion_margen'] = df.groupby('Producto')['Margen_promedio'].pct_change() * 100

    resumen = df.groupby('Producto').agg({
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean',
        'Unidades_vendidas': 'sum',
        'Variacion_precio': 'mean',
        'Variacion_margen': 'mean'
    }).reset_index()

    # Mostrar métricas y gráficos
    st.header(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

    for _, row in resumen.iterrows():
        st.divider()
        col1, col2 = st.columns([1, 3])  # Configurar proporción de columnas

        with col1:
            st.subheader(f"{row['Producto']}")
            st.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}", f"{row['Variacion_precio']:.2f}%")
            st.metric("Margen Promedio", f"{row['Margen_promedio']:.2f}%", f"{row['Variacion_margen']:.2f}%")
            st.metric("Unidades Vendidas", f"{int(row['Unidades_vendidas']):,}")

        with col2:
            # Filtrar los datos por producto
            producto_df = df[df['Producto'] == row['Producto']]

            # Crear gráfico de evolución de ventas
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(producto_df['Fecha'], producto_df['Unidades_vendidas'], label=row['Producto'], color='blue')
            ax.plot(
                producto_df['Fecha'],
                producto_df['Unidades_vendidas'].rolling(3).mean(),
                label='Tendencia',
                color='red',
                linestyle='--'
            )
            ax.set_title("Evolución de Ventas Mensual")
            ax.set_xlabel("Año-Mes")
            ax.set_ylabel("Unidades Vendidas")
            ax.legend()

            # Formatear eje x
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Mostrar menos etiquetas en el eje x
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

else:
    st.warning("Por favor, sube ambos archivos CSV para continuar.")

 