import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# URL = 'https://tp8labivc2-dmckrlsyenqfdibnjlmyvs.streamlit.app/';
# Cargo datos
@st.cache_data
def cargar_datos(file):
        return pd.read_csv(file);
    
# Título de la aplicación
st.title("Info totalidad de sucursales");

# Información personal
def informacion_alumno():
    with st.container():
        st.markdown("**Comisión:** C2");
        st.markdown("**Legajo:** 58.731");
        st.markdown("**Nombre:** Nicolás Álvarez");

informacion_alumno();

# Subir archivos CSV
st.sidebar.header("Carga de archivo")
archivo_gaseosas = st.sidebar.file_uploader("Carga el archivo de gaseosas", type=['csv'])
archivo_vinos = st.sidebar.file_uploader("Carga el archivo de vinos", type=['csv'])

if archivo_gaseosas and archivo_vinos:
    # Cargo los datos
    gaseosas = cargar_datos(archivo_gaseosas)
    vinos = cargar_datos(archivo_vinos)

    if gaseosas is not None and vinos is not None:
        # Selección de tipo de producto
        tipo_producto = st.sidebar.selectbox("Selecciona (Vino o Gseosas)", ["Gaseosas", "Vinos"])

        # Selección de sucursal
        sucursales = ['Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
        sucursal_seleccionada = st.sidebar.selectbox("Selecciona la sede de la sucursal:", ['Todas'] + sucursales)

        # Filtrar los datos
        df = gaseosas if tipo_producto == "Gaseosas" else vinos
        if sucursal_seleccionada != 'Todas':
            df = df[df['Sucursal'] == sucursal_seleccionada]

        if not df.empty:
            # Crear columna de fecha y cálculos con NumPy
            df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + "-" + df['Mes'].astype(str) + "-01", errors='coerce')
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
                st.subheader(f"{row['Producto']}")

                # Crear columnas para informar datos
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}", f"{row['Variacion_precio']:.2f}%")
                col2.metric("Margen Promedio", f"{row['Margen_promedio']:.2f}%", f"{row['Variacion_margen']:.2f}%")
                col3.metric("Unidades Vendidas", f"{int(row['Unidades_vendidas']):,}")

                # Filtrar los datos por producto
                producto_df = df[df['Producto'] == row['Producto']]

                # Creo el gráfico de evolución de ventas
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(producto_df['Fecha'], producto_df['Unidades_vendidas'], label=row['Producto'], color='blue')
                ax.plot(
                    producto_df['Fecha'], 
                    producto_df['Unidades_vendidas'].rolling(3).mean(), 
                    label='Tendencia', 
                    color='green', 
                    linestyle='--'
                )
                ax.set_title("Evolución de Ventas Mensual")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.warning("No se encontraron datos")
else:
    st.warning("Sin ambos archivos CSV no se puede continuar")
