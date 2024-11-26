import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://trabajopractico-bfr5kyfccfe9mx2ex9vjkk.streamlit.app/'
def cargar_datos():
    """Carga un archivo CSV subido por el usuario."""
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
    if archivo is not None:
        return pd.read_csv(archivo)
    return None
#
def calcular_metricas(df):
    """Calcula las métricas requeridas por producto."""
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    resumen = df.groupby('Producto').agg({
        'Precio_promedio': 'mean',
        'Margen_promedio': 'mean',
        'Unidades_vendidas': 'sum'
    }).reset_index()
    return resumen

def graficar_evolucion(df, producto):
    """Genera un gráfico de evolución de ventas con línea de tendencia."""
    df_producto = df[df['Producto'] == producto]
    if 'Año' not in df_producto.columns or 'Mes' not in df_producto.columns:
        st.error("Las columnas 'Año' y 'Mes' no están presentes.")
        return

    # Asegúrate de trabajar con una copia del DataFrame si 'df_producto' es un subset
    df_producto = df_producto.copy()

    # Modificación del DataFrame
    df_producto.loc[:, 'Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str), errors='coerce')

    # Eliminar filas con valores NaN en la columna 'Fecha'
    df_producto = df_producto.dropna(subset=['Fecha'])

    # Ordenar los valores según la columna 'Fecha'
    df_producto = df_producto.sort_values('Fecha')

    if df_producto.empty:
        st.error("No hay datos válidos para graficar.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_producto, x='Fecha', y='Unidades_vendidas', ax=ax, label='Unidades vendidas')
    z = np.polyfit(range(len(df_producto)), df_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_producto['Fecha'], p(range(len(df_producto))), linestyle='--', color='red', label='Tendencia')
    ax.set_title(f"Evolución de Ventas Mensual: {producto}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")  
    st.title("Análisis de Ventas por Sucursal")

   
    with st.sidebar:
        st.header("Información del Alumno")
        st.write("**Legajo:** 59.450")
        st.write("**Nombre:** Ignacio Skibski")
        st.write("**Comisión:** C2")
        
      
        st.subheader("Cargar archivo de datos")
        datos = cargar_datos()

       
        if datos is not None:
            sucursales = ["Todas"] + list(datos['Sucursal'].unique())
            seleccion_sucursal = st.selectbox("Seleccionar Sucursal", sucursales)

    if datos is not None:
      
        if seleccion_sucursal != "Todas":
            datos = datos[datos['Sucursal'] == seleccion_sucursal]

   
        st.header("Datos de Todas las Sucursales")
        resumen = calcular_metricas(datos)

        for index, row in resumen.iterrows():
            col1, col2 = st.columns([1, 2])  
            with col1:
                st.subheader(row['Producto'])
                st.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}")
                st.metric("Margen Promedio", f"{row['Margen_promedio']*100:.2f}%")
                st.metric("Unidades Vendidas", f"{row['Unidades_vendidas']:,}")
            with col2:
                graficar_evolucion(datos, row['Producto'])

if __name__ == "__main__":
    main()