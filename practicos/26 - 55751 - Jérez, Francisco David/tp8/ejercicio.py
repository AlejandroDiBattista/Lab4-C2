import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Configurar la página
st.set_page_config(layout="centered", page_title="Análisis de Ventas", page_icon="📊")

# Estilo para los gráficos
plt.style.use("ggplot")



# Función para mostrar la información del alumno
def mostrar_informacion_alumno():
    st.sidebar.markdown("### **Información del alumno**")
    st.sidebar.markdown("- **Legajo:** 55.555")
    st.sidebar.markdown("- **Nombre:** Juan Pérez")
    st.sidebar.markdown("- **Comisión:** C1")

# Mostrar la información del alumno
mostrar_informacion_alumno()

# Cargar archivo CSV
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Cargar archivo CSV desde la barra lateral
st.sidebar.markdown("### **Cargar archivo CSV**")
archivo_csv = st.sidebar.file_uploader("", type=["csv"])

# Verificar si el archivo fue cargado y procesarlo
if archivo_csv is not None:
    st.title("📈 **Análisis de Ventas**")
    st.markdown("Cargando datos...")

    # Cargar datos
    df = cargar_datos(archivo_csv)
    
    # Mostrar una vista previa de los datos cargados
    st.markdown("### **Vista previa de los datos cargados**")
    st.dataframe(df.head(), use_container_width=True)

    # Filtrar por sucursal
    sucursales = ['Todas'] + df['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar una sucursal", sucursales)
    
    if sucursal_seleccionada != 'Todas':
        df = df[df['Sucursal'] == sucursal_seleccionada]

    # Convertir la columna 'Año' y 'Mes' en una sola columna de tipo datetime
    df['Fecha'] = pd.to_datetime(df[['Año', 'Mes']].astype(str).agg('-'.join, axis=1), format='%Y-%m')

    # Calcular métricas
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    
    # Agrupar datos por producto y fecha
    metricas_producto = df.groupby(['Producto', 'Fecha']).agg(
        Ingreso_total=('Ingreso_total', 'sum'),
        Costo_total=('Costo_total', 'sum'),
        Unidades_vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()

    # Mostrar métricas generales por producto
    resumen_producto = df.groupby('Producto').agg(
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean'),
        Unidades_vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()

    # Mostrar las métricas de cada producto
    for producto in resumen_producto['Producto']:
        st.markdown(f"### 🛍️ **Producto: {producto}**")

        valores_producto = resumen_producto[resumen_producto['Producto'] == producto].iloc[0]
        st.markdown(f"- **Precio Promedio:** ${valores_producto['Precio_promedio']:.2f}")
        st.markdown(f"- **Margen Promedio:** {valores_producto['Margen_promedio']*100:.2f}%")
        st.markdown(f"- **Unidades Vendidas:** {valores_producto['Unidades_vendidas']}")

        # Graficar la evolución de ventas con una línea de tendencia
        ventas_mensuales_producto = df[df['Producto'] == producto].groupby('Fecha')['Unidades_vendidas'].sum()

        fig, ax = plt.subplots(figsize=(6, 4))

        # Graficar unidades vendidas
        ax.plot(ventas_mensuales_producto.index, ventas_mensuales_producto.values, label='Unidades vendidas', color='#1f77b4', marker="o")

        # Calcular la tendencia con regresión lineal
        X = np.arange(len(ventas_mensuales_producto)).reshape(-1, 1)
        y = ventas_mensuales_producto.values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        # Graficar línea de tendencia
        ax.plot(ventas_mensuales_producto.index, y_pred, label='Tendencia', color='#ff7f0e', linestyle='--')

        # Personalizar gráfico
        ax.set_title(f"Evolución de ventas de {producto}", fontsize=12)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades Vendidas")
        ax.legend()
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=ventas_mensuales_producto.index.min(), right=ventas_mensuales_producto.index.max())

        # Añadir borde y ajustar espacio del gráfico
        fig.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Mostrar gráfico
        st.markdown("#### **Gráfico de evolución de ventas**")
        st.pyplot(fig)

else:
    st.title("📊 Análisis de Ventas")
    st.markdown("Por favor, sube un archivo CSV para comenzar el análisis.")
