import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://hernansanchezortiz-tp8.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.942')
        st.markdown('**Nombre:** Hernán Sánchez Ortiz')
        st.markdown('**Comisión:** C2')

def graficar_ventas(datos):
    unidades_vendidas_mensual = datos.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()

    productos = datos['Producto'].unique()

    for producto in productos:
        datos_producto = unidades_vendidas_mensual[unidades_vendidas_mensual['Producto'] == producto]
        
        datos_producto['Fecha'] = pd.to_datetime(datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str) + '-01')

        datos_filtrados = datos[datos['Producto'] == producto]
        datos_filtrados['Precio_promedio'] = datos_filtrados['Ingreso_total'] / datos_filtrados['Unidades_vendidas']
        precio_promedio = datos_filtrados['Precio_promedio'].mean()

        datos_filtrados['Margen_promedio'] = (datos_filtrados['Ingreso_total'] - datos_filtrados['Costo_total']) / datos_filtrados['Ingreso_total']
        margen_promedio = datos_filtrados['Margen_promedio'].mean()

        precio_promedio_anual = datos_filtrados.groupby('Año')['Precio_promedio'].mean()
        variacion_precio = precio_promedio_anual.pct_change().mean() * 100

        margen_promedio_anual = datos_filtrados.groupby('Año')['Margen_promedio'].mean()
        variacion_margen = margen_promedio_anual.pct_change().mean() * 100

        unidades_vendidas = datos_filtrados['Unidades_vendidas'].sum()
        unidades_vendidas_anual = datos_filtrados.groupby('Año')['Unidades_vendidas'].sum()
        variacion_unidades = unidades_vendidas_anual.pct_change().mean() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], label=f"{producto}")

        z_producto = np.polyfit(range(len(datos_producto)), datos_producto['Unidades_vendidas'], 1)
        p_producto = np.poly1d(z_producto)
        ax.plot(datos_producto['Fecha'], p_producto(range(len(datos_producto))), linestyle='--', color='red', label="Tendencia")
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xticks(datos_producto['Fecha'])
        etiquetas = [fecha.strftime('%Y') if fecha.month == 1 else '' for fecha in datos_producto['Fecha']]
        ax.set_xticklabels(etiquetas)

        max_y = datos_producto['Unidades_vendidas'].max()
        if max_y > 10000:
            ax.set_yticks(np.arange(0, max_y + 10000, 10000))

        ax.set_title(f"Evolución de Ventas mensual - {producto}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades vendidas")
        ax.legend(title="Producto")

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader(f"{producto}")
                st.metric(label="Precio Promedio", value=f"${int(precio_promedio):,}".replace(',', '.'), delta=f"{variacion_precio:,.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:,.2f}%", delta=f"{variacion_margen:,.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,}".replace(',', '.'), delta=f"{variacion_unidades:,.2f}%")
            with col2:
                st.pyplot(fig)

def main():
    st.sidebar.title("Cargar archivo de datos")
    uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

    mostrar_informacion_alumno()
    if uploaded_file is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    else:
        datos = pd.read_csv(uploaded_file)

        sucursal = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(datos["Sucursal"].unique()))
        if sucursal != "Todas":
            datos_filtrados = datos[datos["Sucursal"] == sucursal]
            st.title(f"Datos de {sucursal}")
        else:
            datos_filtrados = datos
            st.title("Datos de Todas las Sucursales")

        graficar_ventas(datos_filtrados)

if __name__ == "__main__":
    main()
