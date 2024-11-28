import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-55533.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 55533')
        st.markdown('**Nombre:** Gonzalez Luciano')
        st.markdown('**Comisión:** C2')

def cargar_datos():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.sidebar.success("Archivo cargado correctamente")
        return datos
    else:
        return None

def cambiar_sucursal(datos, sucursal=None):
    if sucursal and sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]
    return datos

def graficar_evolucion(datos):   
    # Graficar evolución por producto
    unidades_vendidas_mensual = datos.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
    productos = datos['Producto'].unique()
    for producto in productos:
        datos_producto = unidades_vendidas_mensual[unidades_vendidas_mensual['Producto'] == producto] 
        evolucion_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

        # Crear la columna 'Fecha' correctamente
        evolucion_producto['Fecha'] = pd.to_datetime(
            evolucion_producto.rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1)[['year', 'month', 'day']]
        )
        evolucion_producto['Fecha_str'] = evolucion_producto['Fecha'].dt.strftime('%Y-%m')

        datos_filtrados = datos[datos['Producto'] == producto]

        # Calcular métricas
        datos_filtrados['Precio_promedio'] = datos_filtrados['Ingreso_total'] / datos_filtrados['Unidades_vendidas']
        promedio_precio = datos_filtrados['Precio_promedio'].mean()

        precios_por_año = datos_filtrados.groupby('Año')['Precio_promedio'].mean()
        variacion_precio = precios_por_año.pct_change().mean() * 100

        datos_filtrados['Ganancia'] = datos_filtrados['Ingreso_total'] - datos_filtrados['Costo_total']
        datos_filtrados['Margen'] = (datos_filtrados['Ganancia'] / datos_filtrados['Ingreso_total']) * 100
        margen_promedio = datos_filtrados['Margen'].mean()

        margen_anual = datos_filtrados.groupby('Año')['Margen'].mean()
        variacion_margen = margen_anual.pct_change().mean() * 100

        unidades_vendidas = datos_filtrados['Unidades_vendidas'].sum()
        ventas_por_año = datos_filtrados.groupby('Año')['Unidades_vendidas'].sum()
        variacion_unidades = ventas_por_año.pct_change().mean() * 100

        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evolucion_producto['Fecha'], evolucion_producto['Unidades_vendidas'], label=f"{producto}")

        # Línea de tendencia
        z_producto = np.polyfit(range(len(evolucion_producto)), evolucion_producto['Unidades_vendidas'], 1)
        p_producto = np.poly1d(z_producto)
        ax.plot(evolucion_producto['Fecha'], p_producto(range(len(evolucion_producto))), linestyle='--', color='red', label="Tendencia")

        ax.grid(True, linestyle='--', alpha=0.7)

        # Configurar las etiquetas del eje x
        ax.set_xticks(evolucion_producto['Fecha'])                  

        etiquetas = []
        for fecha in evolucion_producto['Fecha']:
            if fecha.month == 1:
                etiquetas.append(fecha.strftime('%Y'))
            else:
                etiquetas.append('')

        ax.set_xticklabels(etiquetas)
        plt.xticks(rotation=45)

        # Configurar el eje y
        max_y = evolucion_producto['Unidades_vendidas'].max()
        if max_y > 10000:
            ax.set_yticks(np.arange(0, max_y + 10000, 10000))
        ax.set_ylim(bottom=0)  # Asegurar que el eje y comience en 0

        # Configurar títulos y leyendas
        ax.set_title("Evolución de Ventas Mensual")
        ax.set_xlabel("Año-Mes")
        ax.set_ylabel("Unidades vendidas")
        ax.legend(title="Producto")

        # Mostrar información a la izquierda
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader(f"{producto}")
                st.metric(label="Precio Promedio", value=f"${promedio_precio:,.0f}".replace(',', '.'), delta=f"{variacion_precio:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:,.0f}%".replace(',', '.'), delta=f"{variacion_margen:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(',', '.'), delta=f"{variacion_unidades:.2f}%")
            with col2:
                st.pyplot(fig)

def main():
    datos = cargar_datos()
    if datos is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
    else:
        sucursal = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(datos["Sucursal"].unique()))
        datos_filtrados = cambiar_sucursal(datos, sucursal)
        if sucursal != "Todas":
            st.header(f"Datos de {sucursal}")
        else:
            st.title("Datos de Todas las Sucursales")
        graficar_evolucion(datos_filtrados)

if __name__ == "__main__":
    main()