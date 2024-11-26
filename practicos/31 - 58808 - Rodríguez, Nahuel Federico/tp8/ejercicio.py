import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58808.streamlit.app/'

# Mostrar información del alumno 
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.808')
        st.markdown('**Nombre:** Nahuel Federico Rodríguez')
        st.markdown('**Comisión:** C2')


# Función cambiar de sucursal
def cambiar_sucursal(datos, sucursal=None):
    if sucursal:
        datos = datos[datos["Sucursal"] == sucursal]

    return datos

# Función para graficar la evolución de ventas
def graficar_evolucion(datos):   
    # Graficar evolución por producto
    unidades_vendidas_mensual = datos.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
    productos = datos['Producto'].unique()
    for producto in productos:

        datos_producto = unidades_vendidas_mensual[unidades_vendidas_mensual['Producto'] == producto]
        evolucion_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
        evolucion_producto['Fecha'] = pd.to_datetime(evolucion_producto[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1))
        evolucion_producto['Fecha_str'] = evolucion_producto['Fecha'].dt.strftime('%Y-%m')

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
        ax.plot(evolucion_producto['Fecha'], evolucion_producto['Unidades_vendidas'], label=f"{producto}")
        
        # Línea de tendencia
        z_producto = np.polyfit(range(len(evolucion_producto)), evolucion_producto['Unidades_vendidas'], 1)
        p_producto = np.poly1d(z_producto)
        ax.plot(evolucion_producto['Fecha'], p_producto(range(len(evolucion_producto))), linestyle='--', color='red', label="Tendencia")
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_xticks(evolucion_producto['Fecha'])                  

        etiquetas = []
        for fecha in evolucion_producto['Fecha']:
            if fecha.month == 1:
                etiquetas.append(fecha.strftime('%Y'))
            else:
                etiquetas.append('')
        
        ax.set_xticklabels(etiquetas)
        
        max_y = evolucion_producto['Unidades_vendidas'].max()
        if max_y > 10000:
            ax.set_yticks(np.arange(0, max_y + 10000, 10000))

        ax.set_title(f"Evolucion de Ventas mensual")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades vendidas")
        ax.legend(title="Producto")
        
        # Mostrar información a la izquierda
        
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader(f"{producto}")
                st.metric(label="Precio Promedio", value=f"${int(precio_promedio):,}".replace(',', '.'), delta=f"{variacion_precio:,.2f}%")
                st.metric(label= "Margen Promedio", value=f"{margen_promedio:,.2f}%", delta = f"{variacion_margen:,.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,}".replace(',', '.'), delta=f"{variacion_unidades:,.2f}%")
            with col2:
                st.pyplot(fig)

# Carga de datos
def main():        
    st.sidebar.title("Opciones")
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type="csv")
    
    if uploaded_file is None:
            st.title("Por favor, sube un archivo CSV desde la barra lateral.")
            mostrar_informacion_alumno()

    else:
        datos = pd.read_csv(uploaded_file)                       


        sucursal = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(datos["Sucursal"].unique()))
        if sucursal != "Todas":
            datos_filtrados = cambiar_sucursal(datos, sucursal)
            st.title(f"Datos de {sucursal}")
        else:
            datos_filtrados = cambiar_sucursal(datos)
            st.title("Datos de Todas las Sucursales")

        graficar_evolucion(datos_filtrados)

if __name__ == "__main__":
    main()
