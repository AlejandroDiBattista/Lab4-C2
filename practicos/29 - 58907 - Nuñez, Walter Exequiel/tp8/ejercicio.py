import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58907-hfgro4jrzbyjfb3cuptuj4.streamlit.app/'

st.title("Datos del Alumno")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58907')
        st.markdown('**Nombre:** Nuñez Walter Exequiel')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Título de la aplicación
st.subheader("Por favor, sube un archivo CSV desde la barra lateral izquierda")

# Subir archivo CSV (en la barra lateral)
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV con los datos de ventas", type="csv")

if uploaded_file is not None:
    # Cargar los datos
    data = pd.read_csv(uploaded_file)
    
    # Filtrar por Sucursal
    sucursal = st.sidebar.selectbox("Selecciona una sucursal (o muestra todas):", 
                                    options=["Todas"] + data['Sucursal'].unique().tolist())
    
    if sucursal != "Todas":
        data = data[data['Sucursal'] == sucursal]

    # Título dinámico basado en la sucursal seleccionada
    if sucursal == "Todas":
        st.header("Información Detallada de Todas las Sucursales")
    else:
        st.title(f"Datos de {sucursal}")

    # Calcular estadísticas por producto
    with st.container(border=True):  
        stats = data.groupby('Producto').agg({
            'Unidades_vendidas': 'sum',
            'Ingreso_total': 'sum',
            'Costo_total': 'sum'
        }).reset_index()

        # Cálculo del precio promedio y margen
        stats['Precio_promedio'] = stats['Ingreso_total'] / stats['Unidades_vendidas']
        stats['Margen_promedio'] = (stats['Ingreso_total'] - stats['Costo_total']) / stats['Ingreso_total']

        # Mostrar información y gráficos de cada producto
        
        for index, row in stats.iterrows():
            # Crear dos columnas: información a la izquierda, gráfico a la derecha
            col1, col2 = st.columns([1, 3])

            # Columna 1: Información del producto
            with col1:
                st.subheader(f"{row['Producto']}")
                st.markdown("**Precio Promedio**")
                st.write(f"${row['Precio_promedio']:.2f}")

                st.markdown("**Margen Promedio**")
                st.write(f"{row['Margen_promedio']:.2%}")

                st.markdown("**Unidades Vendidas**")
                st.write(f"{row['Unidades_vendidas']}")

            # Columna 2: Gráfico de evolución de ventas
            with col2:
                producto_data = data[data['Producto'] == row['Producto']]
                producto_data['Periodo'] = producto_data['Año'].astype(str) + "-" + producto_data['Mes'].astype(str).str.zfill(2)
                ventas_mensuales = producto_data.groupby('Periodo')['Unidades_vendidas'].sum().reset_index()

                # Crear el gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=ventas_mensuales, x='Periodo', y='Unidades_vendidas', ax=ax, label=row['Producto'])

                # Añadir línea de tendencia
                x = np.arange(len(ventas_mensuales))  # Crear un eje numérico para los meses
                z = np.polyfit(x, ventas_mensuales['Unidades_vendidas'], 1)  # Ajuste lineal
                p = np.poly1d(z)  # Crear la función de tendencia
                ax.plot(ventas_mensuales['Periodo'], p(x), linestyle="--", color="red", label="Tendencia")

                # Líneas horizontales para cada mes y cada año
                ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)  # Líneas horizontales
                ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.5)  # Líneas verticales

                # Configurar el eje X para mostrar solo los años
                ventas_mensuales['Año'] = ventas_mensuales['Periodo'].str[:4]  # Extraer solo el año
                ax.set_xticks(range(0, len(ventas_mensuales), 12))  # Cada 12 meses (cada año)
                ax.set_xticklabels(ventas_mensuales['Año'][::12])  # Solo mostrar el año

                # Título y etiquetas
                ax.set_title(f"Evolución de Ventas")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()

                st.pyplot(fig)