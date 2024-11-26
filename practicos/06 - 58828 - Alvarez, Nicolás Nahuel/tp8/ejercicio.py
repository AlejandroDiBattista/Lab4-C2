import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58828.streamlit.app/'

def calcular_cambio_porcentual(valor_actual, valor_anterior):
    if valor_anterior == 0 or pd.isna(valor_anterior):
        return None
    return ((valor_actual - valor_anterior) / valor_anterior) * 100

def mostrar_informacion_alumno():
    st.title("Información del Alumno:")
    st.markdown('Legajo: 58.828')
    st.markdown('Nombre: Nicolas Nahuel Alvarez')
    st.markdown('Comisión: C2')
    st.markdown("---")

mostrar_informacion_alumno()

st.sidebar.title("Panel de Control de Ventas")
st.sidebar.markdown("Cargar archivo CSV para análisis de ventas de productos.")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.title("Análisis de Ventas")
    st.markdown("Selecciona una sucursal para visualizar las tendencias de ventas de todos los productos.")

    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ['Todas'] + list(df['Sucursal'].unique()))

    if sucursal != 'Todas':
        df_sucursal = df[df['Sucursal'] == sucursal]
    else:
        df_sucursal = df

    productos = df_sucursal['Producto'].unique()

    for producto in productos:
        df_producto = df_sucursal[df_sucursal['Producto'] == producto]

        df_producto['Fecha'] = pd.to_datetime(
            df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01', format='%Y-%m-%d'
        )

        # Calcular precio promedio, margen promedio y unidades vendidas de forma global (sobre todo el periodo)
        precio_promedio = df_producto['Ingreso_total'].div(df_producto['Unidades_vendidas']).mean()
        margen_promedio = ((df_producto['Ingreso_total'] - df_producto['Costo_total']) / 
                           df_producto['Ingreso_total']).mean() * 100
        unidades_vendidas = df_producto['Unidades_vendidas'].sum()  # Este sí debe ser sum()

        # Calcular los valores del periodo anterior (para comparar)
        df_anterior = df_producto.shift(1)
        precio_promedio_anterior = df_anterior['Ingreso_total'].div(df_anterior['Unidades_vendidas']).mean()
        margen_promedio_anterior = ((df_anterior['Ingreso_total'] - df_anterior['Costo_total']) / 
                                   df_anterior['Ingreso_total']).mean() * 100
        unidades_anterior = df_anterior['Unidades_vendidas'].sum()  # Este sí debe ser sum()

        # Calcular los cambios porcentuales
        cambio_precio = calcular_cambio_porcentual(precio_promedio, precio_promedio_anterior)
        cambio_margen = calcular_cambio_porcentual(margen_promedio, margen_promedio_anterior)
        cambio_unidades = calcular_cambio_porcentual(unidades_vendidas, unidades_anterior)

        with st.container().container(border=True):
            st.subheader(f"{producto}")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{cambio_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{cambio_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{int(unidades_vendidas):,}", f"{cambio_unidades:.2f}%")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))

                promedio_unidades_mes = df_producto.groupby('Fecha')['Unidades_vendidas'].mean()
                ax.plot(promedio_unidades_mes.index, promedio_unidades_mes.values, label='Unidades Vendidas Promedio', color='blue', linewidth=2)

                z = np.polyfit(promedio_unidades_mes.index.map(lambda x: x.toordinal()), promedio_unidades_mes.values, 1)
                p = np.poly1d(z)
                ax.plot(promedio_unidades_mes.index, p(promedio_unidades_mes.index.map(lambda x: x.toordinal())),
                        label="Tendencia", color='red', linestyle='--', linewidth=2)

                ax.set_facecolor('white')
                ax.grid(color='black', linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)

                ax.set_xlabel("Fecha", fontsize=12)
                ax.set_ylabel("Unidades Vendidas", fontsize=12)

                # Ajustar el tamaño de los números del eje Y
                ax.tick_params(axis='y', labelsize=12)  # Aumenta el tamaño de la fuente de los números del eje Y

                def format_func(value, tick_number):
                    return '{:,.0f}'.format(value)

                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

                ax.set_title("Evolución de Ventas Mensuales", fontsize=14, fontweight="bold")

                ax.set_ylim(bottom=0)

                # Mantener las grillas mensuales pero solo mostrar años en las etiquetas
                ax.xaxis.set_major_locator(mdates.YearLocator())  # Marcas principales (años)
                ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Marcas menores (meses)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato de año
                ax.grid(True, which='both')  # Mostrar grillas para marcas principales y menores
                plt.xticks(rotation=0)

                st.pyplot(fig)