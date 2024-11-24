import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evaluación: 7 | Recuperar para promocionar 
# 1. No publica la aplicación (-1)
# 2. No respeta el diseño (métricas y gráfico a la par) (-1)
# 3. Calcula mal el precio promedio (-1)


# Configuración de la página
st.set_page_config(page_title="Cargar archivo de datos", layout="wide")

def mostrar_informacion_alumno():
    st.markdown(
        """
        <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 5px;">
            <p><strong>Legajo:</strong> 58.829</p>
            <p><strong>Nombre:</strong> Albornoz Silva, Alejo Miguel</p>
            <p><strong>Comisión:</strong> C2</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("Información del Alumno")
mostrar_informacion_alumno()

# Instrucciones y área de carga del archivo
st.sidebar.header("Opciones")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv", help="Límite 200MB por archivo")

# Procesar archivo subido
if uploaded_file:
    # Leer el archivo
    ventas_df = pd.read_csv(uploaded_file)

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
        st.title(f"{sucursal_seleccionada}")

    # Filtrar datos por sucursal si se seleccionó una
    if sucursal_seleccionada != "Todas":
        ventas_df = ventas_df[ventas_df["Sucursal"] == sucursal_seleccionada]

    # Obtener lista de productos
    productos = ventas_df["Producto"].unique()

    # Recorrer cada producto y mostrar información
    for producto in productos:
        with st.container():
            st.subheader(f"{producto}")

            # Filtrar datos por producto
            producto_df = ventas_df[ventas_df["Producto"] == producto].copy()

            # Calcular métricas
            Unidades_vendidas = producto_df["Unidades_vendidas"].sum()
            Ingreso_total = producto_df["Ingreso_total"].sum()
            Costo_total = producto_df["Costo_total"].sum()
            Precio_promedio = Ingreso_total / Unidades_vendidas
            Margen_promedio = (Ingreso_total - Costo_total) / Ingreso_total * 100

            # Calcular precio promedio mensual y otras métricas mensuales
            producto_df['Precio_unitario'] = producto_df['Ingreso_total'] / producto_df['Unidades_vendidas']
            precio_mensual = producto_df.groupby('Fecha').agg({
                'Ingreso_total': 'sum',
                'Unidades_vendidas': 'sum',
                'Costo_total': 'sum'
            }).reset_index()
            precio_mensual['Precio_promedio'] = precio_mensual['Ingreso_total'] / precio_mensual['Unidades_vendidas']
            precio_mensual['Margen_promedio'] = (precio_mensual['Ingreso_total'] - precio_mensual['Costo_total']) / precio_mensual['Ingreso_total'] * 100
            precio_mensual = precio_mensual.sort_values('Fecha')

            # Calcular cambios
            if len(precio_mensual) >= 2:
                # Cambio en Precio Promedio
                precio_ultimo = precio_mensual['Precio_promedio'].iloc[-1]
                precio_anterior = precio_mensual['Precio_promedio'].iloc[-2]
                cambio_precio = (precio_ultimo - precio_anterior) / precio_anterior * 100

                # Cambio en Margen Promedio
                margen_ultimo = precio_mensual['Margen_promedio'].iloc[-1]
                margen_anterior = precio_mensual['Margen_promedio'].iloc[-2]
                cambio_margen = margen_ultimo - margen_anterior

                # Cambio en Unidades Vendidas
                unidades_ultimas = precio_mensual['Unidades_vendidas'].iloc[-1]
                unidades_anteriores = precio_mensual['Unidades_vendidas'].iloc[-2]
                cambio_unidades = (unidades_ultimas - unidades_anteriores) / unidades_anteriores * 100
            else:
                cambio_precio = 0
                cambio_margen = 0
                cambio_unidades = 0

            # Mostrar métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Precio Promedio",
                    value=f"${Precio_promedio:.2f}",
                    delta=f"{cambio_precio:.2f}%"
                )
            with col2:
                st.metric(
                    label="Margen Promedio",
                    value=f"{Margen_promedio:.2f}%",
                    delta=f"{cambio_margen:.2f}%"
                )
            with col3:
                st.metric(
                    label="Unidades Vendidas",
                    value=f"{Unidades_vendidas}",
                    delta=f"{cambio_unidades:.2f}%"
                )

            # Gráfico de la evolución de ventas
            ventas_mensual = producto_df.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()

            # Crear gráfico
            plt.figure(figsize=(10, 6))
            plt.plot(ventas_mensual["Fecha"], ventas_mensual["Unidades_vendidas"], marker="o", label=producto, color="blue")

            # Cálculo y trazado de la línea de tendencia
            z = np.polyfit(range(len(ventas_mensual)), ventas_mensual["Unidades_vendidas"], 1)
            p = np.poly1d(z)
            plt.plot(ventas_mensual["Fecha"], p(range(len(ventas_mensual))), "--", color="red", label="Tendencia")

            # Configuración de etiquetas y título
            plt.xlabel("Fecha")
            plt.ylabel("Unidades Vendidas")
            plt.title(f"Evolución de Ventas de {producto} - {sucursal_seleccionada}")
            plt.legend()

            # Activar la cuadrícula
            plt.grid(True)

            # Mostrar gráfico
            st.pyplot(plt)
            plt.close()
else:
    st.info("Por favor, sube un archivo CSV desde la barra lateral.")
