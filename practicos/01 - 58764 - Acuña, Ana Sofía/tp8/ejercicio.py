import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL = 'https://tp8-58764.streamlit.app/'

# Estilos CSS personalizados
def agregar_estilos():
    st.markdown(
        """
        <style>
        /* Estilo del sidebar */
        [data-testid="stSidebar"] {
            background-color: #1a1a1a; /* Negro */
            color: #f8c1d1; /* Rosa claro */
        }

        [data-testid="stSidebar"] h3 {
            color: #ff69b4; /* Rosa intenso */
        }

        /* Fondo principal */
        .main {
            background-color: #f2f0f5; /* Fondo gris claro */
            color: #1a1a1a; /* Negro */
        }

        /* Títulos */
        h1, h2, h3, h4 {
            color: #d63384; /* Rosa oscuro */
        }

        /* Tablas */
        .dataframe {
            border: 2px solid #d63384; /* Borde rosa oscuro */
            border-radius: 8px;
        }

        /* Botones */
        .stButton>button {
            background-color: #ff69b4; /* Rosa intenso */
            color: white;
            border-radius: 8px;
        }

        /* SelectBox y FileUploader */
        .stSelectbox, .stFileUploader {
            border: 2px solid #d63384; /* Rosa oscuro */
            border-radius: 8px;
        }

        /* Footer oculto */
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def mostrar_informacion_alumno():
    st.sidebar.markdown("### Información del Alumno")
    st.sidebar.markdown("**Legajo:** 58764")
    st.sidebar.markdown("**Nombre:** Acuña Ana Sofia")
    st.sidebar.markdown("**Comisión:** C2")

def main():
    # Aplicar estilos
    agregar_estilos()

    # Mostrar información del alumno en el sidebar
    mostrar_informacion_alumno()

    # Título en el área principal
    st.title("Análisis de Ventas")

    # Subir archivo desde el sidebar
    st.sidebar.subheader("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")

    if archivo is not None:
        # Cargar los datos
        df = pd.read_csv(archivo)

        # Selección de sucursales en el sidebar
        st.sidebar.subheader("Filtrar por Sucursal")
        sucursales = ["Todas"] + list(df['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            df = df[df['Sucursal'] == sucursal_seleccionada]

        # Mostrar datos filtrados
        st.subheader(f"Datos cargados: {sucursal_seleccionada}")
        st.dataframe(df)

        # Métricas por producto
        st.subheader("Métricas por producto")
        if not df.empty:
            resultados = df.groupby('Producto').agg(
                Precio_Promedio=("Ingreso_total", lambda x: (x / df.loc[x.index, 'Unidades_vendidas']).mean()),
                Margen_Promedio=("Ingreso_total", lambda x: ((x - df.loc[x.index, 'Costo_total']) / x).mean()),
                Unidades_Vendidas=("Unidades_vendidas", "sum")
            )
            st.dataframe(resultados)

            # Evolución de ventas mensuales
            st.subheader("Evolución de ventas mensuales")
            df['Fecha'] = pd.to_datetime({'year': df['Año'], 'month': df['Mes'], 'day': 1})
            df_ventas_mensuales = df.groupby('Fecha').agg(Ventas_Totales=("Unidades_vendidas", "sum")).reset_index()

            # Calcular la tendencia manualmente
            x = np.arange(len(df_ventas_mensuales))
            y = df_ventas_mensuales['Ventas_Totales']
            n = len(x)

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Fórmulas de mínimos cuadrados
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            tendencia = slope * x + intercept

            # Graficar la evolución
            plt.figure(figsize=(10, 6))
            plt.plot(df_ventas_mensuales['Fecha'], y, label='Ventas Totales', marker='o', color='#d63384')
            plt.plot(df_ventas_mensuales['Fecha'], tendencia, label='Tendencia', linestyle='--', color='black')
            plt.xlabel("Fecha")
            plt.ylabel("Unidades Vendidas")
            plt.title(f"Evolución de ventas mensuales: {sucursal_seleccionada}")
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("No hay datos para mostrar en esta sucursal.")

if __name__ == "__main__":
    main()
