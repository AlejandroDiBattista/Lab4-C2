import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

URL = 'https://parcial2tp8-gb3vponvosdgjtsjgknk4n.streamlit.app/'

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.title("Análisis de Ventas por Producto")

def show_student_info():
    with st.container():
        st.markdown("**Comisión:** C2")
        st.markdown("**Legajo:** 47.417")
        st.markdown("**Nombre:** Rosales José Ignacio")

show_student_info()

st.sidebar.header("Carga de archivo")
sales_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if sales_file:
    data = load_data(sales_file)

    if data is not None:
        required_columns = ['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
        if not all(col in data.columns for col in required_columns):
            st.error("El archivo debe contener las columnas: " + ", ".join(required_columns))
        else:
            selected_branch = st.sidebar.selectbox(
                "Selecciona una sucursal (o 'Todas' para ver todas)",
                ['Todas'] + data['Sucursal'].unique().tolist()
            )
            if selected_branch != 'Todas':
                data = data[data['Sucursal'] == selected_branch]

            data['Fecha'] = pd.to_datetime(
                data['Año'].astype(str) + "-" + data['Mes'].astype(str) + "-01", 
                errors='coerce'
            )
            data['average_price'] = np.where(
                data['Unidades_vendidas'] > 0, 
                data['Ingreso_total'] / data['Unidades_vendidas'], 
                0
            )
            data['average_margin'] = np.where(
                data['Ingreso_total'] > 0,
                (data['Ingreso_total'] - data['Costo_total']) / data['Ingreso_total'] * 100,
                0
            )

            summary = data.groupby('Producto').agg({
                'average_price': 'mean',
                'average_margin': 'mean',
                'Unidades_vendidas': 'sum'
            }).reset_index()

            st.header(f"Datos de {'Todas las Sucursales' if selected_branch == 'Todas' else selected_branch}")

            for _, row in summary.iterrows():
                st.subheader(f"{row['Producto']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Promedio", f"${row['average_price']:.2f}")
                col2.metric("Margen Promedio", f"{row['average_margin']:.2f}%")
                col3.metric("Unidades Vendidas", f"{int(row['Unidades_vendidas']):,}")

                product_data = data[data['Producto'] == row['Producto']]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(
                    product_data['Fecha'], 
                    product_data['Unidades_vendidas'], 
                    label=row['Producto'], 
                    color='blue'
                )
                ax.plot(
                    product_data['Fecha'], 
                    product_data['Unidades_vendidas'].rolling(3).mean(), 
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
        st.warning("No se encontraron data")
else:
    st.info("Por favor, sube un archivo CSV para continuar")
