import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.942')
        st.markdown('**Nombre:** Hernán Sánchez Ortiz')
        st.markdown('**Comisión:** C2')

def main():
    mostrar_informacion_alumno()

    st.title("Analisis de Ventas")
    st.sidebar.header("Opciones")

    archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.success("Datos cargados exitosamente!")
        st.dataframe(datos.head())

        sucursales = datos['Sucursal'].unique()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(sucursales))

        if sucursal_seleccionada != "Todas":
            datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        
        datos['Precio_promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
        datos['Costo_total'] = datos['Precio_promedio'] * datos['Unidades_vendidas']

        resumen = datos.groupby('Producto').agg(
            unidades_vendidas=('Unidades_vendidas', 'sum'),
            precio_promedio=('Ingreso_total', lambda x: x.sum() / datos['Unidades_vendidas'].sum()),
            margen_promedio=('Ingreso_total', lambda x: ((x.sum() - datos['Costo_total'].sum()) / x.sum()))
        ).reset_index()

        st.write("Resumen por producto:")
        st.dataframe(resumen)

        datos['mes'] = pd.to_datetime(datos[['Año', 'Mes']].astype(str).agg('-'.join, axis=1), format='%Y-%m')
        ventas_mensuales = datos.groupby('mes').agg(unidades_vendidas=('Unidades_vendidas', 'sum')).reset_index()
        st.write("Ventas mensuales agrupadas:")
        st.dataframe(ventas_mensuales)

        X = np.arange(len(ventas_mensuales))
        y = ventas_mensuales['unidades_vendidas']
        n = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xx = np.sum(X**2)
        sum_xy = np.sum(X * y)
    
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
        b = (sum_y - m * sum_x) / n
        tendencia = m * X + b

        plt.figure(figsize=(10, 6))
        plt.plot(ventas_mensuales['mes'].astype(str), ventas_mensuales['unidades_vendidas'], label="Ventas", marker="o")
        plt.plot(ventas_mensuales['mes'].astype(str), tendencia, color='red', linestyle="--", label="Tendencia")
        plt.xticks(rotation=45)
        plt.title("Evolucion de Ventas")
        plt.xlabel("Mes")
        plt.ylabel("Unidades Vendidas")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()