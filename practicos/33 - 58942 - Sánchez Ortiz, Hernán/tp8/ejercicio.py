import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

    if archivo:
        datos = pd.read_csv(archivo)
        st.write("Datos cargados exitosamente!")
        st.dataframe(datos.head())

        sucursales = datos['sucursal'].unique()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar sucursal", ["Todas"] + list(sucursales))

        if sucursal_seleccionada != "Todas":
            datos = datos[datos['sucursal'] == sucursal_seleccionada]
        
        datos['ingreso_total'] = datos['precio'] * datos['unidades_vendidas']
        datos['costo_total'] = datos['costo_unitario'] * datos['unidades_vendidas']

        resumen = datos.groupby('producto').agg(
            unidades_vendidas=('unidadaes_vendidas', 'sum'),
            precio_promedio=('ingreso_total', lambda x: x.sum() / datos['unidades_vendidas'].sum()),
            margen_promedio=('ingreso_total', lambda x: ((x.sum() - datos['costo_total'].sum()) / x.sum()))
        ).reset_index()

        st.write("Resumen por producto:")
        st.dataframe(resumen)

        datos['mes'] = pd.to_datetime(datos['fecha']).dt.to_period('M')
        ventas_mensuales = datos.groupby('mes').agg(unidades_vendidas=('unidades_vendidas', 'sum')).reset_index()

        X = np.arange(len(ventas_mensuales)).reshape(-1, 1)
        y = ventas_mensuales['unidades_vendidas']
        modelo = LinearRegression()
        modelo.fit(X, y)
        tendencia = modelo.predict(X)

        plt.figure(figsize=(10, 6))
        plt.plot(ventas_mensuales['mes'].astype(str), ventas_mensuales['unidades_vendidas'], label="Ventas", marker="o")
        plt.plot(ventas_mensuales['mes'].astype(str), tendencia, color='red', linestyle="--", label="Tendencia")
        plt.xtics(rotation=45)
        plt.title("Evolucion de Ventas")
        plt.xlabel("Mes")
        plt.ylabel("Unidades Vendidas")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()