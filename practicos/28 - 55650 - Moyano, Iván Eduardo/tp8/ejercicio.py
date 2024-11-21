import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis de ventas", layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 55.650')
        st.markdown('**Nombre:** Moyano Iván Eduardo')
        st.markdown('**Comisión:** C2')

mostrar_informacion_alumno()

st.title("Análisis de Ventas por Producto")

uploaded_file = st.file_uploader("Sube tu archivo CSV con los datos de ventas", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    expected_columns = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
    if all(column in df.columns for column in expected_columns):
        sucursal = st.selectbox("Selecciona una sucursal", options=["Todas"] + list(df["Sucursal"].unique()))
        if sucursal != "Todas":
            df = df[df["Sucursal"] == sucursal]

        df_metrics = df.groupby("Producto").agg(
            Precio_promedio = ("Ingreso_total", lambda x: x.sum() / df.loc[df["Producto"] == x.name, "Unidades_vendidas"].sum()),
            Margen_promedio = ("Ingreso_total", lambda x: ((x.sum() - df.loc[df["Producto"] == x.name, "Costo_total"].sum()) / x.sum())),
            Unidades_vendidas = ("Unidades_vendidas", "sum")
        ).reset_index()

        st.markdown("### Evolución de Ventas Mensuales")
        if not (df["Año"].dtype in [int, np.int64] and df["Mes"].dtype in [int, np.int64]):
            df["Año"] = pd.to_numeric(df["Año"], errors= "coerce")
            df["Mes"] = pd.to_numeric(df["Mes"], errors = "coerce")
        
        df = df.dropna(subset = ["Año", "Mes"])

        df["Año"] = df["Año"].astype(int)
        df["Mes"] = df["Mes"].astype(int)

        df["Fecha"] = pd.to_datetime({'year': df["Año"], 'month': df["Mes"], 'day': 1}, errors= "coerce")

        df = df.dropna(subset= ["Fecha"])

        df_monthly = df.groupby("Fecha").agg(Unidades_vendidas= ("Unidades_vendidas", "sum")).reset_index()

        z = np.polyfit(range(len(df_monthly)), df_monthly["Unidades_vendidas"], 1)
        p = np.poly1d(z)

        plt.figure(figsize=(10,6))
        plt.plot(df_monthly["Fecha"], df_monthly["Unidades_vendidas"], label="Unidades_vendidas", marker= "o")
        plt.plot(df_monthly["Fecha"], p(range(len(df_monthly))), label="Tendencia", linestyle="--", color= "red")
        plt.xlabel("Mes")
        plt.ylabel("Unidades Vendidas")
        plt.title("Evolución de Ventas")
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("El archivo no tiene las columnas esperadas. Por favor, verifica el formato.")