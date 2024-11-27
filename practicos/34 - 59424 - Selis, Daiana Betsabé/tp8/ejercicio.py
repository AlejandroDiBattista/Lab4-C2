import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
#url = 'https://tp8-parcial2-daiana-selis.streamlit.app/'
url = 'https://tp8-parcial2-daiana-selis.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.424')
        st.markdown('**Nombre:** Daiana Selis')
        st.markdown('**Comisión:** C2')

st.title("Análisis de Ventas por Producto")

archivo = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    required_columns = ['Sucursal', 'Producto', 'Año','Unidades_vendidas','Ingreso_total','Costo_total']
    if not all (col in df.columns for col in required_columns):
        st.error("El archivo debe contener las columnas : Sucursal , Producto , Año, Unidades_vendiadas, Ingreso_total, Costo_total")
    else:
        sucursal_opcion = st.sidebar.selectbox("Selecciona la sucursal (o 'Todos' para ver todas)", ['Todos'] + df['Sucursal'].unique().tolist()  )
        if sucursal_opcion != 'Todos':
            df = df[df['Sucursal'] == sucursal_opcion]

        productos = df['Producto'].unique()

        df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']

        last_month = df[['Año', 'Mes']].max() 
        last_year, last_month = last_month['Año'], last_month['Mes']
        if last_month == 1:
            previous_month = 12
            previous_year = last_year - 1
        else:
            previous_month = last_month - 1
            previous_year = last_year
        
        st.subheader("Datos de " + (sucursal_opcion if sucursal_opcion != 'Todos' else 'Todas las Sucursales'))
        
        df_last_month = df[(df['Año'] == last_year) & (df['Mes'] == last_month)]
      
        df_previous_month = df[(df['Año'] == previous_year) & (df['Mes'] == previous_month)]
    
        units_last_month = df_last_month[['Producto', 'Unidades_vendidas']]
        
        units_previous_month = df_previous_month[['Producto', 'Unidades_vendidas']]

        for producto in productos:

            model = LinearRegression()
            label_encoder = LabelEncoder()
            
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.title(producto)
                    df_filtered = df[df["Producto"] == producto]
                    precio_promedio = df_filtered['Precio_promedio'].mean()

                    precio_promedio_anual = df_filtered.groupby('Año')['Precio_promedio'].mean()
                    variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

                    df_filtered['Ganancia'] = df_filtered['Ingreso_total'] - df_filtered['Costo_total']
                    df_filtered['Margen'] = (df_filtered['Ganancia'] / df_filtered['Ingreso_total']) * 100
                    margen_promedio = df_filtered['Margen'].mean()

                    margen_promedio_anual = df_filtered.groupby('Año')['Margen'].mean()
                    variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

                    unidades_promedio = df_filtered['Unidades_vendidas'].mean()
                    unidades_vendidas = df_filtered['Unidades_vendidas'].sum()

                    unidades_por_año = df_filtered.groupby('Año')['Unidades_vendidas'].sum()
                    variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

                    st.metric("Precio Promedio", f"${precio_promedio:,.0f}".replace(",","."), f"{variacion_precio_promedio_anual:.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:,.0f}%".replace(",","."), f"{variacion_margen_promedio_anual:.2f}%")
                    st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",","."), f"{variacion_anual_unidades:.2f}%")
                with col2:           
                    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))

                    ventas_producto = df.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
                    data = ventas_producto[ventas_producto['Producto'] == producto]
                    data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str)

                    data['Periodo_encoded'] = label_encoder.fit_transform(data['Periodo'])

                    X = data['Periodo_encoded'].values.reshape(-1, 1)
                    y = data['Unidades_vendidas']

                    model.fit(X, y)
                    y_pred = model.predict(X)

                    ax1.plot(data['Periodo'], y, label=producto)
                    ax1.plot(X, y_pred, label='Tendencia', color='red', linestyle='--')

                    ax1.set_title('Evolucion de ventas Mensual')
                    ax1.set_xlabel('Período (Año-Mes)')
                    ax1.set_ylabel('Unidades Vendidas')

                    
                    ax1.legend()
                    tick_positions = range(0, len(data), 12) 
                    tick_labels = data["Año"].iloc[tick_positions]  
                    plt.xticks(tick_positions, tick_labels)
                    ax = plt.gca()  
                    ax.set_xticks(range(len(X)))

                    ytick_positions = range(4)
                    ytick_labels = [(i + 1) * max(y) / 4 for i in ytick_positions]

                    plt.ylim(0, None)

                    plt.grid(which='both')

                    st.pyplot(plt)
            
    

else:
    mostrar_informacion_alumno()
    st.subheader("Sube tu archivo CSV de ventas")
    st.info("Por favor, sube un archivo CSV para comenzar.")

