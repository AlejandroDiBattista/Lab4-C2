import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
#url = 'https://tp8-parcial2-daiana-selis.streamlit.app/'
url = 'https://tp8-parcial2-daiana-selis.streamlit.app/'

st.title("Análisis de Ventas por Producto")

archivo = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    
    st.subheader(f"Vista previa de los datos cargados")
    st.write(df.head())

    required_columns = ['Sucursal', 'Producto', 'Año','Unidades_vendidas','Ingreso_total','Costo_total']
    if not all (col in df.columns for col in required_columns):
        st.error("El archivo debe contener las columnas : Sucursal , Producto , Año, Unidades_vendiadas, Ingreso_total, Costo_total")
    else:
        sucursal_opcion = st.sidebar.selectbox("Selecciona la sucursal (o 'Todos' para ver todas)", df['Sucursal'].unique().tolist() + ['Todos'] )
        if sucursal_opcion != 'Todos':
            df = df[df['Sucursal'] == sucursal_opcion]

        productos = df['Producto'].unique()

        df_grouped = df.groupby('Producto').agg(
            Precio_promedio=('Ingreso_total', lambda x: x.sum() / df.loc[x.index, 'Unidades_vendidas'].sum()),
            Margen_promedio=('Ingreso_total', lambda x: (x.sum() - df.loc[x.index, 'Costo_total'].sum()) / x.sum()),
            Unidades_vendidas=('Unidades_vendidas', 'sum')
        ).reset_index()

        last_month = df[['Año', 'Mes']].max()  # Find the latest year and month
        last_year, last_month = last_month['Año'], last_month['Mes']
        if last_month == 1:
            previous_month = 12
            previous_year = last_year - 1
        else:
            previous_month = last_month - 1
            previous_year = last_year

        df_filtered = df[(df['Año'] < last_year) | ((df['Año'] == last_year) & (df['Mes'] < last_month))]
        # Aggregate the data until the last available month
        df_grouped_filtered = df_filtered.groupby('Producto').agg(
            Precio_promedio=('Ingreso_total', lambda x: x.sum() / df_filtered.loc[x.index, 'Unidades_vendidas'].sum()),
            Margen_promedio=('Ingreso_total', lambda x: (x.sum() - df_filtered.loc[x.index, 'Costo_total'].sum()) / x.sum()),
            Unidades_vendidas=('Unidades_vendidas', 'sum')
        ).reset_index()
    
        st.subheader("Resumen por Producto")
        st.write(df_grouped)
        
        st.subheader("Evolución de las ventas por Mes")
        
        df_last_month = df[(df['Año'] == last_year) & (df['Mes'] == last_month)]
        # Filter the data for the previous month
        df_previous_month = df[(df['Año'] == previous_year) & (df['Mes'] == previous_month)]
        # Get the sold units for each product in the last month
        units_last_month = df_last_month[['Producto', 'Unidades_vendidas']]
        # Get the sold units for each product in the previous month
        units_previous_month = df_previous_month[['Producto', 'Unidades_vendidas']]

        for producto in productos:

            model = LinearRegression()
            label_encoder = LabelEncoder()
            
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    producto_info = df_grouped[df_grouped['Producto'] == producto].iloc[0]
                    producto_info_last = df_grouped_filtered[df_grouped_filtered['Producto'] == producto].iloc[0]
                    
                    ulm = units_last_month[units_last_month['Producto'] == producto].values[0][1]
                    upm = units_previous_month[units_previous_month['Producto'] == producto].values[0][1]
                    porcentajeCambioUnidades = (ulm - upm) / upm * 100 
                    porcentajeCambioMargen = (producto_info['Margen_promedio'] - producto_info_last['Margen_promedio']) / producto_info_last['Margen_promedio'] * 100
                    porcentajeCambioPrecio = (producto_info['Precio_promedio'] - producto_info_last['Precio_promedio']) / producto_info_last['Precio_promedio'] * 100

                    st.title(producto)
                    st.write(f"**Unidades Vendidas**: {producto_info['Unidades_vendidas']:,d}")
                    if(porcentajeCambioUnidades > 0):
                        st.markdown(f'<span style="color:green">⬆{porcentajeCambioUnidades:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{porcentajeCambioUnidades:.2f}%</span>', unsafe_allow_html=True)

                    st.write(f"**Margen Promedio**: {producto_info['Margen_promedio']*100:.2f}%")
                    if(porcentajeCambioMargen > 0):
                        st.markdown(f'<span style="color:green">⬆{porcentajeCambioMargen:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{porcentajeCambioMargen:.2f}%</span>', unsafe_allow_html=True)

                    st.write(f"**Precio Promedio**: ${producto_info['Precio_promedio']:,.2f}")
                    if(porcentajeCambioPrecio > 0):
                        st.markdown(f'<span style="color:green">⬆{porcentajeCambioPrecio:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{porcentajeCambioPrecio:.2f}%</span>', unsafe_allow_html=True)
                with col2:           
                    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 15))

                    # 1. Ventas por producto a lo largo del tiempo
                    ventas_producto = df.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
                    data = ventas_producto[ventas_producto['Producto'] == producto]
                    data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str)

                    data['Periodo_encoded'] = label_encoder.fit_transform(data['Periodo'])

                    X = data['Periodo_encoded'].values.reshape(-1, 1)
                    y = data['Unidades_vendidas']

                    model.fit(X, y)
                    y_pred = model.predict(X)

                    ax1.grid(True)

                    ax1.plot(data['Periodo'], y, label=producto)
                    ax1.plot(X, y_pred, label='Tendencia (Regresión Lineal)', color='red', linestyle='--')

                    ax1.set_title('Ventas por Producto')
                    ax1.set_xlabel('Período (Año-Mes)')
                    ax1.set_ylabel('Unidades Vendidas')

                    # ax1.set_ylim([0, 10000])
                    ax1.legend()
                    plt.xticks(rotation=90)
                    st.pyplot(plt)
            
    def mostrar_informacion_alumno():
        with st.container(border=True):
            st.markdown('**Legajo:** 59.424')
            st.markdown('**Nombre:** Daiana Selis')
            st.markdown('**Comisión:** C2')

    mostrar_informacion_alumno()

else:
    st.subheader("Sube tu archivo CSV de ventas")
    st.info("Por favor, sube un archivo CSV para comenzar.")
