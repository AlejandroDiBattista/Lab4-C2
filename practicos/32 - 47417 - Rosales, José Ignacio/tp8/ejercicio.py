import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Application URL
url = "https://rosalesjoseignaciotp8.streamlit.app/"

st.title("Sales Analysis by Producto")

uploaded_file = st.sidebar.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Preview of Uploaded Data")
    st.write(data.head())

    required_columns = ['Sucursal', 'Producto', 'Año', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
    if not all(col in data.columns for col in required_columns):
        st.error("The file must contain the columns: Sucursal, Producto, Año, Unidades_vendidas, Ingreso_total, Costo_total")
    else:
        Sucursal_option = st.sidebar.selectbox("Select a Sucursal (or 'All' to view all)", data['Sucursal'].unique().tolist() + ['All'])
        if Sucursal_option != 'All':
            data = data[data['Sucursal'] == Sucursal_option]

        Productos = data['Producto'].unique()

        grouped_data = data.groupby('Producto').agg(
            Average_price=('Ingreso_total', lambda x: x.sum() / data.loc[x.index, 'Unidades_vendidas'].sum()),
            Average_margin=('Ingreso_total', lambda x: (x.sum() - data.loc[x.index, 'Costo_total'].sum()) / x.sum()),
            Unidades_vendidas=('Unidades_vendidas', 'sum')
        ).reset_index()

        last_Mes = data[['Año', 'Mes']].max()  
        last_Año, last_Mes = last_Mes['Año'], last_Mes['Mes']
        if last_Mes == 1:
            previous_Mes = 12
            previous_Año = last_Año - 1
        else:
            previous_Mes = last_Mes - 1
            previous_Año = last_Año

        filtered_data = data[(data['Año'] < last_Año) | ((data['Año'] == last_Año) & (data['Mes'] < last_Mes))]
        
        filtered_grouped_data = filtered_data.groupby('Producto').agg(
            Average_price=('Ingreso_total', lambda x: x.sum() / filtered_data.loc[x.index, 'Unidades_vendidas'].sum()),
            Average_margin=('Ingreso_total', lambda x: (x.sum() - filtered_data.loc[x.index, 'Costo_total'].sum()) / x.sum()),
            Unidades_vendidas=('Unidades_vendidas', 'sum')
        ).reset_index()
    
        st.subheader("Summary by Producto")
        st.write(grouped_data)
        
        st.subheader("Sales Evolution by Mes")
        
        last_Mes_data = data[(data['Año'] == last_Año) & (data['Mes'] == last_Mes)]
        
        previous_Mes_data = data[(data['Año'] == previous_Año) & (data['Mes'] == previous_Mes)]
        
        units_last_Mes = last_Mes_data[['Producto', 'Unidades_vendidas']]
        
        units_previous_Mes = previous_Mes_data[['Producto', 'Unidades_vendidas']]

        for Producto in Productos:

            model = LinearRegression()
            label_encoder = LabelEncoder()
            
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    Producto_info = grouped_data[grouped_data['Producto'] == Producto].iloc[0]
                    Producto_info_last = filtered_grouped_data[filtered_grouped_data['Producto'] == Producto].iloc[0]
                    
                    ulm = units_last_Mes[units_last_Mes['Producto'] == Producto].values[0][1]
                    upm = units_previous_Mes[units_previous_Mes['Producto'] == Producto].values[0][1]
                    unit_change_percentage = (ulm - upm) / upm * 100 
                    margin_change_percentage = (Producto_info['Average_margin'] - Producto_info_last['Average_margin']) / Producto_info_last['Average_margin'] * 100
                    price_change_percentage = (Producto_info['Average_price'] - Producto_info_last['Average_price']) / Producto_info_last['Average_price'] * 100

                    st.title(Producto)
                    st.write(f"**Units Sold**: {Producto_info['Unidades_vendidas']:,d}")
                    if unit_change_percentage > 0:
                        st.markdown(f'<span style="color:green">⬆{unit_change_percentage:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{unit_change_percentage:.2f}%</span>', unsafe_allow_html=True)

                    st.write(f"**Average Margin**: {Producto_info['Average_margin']*100:.2f}%")
                    if margin_change_percentage > 0:
                        st.markdown(f'<span style="color:green">⬆{margin_change_percentage:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{margin_change_percentage:.2f}%</span>', unsafe_allow_html=True)

                    st.write(f"**Average Price**: ${Producto_info['Average_price']:,.2f}")
                    if price_change_percentage > 0:
                        st.markdown(f'<span style="color:green">⬆{price_change_percentage:.2f}%</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color:red">⬇{price_change_percentage:.2f}%</span>', unsafe_allow_html=True)

                with col2:   
                    fig, ax1 = plt.subplots(figsize=(12, 15))

                    Producto_sales = data.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
                    chart_data = Producto_sales[Producto_sales['Producto'] == Producto]
                    chart_data['Period'] = chart_data['Año'].astype(str) + '-' + chart_data['Mes'].astype(str)

                    chart_data['Period_encoded'] = label_encoder.fit_transform(chart_data['Period'])

                    X = chart_data['Period_encoded'].values.reshape(-1, 1)
                    y = chart_data['Unidades_vendidas']

                    model.fit(X, y)
                    y_pred = model.predict(X)

                    ax1.grid(True)
                    ax1.plot(chart_data['Period'], y, label=Producto)
                    ax1.plot(X, y_pred, label='Trend (Linear Regression)', color='red', linestyle='--')

                    ax1.set_title('Sales by Producto')
                    ax1.set_xlabel('Period (Año-Mes)')
                    ax1.set_ylabel('Units Sold')

                    ax1.legend()
                    plt.xticks(rotation=90)
                    st.pyplot(plt)
            
    def mostrar_informacion_alumno():
        with st.container():
            st.markdown('**Student ID:** 47.417')
            st.markdown('**Name:** Rosales José Ignacio')
            st.markdown('**Group:** C2')

    mostrar_informacion_alumno()

else:
    st.subheader("Upload your sales CSV file")
    st.info("Please upload a CSV file to begin.")