import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Título
st.title('Estimación de Ventas Diarias')

## Leer datos
data = pd.read_csv('ventas.csv')
st.write("Datos de Ventas Diarias")
st.dataframe(data)

## Preparar los datos
X = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)

## Normalizar los datos
X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
X_norm = (X - X_min) / (X_max - X_min)
y_norm = (y - y_min) / (y_max - y_min)

## Sidebar
st.sidebar.header('Parámetros de la Red Neuronal')

learning_rate = st.sidebar.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1)
epochs = st.sidebar.slider('Cantidad de Épocas', 10, 10000, 100)
hidden_neurons = st.sidebar.slider('Neurona en la capa oculta', 1, 100, 5)

## Entrenar la red neuronal
if st.sidebar.button('Entrenar'):
    # Convertir datos a tensores
    dias = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
    ventas_real = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)

    class RedNeuronal(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RedNeuronal, self).__init__()
            self.hidden = nn.Linear(input_size, hidden_size)
            self.output = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.hidden(x))
            x = self.output(x)
            return x

    # Instanciar la red neuronal
    modelo = RedNeuronal(input_size=1, hidden_size=hidden_neurons, output_size=1)

    # Definir la función de pérdida y el optimizador
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=learning_rate)

    # Para el historial de pérdidas
    historial_pérdida = []

    ## Entrenar Red Neuronal
    for epoch in range(epochs):

        predicciones = modelo(dias)
        perdida = criterio(predicciones, ventas_real)

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        # Para almacenar el valor de la pérdida en cada época
        historial_pérdida.append(perdida.item())

        # Para la barra de progreso
        if epoch % (epochs // 10) == 0:
            st.sidebar.progress((epoch + 1) / epochs)

    st.sidebar.success('Entrenamiento completado exitosamente.')

    # Mostrar gráfico de evolución de la pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(historial_pérdida)
    plt.title('Evolución de la Función de Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    st.pyplot(plt)

    # Graficar los datos de ventas y las predicciones
    with torch.no_grad():
        predicciones_finales = modelo(dias)
        predicciones_finales_invertidas = y_min + (y_max - y_min) * (predicciones_finales.numpy() + 1) / 2

    ## Graficar las predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
    plt.plot(data['dia'], predicciones_finales_invertidas, color='red', label='Predicciones')
    plt.title('Predicción de Ventas Diarias')
    plt.xlabel('Día')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)