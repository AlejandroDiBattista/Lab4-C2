import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

st.sidebar.header('Parámetros de Entrenamiento')
learning_rate = st.sidebar.slider('Aprendizaje', 0.0, 1.0, 0.01, 0.01)
epochs = st.sidebar.number_input('Repeticiones', min_value=10, max_value=10000, value=1000, step=10)
hidden_neurons = st.sidebar.number_input('Neuronas Capa Oculta', min_value=1, max_value=100, value=10, step=1)


data = pd.read_csv('ventas.csv')

if st.sidebar.button('Entrenar'):
    x = data[['dia']].values
    y = data[['ventas']].values

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    x_mean = x_tensor.mean()
    x_std = x_tensor.std()
    x_norm = (x_tensor - x_mean) / x_std

    y_mean = y_tensor.mean()
    y_std = y_tensor.std()
    y_norm = (y_tensor - y_mean) / y_std

    class Net(nn.Module):
        def __init__(self, hidden_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(1, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = Net(hidden_neurons)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    progress_bar = st.progress(0)
    loss_list = []

    for epoch in range(int(epochs)):
        outputs = model(x_norm)
        loss = criterion(outputs, y_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

    st.success('Entrenamiento exitoso')

    fig1, ax1 = plt.subplots()
    ax1.plot(range(int(epochs)), loss_list, color='green', label='Pérdidas')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Evolución de la Función de Costo')
    ax1.legend()
    st.pyplot(fig1)

    with torch.no_grad():
        predicted = model(x_norm)
        predicted = predicted * y_std + y_mean  


    if len(predicted) == len(data['dia']):
        fig2, ax2 = plt.subplots()
        ax2.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
        ax2.plot(data['dia'], predicted.numpy(), color='red', label='Curva de Ajuste')
        ax2.set_xlabel('Día del Mes')
        ax2.set_ylabel('Ventas')
        ax2.set_title('Estimación de Ventas Diarias')
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.error("Las dimensiones de las predicciones y los días no coinciden.")

else:
    fig, ax = plt.subplots()
    ax.scatter(data['dia'], data['ventas'], color='blue')
    ax.set_xlabel('Día del Mes')
    ax.set_ylabel('Ventas')
    ax.set_title('Ventas Diarias')
    st.pyplot(fig)
