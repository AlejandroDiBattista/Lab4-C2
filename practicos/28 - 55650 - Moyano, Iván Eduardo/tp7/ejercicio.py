import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## Crear Red Neuronal
class RedNeuronal(nn.Module):
    def __init__(self, hidden_neurons):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
    

st.sidebar.header("Parametros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)

## Leer Datos

data = pd.read_csv('Ventas.csv')
dias = data['dia'].values
ventas = data['ventas'].values

st.subheader("Datos originales de Ventas")
fig, ax = plt.subplots()
ax.plot(dias, ventas, 'o-', label="Ventas reales")
ax.set_xlabel("Día")
ax.set_ylabel("Ventas")
st.pyplot(fig)

## Normalizar Datos

dias_norm = (dias - np.mean(dias)) / np.std(dias)
ventas_norm = (ventas - np.mean(ventas)) / np.std(ventas)

X = torch.tensor(dias_norm, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(ventas_norm, dtype=torch.float32).view(-1, 1)

## Entrenar Red Neuronal

if st.sidebar.button("Entrenar"):
    model = RedNeuronal(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    cost_history = []
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        cost_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    st.success("Entrenamiento finalizado con exito")


## Guardar Modelo

    torch.save(model.state_dict(), 'modelo_ventas.pth')
    st.write("Modelo guardado como 'modelo_ventas.pth'")

    st.subheader("Evolucion del Costo durante el Entrenamiento")
    fig, ax = plt.subplots()
    ax.plot(cost_history)
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Costo")
    st.pyplot(fig)


## Graficar Predicciones

    st.subheader("Prediccion de Ventas vs. Datos Reales")
    predictions = model(X).detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(dias, ventas, 'o-', label="Datos reales")
    ax.plot(dias, predictions * np.std(ventas) + np.mean(ventas), '-', label="Prediccion")
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)


st.title('Estimación de Ventas Diarias')