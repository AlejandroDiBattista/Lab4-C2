import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#Crear Red Neuronal
class VentasNN(nn.Module):
    def __init__(self, capa_entrada, capa_oculta, capa_salida):
        super(VentasNN, self).__init__()
        self.capa_entrada = nn.Linear(capa_entrada, capa_oculta)
        self.capa_oculta = nn.Linear(capa_oculta, capa_salida)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.capa_entrada(x))
        x = torch.relu(self.capa_oculta(x))
        return x
    
data = pd.read_csv('ventas.csv')

st.sidebar.header("Parámetros de la Red Neuronal")

# Parámetros de entrada en el panel izquierdo
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_size = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)

# Convertir los datos en tensores
X = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)

xnew = np.linspace(data["dia"].min(), data["dia"].max(), 300)
spl = make_interp_spline(data["dia"], data["ventas"], k=3)
y_smooth = spl(xnew)

st.subheader("Datos de Ventas Diarias")
fig, ax = plt.subplots()

ax.plot(xnew, y_smooth, color="red", label="Curva de Ajuste")  # Curva de ajuste
ax.scatter(data["dia"], data["ventas"], color="blue", label="Datos Reales")   # Puntos de datos reales
plt.xlabel("Dia del mes")
plt.ylabel("Ventas") 
ax.set_title("Ventas por Día del Mes")    
ax.legend() 
st.pyplot(fig)

if st.sidebar.button("Entrenar"):
    dias = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
    ventas = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)

    model = VentasNN(capa_entrada=1, capa_oculta=hidden_size, capa_salida=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Mostrar barra de progreso
    progress_bar = st.sidebar.progress(0)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(dias)
        loss = criterion(predictions, ventas)
        loss.backward()
        optimizer.step()

        # Guardar el historial de pérdida para el gráfico
        loss_history.append(loss.item())

        # Actualizar barra de progreso
        if epoch % (epochs // 100) == 0:  # Actualizar cada 1%
            progress_bar.progress(epoch / epochs)

    st.sidebar.success("Entrenamiento completado")

    # Mostrar gráfico de la pérdida
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(range(epochs),loss_history, color="green")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")  # Pérdida es el error medio cuadrático (MSE)
    ax_loss.set_title("Función de Pérdida")
    st.sidebar.pyplot(fig_loss)
    plt.show()