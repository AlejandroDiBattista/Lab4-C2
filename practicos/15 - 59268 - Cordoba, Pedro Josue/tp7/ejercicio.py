import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Cargar el archivo de ventas
ventas_df = pd.read_csv("ventas.csv")

# Definir la red neuronal
class SalesNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(SalesNet, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

# Función de entrenamiento
def train_model(model, data, targets, lr, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_values = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        
        # Actualizar barra de progreso
        progress_bar.progress((epoch + 1) / epochs)
    
    return model, loss_values

# Configuración de parámetros
st.sidebar.title("Parámetros de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en la capa oculta", 1, 100, 5)

# Botón de entrenamiento
if st.sidebar.button("Entrenar"):
    # Preparación de datos
    days = torch.tensor(ventas_df['dia'].values, dtype=torch.float32).view(-1, 1)
    sales = torch.tensor(ventas_df['ventas'].values, dtype=torch.float32).view(-1, 1)

    # Inicialización de la red neuronal
    model = SalesNet(hidden_neurons)
    progress_bar = st.progress(0)
    
    # Entrenamiento
    model, loss_values = train_model(model, days, sales, learning_rate, epochs)
    
    # Mostrar mensaje de éxito
    st.success("Entrenamiento exitoso")
    
    # Gráfico de la función de costo
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_values, color='green')
    ax_loss.set_xlabel('Época')
    ax_loss.set_ylabel('Pérdida')
    st.pyplot(fig_loss)
    
    # Gráfico de predicciones
    model.eval()
    with torch.no_grad():
        predicted_sales = model(days).numpy()
        
    fig_sales, ax_sales = plt.subplots()
    ax_sales.scatter(ventas_df['dia'], ventas_df['ventas'], color='blue', label="Datos Reales")
    ax_sales.plot(ventas_df['dia'], predicted_sales, color='red', label="Curva de Ajuste")
    ax_sales.set_xlabel('Día del Mes')
    ax_sales.set_ylabel('Ventas')
    ax_sales.legend()
    st.pyplot(fig_sales)

