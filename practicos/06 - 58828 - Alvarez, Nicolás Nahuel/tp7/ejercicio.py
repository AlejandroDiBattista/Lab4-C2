import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Cargar y procesar los datos
data = pd.read_csv('ventas.csv')
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values

# Normalización de los datos
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

# Convertir a tensores
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Definición de la red neuronal
class Net(nn.Module):
    def __init__(self, hidden_neurons):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)  # Capa oculta
        self.output = nn.Linear(hidden_neurons, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Activación ReLU en la capa oculta
        x = self.output(x)
        return x

# Configuración de Streamlit
st.title('Estimación de Ventas Diarias')

# Panel de entrada (izquierda)
col1, col2 = st.columns([1, 2])

with col1:
    learning_rate = st.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1)
    epochs = st.slider('Cantidad de Épocas', 10, 10000, 100)
    hidden_neurons = st.slider('Neuronas en la Capa Oculta', 1, 100, 5)
    train_button = st.button('Entrenar')

# Panel de resultados (derecha)
with col2:
    if train_button:
        # Inicializar el modelo, la función de pérdida y el optimizador
        model = Net(hidden_neurons)
        criterion = nn.MSELoss()  # Función de pérdida (error cuadrático medio)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Optimización

        loss_history = []
        progress_bar = st.progress(0)  # Barra de progreso inicializada

        # Entrenamiento del modelo
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                X_batch, y_batch = batch
                optimizer.zero_grad()  # Gradientes a cero
                y_pred = model(X_batch)  # Predicciones del modelo
                loss = criterion(y_pred.squeeze(), y_batch)  # Calcular la pérdida
                loss.backward()  # Retropropagación
                optimizer.step()  # Optimizar los pesos

            loss_history.append(loss.item())
            progress_bar.progress((epoch + 1) / epochs)  # Actualización de la barra de progreso

        st.success('Entrenamiento completado con éxito')

        # Gráfico de la evolución de la función de costo
        plt.figure(figsize=(8, 5))
        plt.plot(range(epochs), loss_history, label='Función de Costo')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Evolución de la Función de Costo')
        plt.legend()
        st.pyplot(plt)

        # Predicciones de la red neuronal
        model.eval()  # Modo de evaluación
        with torch.no_grad():
            predictions = model(X_tensor).numpy()  # Realizar predicciones

        # Desnormalizar las predicciones
        predictions = predictions * y.std() + y.mean()

        # Gráfico de ventas reales vs predicciones (línea para predicciones)
        plt.figure(figsize=(8, 5))
        plt.scatter(data['dia'], data['ventas'], color='blue', label='Datos reales')
        plt.plot(data['dia'], predictions, color='red', label='Predicción', linestyle='-', linewidth=2)  # Línea para predicciones
        plt.xlabel('Día')
        plt.ylabel('Ventas')
        plt.title('Predicción de Ventas Diarias')
        plt.legend()
        st.pyplot(plt)
