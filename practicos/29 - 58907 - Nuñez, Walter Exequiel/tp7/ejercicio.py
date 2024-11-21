import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

print(torch.__version__)


## Crear Red Neuronal
class VentasNet(nn.Module):
    def __init__(self):
        super(VentasNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Capa de entrada
        self.fc2 = nn.Linear(10, 1)  # Capa de salida
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

modelo = VentasNet()
## Leer Datos
data = {
    "dia": list(range(1, 31)),
    "ventas": [195, 169, 172, 178, 132, 123, 151, 127, 96, 110, 86, 82, 94, 60, 63,
               76, 69, 98, 77, 71, 134, 107, 120, 99, 126, 150, 136, 179, 173, 194]
}
df = pd.DataFrame(data)
st.write("Datos de ventas diarias:")
st.write(df)

## Normalizar Datos
ventas = df['ventas'].values.astype(float)
ventas_normalized = (ventas - ventas.min()) / (ventas.max() - ventas.min())
ventas_tensor = torch.tensor(ventas_normalized, dtype=torch.float32).view(-1, 1)
## Entrenar Red Neuronal
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

num_epochs = 500
dias_tensor = torch.arange(1, 31, dtype=torch.float32).view(-1, 1)

for epoch in range(num_epochs):
    modelo.train()
    optimizador.zero_grad()

    # Forward pass
    predicciones = modelo(dias_tensor)
    
    # Calcular pérdida
    perdida = criterio(predicciones, ventas_tensor)
    perdida.backward()
    optimizador.step()
    
    # Mostrar progreso
    if (epoch+1) % 50 == 0:
        st.write(f"Época [{epoch+1}/{num_epochs}], Pérdida: {perdida.item():.4f}")


## Guardar Modelo
torch.save(modelo.state_dict(), 'modelo_ventas.pth')
st.write("Modelo guardado como 'modelo_ventas.pth'")

## Graficar Predicciones
modelo.eval()
predicciones = modelo(dias_tensor).detach().numpy()

ventas_predichas = predicciones * (ventas.max() - ventas.min()) + ventas.min()

fig, ax = plt.subplots()
ax.plot(df['dia'], ventas, label='Ventas Reales')
ax.plot(df['dia'], ventas_predichas, label='Predicción', linestyle='--')
ax.set_xlabel('Día')
ax.set_ylabel('Ventas')
ax.legend()
st.pyplot(fig)

st.title('Estimación de Ventas Diarias')