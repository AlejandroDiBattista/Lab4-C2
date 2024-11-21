from fasthtml.common import *


# Crear la aplicación 
app, rt = fast_app()

# Variable global para el contador
contador = 1

# Componentes 
def Contador():
    return Div(f"Contador ", Strong(f" {contador}"), id="contador")

def Incremento(cantidad):
    return Button(f"+ {cantidad}", hx_put=f"/incrementar/{cantidad}", hx_target="#contador", cls="outline")

# Ruta raíz
@rt('/')
def get(): # Genera la página principal
    return Titled("Mi contador", Contador(), Incremento(1), Incremento(5))

# Ruta para incrementar el contador
@rt('/incrementar/{cantidad}') # Generar solo el contador
def put(cantidad: int = 1):
    global contador
    contador += cantidad
    return Contador()

serve()
