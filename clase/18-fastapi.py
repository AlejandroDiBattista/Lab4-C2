from fastapi import FastAPI
from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy import func

app = FastAPI()

class Contacto(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    nombre: str = Field(..., min_length=1)
    apellido: str = Field(..., min_length=1)
    telefono: str = Field(..., min_length=6, max_length=12)

# Configuración de la base de datos
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)

# Crear las tablas
SQLModel.metadata.create_all(engine)

def init_db():
    with Session(engine) as session:
        contactos_existentes = session.exec(select(Contacto)).all()
        if not contactos_existentes:
            contactos_ejemplo = [
                Contacto(nombre="Juan", apellido="Perez", telefono="123456789"),
                Contacto(nombre="Ana", apellido="Gomez", telefono="987654321"),
                Contacto(nombre="Luis", apellido="Martinez", telefono="555555555")
            ]
            session.add_all(contactos_ejemplo)
            session.commit()

init_db()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = ''):
    return {"item_id": item_id, "q": q}

@app.post("/contactos/")
def create_contacto(contacto: Contacto):
    with Session(engine) as session:
        session.add(contacto)
        session.commit()
        session.refresh(contacto)
    return contacto

@app.get("/contactos/")
def read_contactos():
    with Session(engine) as session:
        contactos = session.exec(
            select(Contacto).where(func.length(Contacto.nombre) >= 3)
        ).all()
    return contactos

@app.get("/contactos/{contacto_id}")
def read_contacto(contacto_id: int):
    with Session(engine) as session:
        contacto = session.get(Contacto, contacto_id)
        if contacto is None:
            return {"error": "Contacto no encontrado"}
    return contacto

@app.delete("/contactos/{contacto_id}")
def delete_contacto(contacto_id: int):
    with Session(engine) as session:
        contacto = session.get(Contacto, contacto_id)
        if contacto is None:
            return {"error": "Contacto no encontrado"}
        session.delete(contacto)
        session.commit()
    return {"message": "Contacto borrado exitosamente"}