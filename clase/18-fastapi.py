from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import List

app = FastAPI()

# Configuración de la base de datos
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)

# Definir el modelo Contacto
class Contacto(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    nombre: str
    apellido: str
    telefono: str

# Dependencia para abrir y cerrar sesión de la base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Crear un contacto
@app.post("/contactos/", response_model=Contacto)
def crear_contacto(contacto: Contacto, session: Session = Depends(get_session)):
    session.add(contacto)
    session.commit()
    session.refresh(contacto)
    return contacto

# Obtener todos los contactos
@app.get("/contactos/", response_model=List[Contacto])
def obtener_contactos(session: Session = Depends(get_session)):
    contactos = session.exec(select(Contacto)).all()
    return contactos

# Obtener un contacto por ID
@app.get("/contactos/{contacto_id}", response_model=Contacto)
def obtener_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    return contacto

# Borrar un contacto por ID
@app.delete("/contactos/{contacto_id}", response_model=Contacto)
def borrar_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    session.delete(contacto)
    session.commit()
    return contacto

# Actualizar un contacto por ID
@app.put("/contactos/{contacto_id}", response_model=Contacto)
def actualizar_contacto(contacto_id: int, contacto_actualizado: Contacto, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    contacto.nombre = contacto_actualizado.nombre
    contacto.apellido = contacto_actualizado.apellido
    contacto.telefono = contacto_actualizado.telefono
    session.commit()
    session.refresh(contacto)
    return contacto