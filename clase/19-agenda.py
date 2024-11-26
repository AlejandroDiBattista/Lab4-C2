from fasthtml.common import *

app, rt, contactos, Contacto = fast_app(live=True, 
                        db_file='agenda.db', # Nombre de la base de datos
                        id= int,             # Campos de la clase
                        nombre = str,
                        apellido = str,
                        telefono = str,
                        pk='id'             # Campo clave primaria
                    )


# contactos = lista de contactos
# Contacto  = clase Contacto

if contactos.count == 0: # Si no hay contactos, agregar algunos
    contactos.insert(Contacto(nombre="Juan", apellido="Perez", telefono="123456"))
    contactos.insert(Contacto(nombre="Maria", apellido="Lopez", telefono="654321"))
    contactos.insert(Contacto(nombre="Pedro", apellido="Gomez", telefono="987654"))


def MostrarContacto(contacto):
    return Tr(
        Td(contacto.nombre),
        Td(contacto.apellido),
        Td(contacto.telefono),
        Td(A("Editar",   hx_post=f"/contacto/{contacto.id}")),
        Td(A("Eliminar", hx_delete=f"/contacto/{contacto.id}")),
    )


@rt('/')
def get():
    return Titled("Agenda",
        Button("Agregar", hx_get="/contacto", cls="outline", id="editor"),
        Table(
            Tr(Th("Nombre"), Th("Apellido"), Th("Telefono"), Th(""), Th("")),
            *[MostrarContacto(c) for c in contactos()]
        ),
    )


@rt('/contacto')
def get():
    return Titled("Agregar",
        Form(
            Fieldset(
                Label("Nombre",   Input(type='text',   name='nombre',   placeholder='Nombre')),
                Label("Apellido", Input(type='text',   name='apellido', placeholder='Apellido')),
                Label("Telefono", Input(type='number', name='telefono', placeholder='Telefono'))
            ),
            Div(
                Button('Aceptar',  hx_post='/agregar'),
                Button('Cancelar', onclick="window.history.back()"),
                style="display: flex; justify-content: space-between;"
            )
        )
    )


@rt('/contacto/{id}')
def post(id:int):
    contacto = contactos.get(id)
    return Titled("Editar",
        Form(
            Fieldset(
                Label("Nombre",   Input(type='text',   name='nombre',   value=contacto.nombre)),
                Label("Apellido", Input(type='text',   name='apellido', value=contacto.apellido)),
                Label("Telefono", Input(type='number', name='telefono', value=contacto.telefono))
            ),
            Div(
                Button('Aceptar',  hx_put=f'/contacto/{contacto.id}', hx_target='editor'),
                Button('Cancelar', onclick="window.history.back()"),
                style="display: flex; justify-content: space-between;"
            )
        )
    )


@rt('/contacto/{id}')
def put(id:int, contacto:Contacto):
    contactos.insert(contacto)
    return Redirect('/')


@rt('/contacto/{id}')
def delete(id:int):
    contactos.delete(id)
    return Redirect('/')

serve()
