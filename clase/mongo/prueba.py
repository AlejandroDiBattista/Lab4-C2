from pymongo import MongoClient

# URL de conexión a MongoDB
url_atlas = 'mongodb+srv://adibattista:clase112@cluster0.i8cd3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
url = 'mongodb://localhost:27017'

# Nombre de la base de datos
db_name = 'Agenda'

# Crear una instancia de MongoClient
client = MongoClient(url)

def main():
    # Conectar al cliente
    db = client[db_name]
    collection = db['Contactos']

    # -------------------------------------------------------------------------------
    # Verificar si se desea borrar todos los contactos
    borrar_todos = False  # Cambiar a True para borrar todos los contactos
    if borrar_todos:
        collection.delete_many({})
        # DELETE FROM Contactos;
        print('Todos los contactos han sido borrados')
        
    # -------------------------------------------------------------------------------
    # Verificar si hay contactos; si no, agregar contactos de ejemplo
    count = collection.count_documents({})
    if count == 0:
        collection.insert_many([
            {'nombre': 'Juan',   'apellido': 'Pérez',     'telefono': '123456789', 'email': 'juan@example.com'},
            {'nombre': 'María',  'apellido': 'Gómez',     'telefono': '987654321', 'email': 'maria@example.com'},
            {'nombre': 'Carlos', 'apellido': 'Sánchez',   'telefono': '555555555', 'email': 'carlos@example.com'},
            {'nombre': 'Ana',    'apellido': 'López',     'telefono': '444444444', 'email': 'ana@example.com'},
            {'nombre': 'Laura',  'apellido': 'Pérez',     'telefono': '333333333', 'email': 'laura@example.com'},
            {'nombre': 'Miguel', 'apellido': 'Pérez',     'telefono': '666666666', 'email': 'miguel@example.com'},
            {'nombre': 'Sofía',  'apellido': 'Martínez',  'telefono': '777777777', 'email': 'sofia@example.com'},
            {'nombre': 'David',  'apellido': 'García',    'telefono': '888888888', 'email': 'david@example.com'},
            {'nombre': 'Lucía',  'apellido': 'Hernández', 'telefono': '999999999', 'email': 'lucia@example.com'},
        ])
        # INSERT INTO Contactos (nombre, apellido, telefono, email) VALUES
        # ('Juan', 'Pérez', '123456789', 'juan@example.com'),
        # ('María', 'Gómez', '987654321', 'maria@example.com'),
        # ...
        print('Se han insertado 9 contactos adicionales')

    # -------------------------------------------------------------------------------
    # Listar todos los contactos
    contactos = list(collection.find())
    # SELECT * FROM Contactos;
    print('Lista de contactos:', contactos)

    # -------------------------------------------------------------------------------
    # Listar contactos cuyo apellido sea 'López' ordenados por apellido y nombre, limitando a 3 registros
    contactos_lopez = list(collection
                           .find({'apellido': 'López'})
                           .sort([('apellido', 1), ('nombre', 1)])
                           .limit(3))
    # SELECT * FROM Contactos WHERE apellido = 'López' ORDER BY apellido ASC, nombre ASC LIMIT 3;
    print('Contactos de apellido López (máximo 3):', contactos_lopez)

    # -------------------------------------------------------------------------------
    # Listar contactos cuyo apellido sea 'Pérez' ordenados por apellido y nombre, limitando a 3 registros usando agregación
    contactos_perez = list(collection.aggregate([
        { '$match': { 'apellido': 'Pérez' } },
        { '$sort': { 'apellido': 1, 'nombre': 1 } },
        { '$limit': 3 }
    ]))
    # SELECT * FROM Contactos WHERE apellido = 'Pérez' ORDER BY apellido ASC, nombre ASC LIMIT 3;
    print('Contactos de apellido Pérez (máximo 3):', contactos_perez)

    # -------------------------------------------------------------------------------
    # Agregar un nuevo contacto
    nuevo_contacto = {'nombre': 'Luis Fernández', 'telefono': '222333444', 'email': 'luis@example.com'}
    collection.insert_one(nuevo_contacto)
    # INSERT INTO Contactos (nombre, telefono, email) VALUES ('Luis Fernández', '222333444', 'luis@example.com');
    print('Contacto agregado:', nuevo_contacto)

    # -------------------------------------------------------------------------------
    # Modificar un contacto existente
    collection.update_one(
        {'nombre': 'Juan Pérez'},
        {'$set': {'telefono': '111111111'}}
    )
    # UPDATE Contactos SET telefono = '111111111' WHERE nombre = 'Juan Pérez';
    print('Contacto modificado: Juan Pérez')

    # -------------------------------------------------------------------------------
    # Eliminar un contacto
    collection.delete_one({'nombre': 'Ana'})
    # DELETE FROM Contactos WHERE nombre = 'Ana';
    print('Contacto eliminado: Ana López')

    # -------------------------------------------------------------------------------
    # Agregar 'x' al final del teléfono de todos los Pérez
    collection.update_many(
        {'apellido': 'Pérez'},
        [{'$set': {'telefono': {'$concat': ['$telefono', 'x']}}}]
    )
    # UPDATE Contactos SET telefono = CONCAT(telefono, 'x') WHERE apellido = 'Pérez';
    print("Se ha agregado 'x' al final del teléfono de todos los Pérez")

    # -------------------------------------------------------------------------------
    # Obtener la cantidad de veces que se repite cada apellido
    conteo_apellidos = collection.aggregate([
        { '$group': { '_id': '$apellido', 'total': { '$sum': 1 } } }
    ])
    # SELECT apellido, COUNT(*) AS total FROM Contactos GROUP BY apellido;
    print('Cantidad de veces que se repite cada apellido:', list(conteo_apellidos))

    # Cerrar la conexión
    client.close()

if __name__ == '__main__':
    main()
