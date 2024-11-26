// Importar el cliente de MongoDB
const { MongoClient } = require('mongodb');

// URL de conexión a MongoDB
const url_atlas = 'mongodb+srv://adibattista:clase112@cluster0.i8cd3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
const url = 'mongodb://localhost:27017';

// Nombre de la base de datos
const dbName = 'Agenda';

// Crear una instancia de MongoClient
const client = new MongoClient(url_atlas);

// Función principal asíncrona
async function main() {
    // Conectar al cliente
    await client.connect();
    console.log('Conectado correctamente al servidor');

    const db = client.db(dbName);
    const collection = db.collection('Contactos');

    // -------------------------------------------------------------------------------
    // Verificar si se desea borrar todos los contactos
    const borrarTodos = false; // Cambiar a true para borrar todos los contactos
    if (borrarTodos) {
        await collection.deleteMany({});
        // DELETE FROM Contactos;
        console.log('Todos los contactos han sido borrados');
    }

    // -------------------------------------------------------------------------------
    // Verificar si hay contactos; si no, agregar 4 contactos de ejemplo
    const count = await collection.countDocuments();
    if (count === 0) {
        await collection.insertMany([
            { nombre: 'Juan',   apellido: 'Pérez',     telefono: '123456789', email: 'juan@example.com'   },
            { nombre: 'María',  apellido: 'Gómez',     telefono: '987654321', email: 'maria@example.com'  },
            { nombre: 'Carlos', apellido: 'Sánchez',   telefono: '555555555', email: 'carlos@example.com' },
            { nombre: 'Ana',    apellido: 'López',     telefono: '444444444', email: 'ana@example.com'    },
            { nombre: 'Laura',  apellido: 'Pérez',     telefono: '333333333', email: 'laura@example.com'  },
            { nombre: 'Miguel', apellido: 'Pérez',     telefono: '666666666', email: 'miguel@example.com' },
            { nombre: 'Sofía',  apellido: 'Martínez',  telefono: '777777777', email: 'sofia@example.com'  },
            { nombre: 'David',  apellido: 'García',    telefono: '888888888', email: 'david@example.com'  },
            { nombre: 'Lucía',  apellido: 'Hernández', telefono: '999999999', email: 'lucia@example.com'  },
        ]);
        
        // INSERT INTO Contactos (nombre, apellido, telefono, email) VALUES
        // ('Juan', 'Pérez', '123456789', 'juan@example.com'),
        // ('María', 'Gómez', '987654321', 'maria@example.com'),
        // ...
        console.log('Se han insertado 9 contactos adicionales');
    }
    
    // -------------------------------------------------------------------------------
    // Listar todos los contactos 
    const contactos = await collection.find().toArray();
    // SELECT * FROM Contactos;
    console.log('Lista de contactos:', contactos);

    // -------------------------------------------------------------------------------
    // Listar contactos cuyo apellido sea 'Pérez' ordenados por apellido y nombre, limitando a 3 registros
    const contactosLopez = await collection
        .find({ apellido: 'López' })
        .sort({ apellido: 1, nombre: 1 })
        .limit(3)
        .toArray();
    // SELECT * FROM Contactos WHERE apellido = 'López' ORDER BY apellido ASC, nombre ASC LIMIT 3;
    console.log('Contactos de apellido López (máximo 3):', contactosLopez);

    // -------------------------------------------------------------------------------
    // Listar contactos cuyo apellido sea 'Pérez' ordenados por apellido y nombre, limitando a 3 registros usando agregación
    const contactosPerez = await collection.aggregate([
        { $match: { apellido: 'Pérez' } },
        { $sort: { apellido: 1, nombre: 1 } },
        { $limit: 3 }
    ]).toArray();
    // SELECT * FROM Contactos WHERE apellido = 'Pérez' ORDER BY apellido ASC, nombre ASC LIMIT 3;
    console.log('Contactos de apellido Pérez (máximo 3):', contactosPerez);
    // -------------------------------------------------------------------------------
    // Agregar un nuevo contacto 
    const nuevoContacto = { nombre: 'Luis Fernández', telefono: '222333444', email: 'luis@example.com' };
    await collection.insertOne(nuevoContacto);
    // INSERT INTO Contactos (nombre, telefono, email) VALUES ('Luis Fernández', '222333444', 'luis@example.com');
    console.log('Contacto agregado:', nuevoContacto);

    // -------------------------------------------------------------------------------
    // Modificar un contacto existente 
    await collection.updateOne(
        { nombre: 'Juan' },
        { $set: { telefono: '111111111' } }
    );
    // UPDATE Contactos SET telefono = '111111111' WHERE nombre = 'Juan';
    console.log('Contacto modificado: Juan');
  
    // -------------------------------------------------------------------------------  
    // Eliminar un contacto -
    await collection.deleteOne({ nombre: 'Ana' });
    // DELETE FROM Contactos WHERE nombre = 'Ana';

    // -------------------------------------------------------------------------------
    // Agregar 'x' al final del teléfono de todos los Pérez -
    await collection.updateMany(
      { apellido: 'Pérez' },
      [
        {
          $set: {
            telefono: { $concat: [ "$telefono", "x" ] }
          }
        }
      ]
    );
    // UPDATE Contactos SET telefono = CONCAT(telefono, 'x') WHERE apellido = 'Pérez';
    console.log("Se ha agregado 'x' al final del teléfono de todos los Pérez");

    // -------------------------------------------------------------------------------
    // Obtener la cantidad de veces que se repite cada apellido
    const conteoApellidos = await collection.aggregate([
      { $group: { _id: "$apellido", total: { $sum: 1 } } }
    ]).toArray();
    // SELECT apellido, COUNT(*) AS total FROM Contactos GROUP BY apellido;
    console.log('Cantidad de veces que se repite cada apellido:', conteoApellidos);

    console.log('Contacto eliminado: Ana López');

    // Cerrar la conexión
    await client.close();
}

// Ejecutar la función principal
main().catch(console.error);
