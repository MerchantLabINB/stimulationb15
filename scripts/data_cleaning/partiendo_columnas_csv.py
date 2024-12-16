import pandas as pd

# Ruta de tu archivo CSV
file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'

# Cargar el archivo CSV
data = pd.read_csv(file_path)

# Mostrar las primeras filas para entender su estructura
print("Datos originales:")
print(data.head())

# Definir una función que separe la columna y maneje errores
def split_video_column(value):
    try:
        # Intentar dividir la cadena por salto de línea
        lateral, frontal = value.split('\n', 1)
        return lateral, frontal
    except ValueError:
        # Si no se pueden dividir en dos partes, devolver NaN
        return pd.NA, pd.NA

# Obtener la posición de la columna original 'Archivos de video'
video_col_index = data.columns.get_loc('Archivos de video')

# Aplicar la función a cada valor de la columna 'Archivos de video'
data['Camara Lateral'], data['Camara Frontal'] = zip(*data['Archivos de video'].map(split_video_column))

# Insertar las dos nuevas columnas en la posición original
data.insert(video_col_index, 'Camara Lateral', data.pop('Camara Lateral'))
data.insert(video_col_index + 1, 'Camara Frontal', data.pop('Camara Frontal'))

# Eliminar la columna original 'Archivos de video'
data = data.drop(columns=['Archivos de video'])

# Verifica que se haya realizado la separación y eliminación de la columna original
print("\nDatos después de separar y eliminar la columna 'Archivos de video':")
print(data.head())

# Guardar el nuevo archivo CSV sin redundancias
output_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_modificado.csv'
data.to_csv(output_path, index=False)
print(f"Archivo CSV modificado guardado en: {output_path}")
