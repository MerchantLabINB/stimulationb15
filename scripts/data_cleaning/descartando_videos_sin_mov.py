import pandas as pd

# Ruta de tu archivo CSV
file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'

# Cargar el archivo CSV
data = pd.read_csv(file_path)

# Mostrar las primeras filas para entender su estructura
print("Datos originales:")
print(data.head())

# Asegurarse de que la columna 'Movimiento evocado' esté en formato numérico (convertir a número)
data['Movimiento evocado'] = pd.to_numeric(data['Movimiento evocado'], errors='coerce')

# Añadir una nueva columna 'Descartar' que marque con 'Sí' si el valor es 0 o NaN, de lo contrario 'No'
data['Descartar'] = data['Movimiento evocado'].apply(lambda x: 'Sí' if pd.isna(x) or x == 0 else 'No')

# Verificar las primeras filas después de añadir la columna 'Descartar'
print("\nDatos después de añadir la columna 'Descartar':")
print(data[['Movimiento evocado', 'Descartar']].head())

# Guardar el archivo CSV modificado con la codificación UTF-8
output_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_con_descartar.csv'
data.to_csv(output_path, index=False, encoding='utf-8')
print(f"Archivo CSV modificado guardado en: {output_path}")
