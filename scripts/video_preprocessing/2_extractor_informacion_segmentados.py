import os
import csv

# Ruta de la carpeta videosSegmentados
ruta_videos_segmentados = 'videosSegmentados'

# Lista para almacenar la información de los archivos
informacion_archivos = []

# Recorrer todas las carpetas y archivos dentro de videosSegmentados
for carpeta, subcarpetas, archivos in os.walk(ruta_videos_segmentados):
    if "Lateral" not in carpeta:
        print(f"Omitiendo carpeta: {carpeta}")
        continue

    # Lista temporal para almacenar los archivos de cada carpeta
    archivos_en_carpeta = []

    for archivo in archivos:
        # Mostrar un mensaje de procesamiento
        print(f"Procesando archivo: {archivo}")

        # Verificar si el archivo tiene la extensión ".mp4"
        if not archivo.endswith(".mp4"):
            print(f"El archivo {archivo} no es un archivo de video, se omitirá.")
            continue

        # Dividir el nombre del archivo para obtener el número de frame y tiempo
        partes_nombre = archivo.split('_')

        # Verificar si hay suficientes partes en el nombre para evitar IndexError
        if len(partes_nombre) >= 4:
            try:
                numero_frame = int(partes_nombre[2])  # El número de frame
                tiempo = partes_nombre[3].split('.')[0]  # Tiempo sin la extensión del archivo

                # Extraer la carpeta a la que pertenece
                carpeta_pertenece = os.path.basename(carpeta)

                # Guardar la información en la lista temporal
                archivos_en_carpeta.append([carpeta_pertenece, archivo, numero_frame, tiempo])
            except Exception as e:
                print(f"Error procesando el archivo {archivo}: {e}")
        else:
            print(f"El archivo {archivo} no tiene el formato esperado, se omitirá.")
    
    # Ordenar los archivos de la carpeta por el número de frame
    archivos_en_carpeta.sort(key=lambda x: x[2])

    # Asignar número ordinal para los archivos dentro de la misma carpeta
    for index, archivo_info in enumerate(archivos_en_carpeta):
        archivo_info.append(index + 1)  # Añadir el número ordinal
        informacion_archivos.append(archivo_info)

# Escribir la información en un archivo CSV
with open('informacion_archivos.csv', mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    # Escribir la fila de encabezados
    escritor_csv.writerow(['CarpetaPertenece', 'NombreArchivo', 'NumeroFrame', 'Tiempo', 'NumeroOrdinal'])
    # Escribir la información de los archivos
    escritor_csv.writerows(informacion_archivos)
