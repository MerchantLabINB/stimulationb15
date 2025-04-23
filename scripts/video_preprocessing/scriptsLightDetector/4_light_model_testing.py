import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el archivo CSV con información de estímulos, estableciendo explícitamente el tipo de dato para 'Día experimental' como cadena
csv_path = "Stimuli_information.csv"
print(f"Cargando archivo CSV desde {csv_path}")
data = pd.read_csv(csv_path, dtype={'Día experimental': str})

# Definir ROIs para cada fecha
rois = {
    '09/05': (147, 1334, 86, 80),
    '15/05': (131, 1200, 86, 80),
    '18/05': (134, 1049, 86, 80),
    '23/05': (134, 953, 86, 80),
    '24/05': (144, 953, 86, 80),
    '28/05': (227, 1155, 86, 80)
}

# Establecer el directorio de videos para el 28 de mayo
video_dir = r'C:\Users\samae\Documents\GitHub\GUI_pattern_generator\data\datos_tesis\videosXavy\mayo28\Lateral'

# Cargar el modelo entrenado
model = tf.keras.models.load_model('redlight_detector_model_augmented_2808.h5')

# Función para encontrar el archivo de video correcto basado en un identificador parcial
def find_video_file(video_dir, identifier):
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if identifier in file:
                return os.path.join(root, file)
    return None

# Función para procesar el video y detectar cuadros con luz
def detect_light_in_video(video_path, roi, model):
    cap = cv2.VideoCapture(video_path)
    light_frames = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extraer la ROI del cuadro
        x, y, w, h = roi
        frame_roi = frame[y:y+h, x:x+w]
        
        # Convertir a escala de grises y normalizar
        frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        frame_normalized = frame_gray / 255.0
        frame_normalized = np.expand_dims(frame_normalized, axis=-1)  # Agregar canal
        frame_normalized = np.expand_dims(frame_normalized, axis=0)   # Agregar tamaño de lote
        
        # Hacer la predicción con el modelo
        prediction = model.predict(frame_normalized)[0][0]
        
        if prediction > 0.5:  # Umbral de clasificación
            light_frames.append(frame_idx)
        
        frame_idx += 1
    
    cap.release()
    return light_frames

# Iterar a través de las filas en el archivo CSV
for index, row in data.iterrows():
    date_str = row['Día experimental']
    
    # Manejar valores de fecha faltantes
    if pd.isna(date_str):
        print(f"Fecha faltante en la fila {index}, omitiendo...")
        continue

    print(f"Procesando fecha: {date_str}")
    
    video_files = row['Archivos de video']
    
    if pd.isna(video_files):
        print("No se encontraron archivos de video para esta entrada.")
        continue

    # Coincidir directamente la cadena de fecha con las claves en el diccionario ROI
    if date_str not in rois:
        print(f"No se encontró ROI para la fecha: {date_str}")
        continue
    
    roi = rois[date_str]

    # Extraer los identificadores de video
    video_identifiers = video_files.splitlines()
    all_start_frames = []
    all_end_frames = []

    for video_identifier in video_identifiers:
        video_identifier = video_identifier.strip()  # Eliminar espacios en blanco
        # Encontrar la ruta del video usando el identificador
        video_path = find_video_file(video_dir, video_identifier)
        if not video_path:
            print(f"No se encontró el archivo de video: {video_identifier}")
            continue

        print(f"Procesando video: {video_path}")

        # Detectar cuadros con luz en el video
        light_frames = detect_light_in_video(video_path, roi, model)
        if not light_frames:
            continue

        # Crear una matriz de cuadros de luz detectados
        frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        light_matrix = np.zeros(frame_count)
        light_matrix[light_frames] = 1

        # Graficar la detección de luz para el video actual
        plt.figure(figsize=(10, 4))
        plt.plot(light_matrix, label='Detección de luz')
        plt.xlabel('Número de cuadro')
        plt.ylabel('Luz detectada (1: Sí, 0: No)')
        plt.title(f'Detección de luz en el video: {os.path.basename(video_path)}')
        plt.legend()
        plt.show()

        # Encontrar intervalos donde se detecta la luz
        light_intervals = []
        current_start = None
        for idx in light_frames:
            if current_start is None:
                current_start = idx
            elif idx - light_frames[light_frames.index(idx) - 1] > 1:
                light_intervals.append((current_start, light_frames[light_frames.index(idx) - 1]))
                current_start = idx
        if current_start is not None:
            light_intervals.append((current_start, light_frames[-1]))

        # Filtrar detecciones de un solo cuadro
        light_intervals = [(start, end) for start, end in light_intervals if end - start > 1]

        # Separar cuadros de inicio y fin
        start_frames = [start for start, _ in light_intervals]
        end_frames = [end for _, end in light_intervals]
        print(f"Cuadros de inicio: {start_frames}")
        print(f"Cuadros de fin: {end_frames}")
        
        # Agregar a la lista general para esta fila
        all_start_frames.extend(start_frames)
        all_end_frames.extend(end_frames)

    # Convertir las listas de cuadros a cadenas separadas por comas
    data.at[index, 'Start frame (lateral)'] = ', '.join(map(str, all_start_frames))
    data.at[index, 'Corresponding end frame (lateral)'] = ', '.join(map(str, all_end_frames))

# Guardar el archivo CSV actualizado
output_csv_path = "Updated_Stimuli_information2805.csv"
data.to_csv(output_csv_path, index=False)
print(f"Archivo CSV actualizado guardado en: {output_csv_path}")
