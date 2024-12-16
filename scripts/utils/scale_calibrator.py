import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime

# Definir la ruta de la carpeta de logs
log_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\logs'

# Crear el directorio de logs si no existe
os.makedirs(log_dir, exist_ok=True)

# Tamaño del círculo de selección
circle_radius = 2  # Radio más pequeño para mayor precisión

# Función para hacer zoom en una imagen
def zoom_image(image, scale):
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    zoomed_image = cv2.resize(image, (new_width, new_height))
    return zoomed_image

# Función para capturar dos puntos en una imagen usando OpenCV con posibilidad de zoom
def select_points(image):
    points = []
    zoom_factor = 1.0  # Factor de zoom inicial
    display_image = image.copy()

    def display_zoom():
        zoomed_image = zoom_image(display_image, zoom_factor)
        cv2.imshow('image', zoomed_image)

    # Función de callback para obtener los puntos clicados
    def click_event(event, x, y, flags, param):
        nonlocal display_image, zoom_factor

        if event == cv2.EVENT_LBUTTONDOWN:
            # Calcular la posición en la imagen original
            original_x = int(x / zoom_factor)
            original_y = int(y / zoom_factor)
            points.append((original_x, original_y))

            # Dibujar un círculo en la imagen original
            cv2.circle(display_image, (original_x, original_y), circle_radius, (0, 255, 0), -1)

            # Mostrar la imagen con el nuevo punto
            display_zoom()

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Zoom in
                zoom_factor += 0.1
            else:  # Zoom out
                zoom_factor = max(0.1, zoom_factor - 0.1)
            display_zoom()

    # Mostrar la imagen y esperar a que el usuario seleccione dos puntos
    cv2.imshow('image', display_image)
    cv2.setMouseCallback('image', click_event)
    
    # Esperar hasta que se seleccionen dos puntos
    while len(points) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return points

# Función para calcular la distancia entre dos puntos
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Función para abrir un cuadro de diálogo para seleccionar un video
def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    video_path = filedialog.askopenfilename(title="Selecciona el video",
                                            filetypes=(("Archivos de video", "*.mp4 *.avi *.mov"), ("Todos los archivos", "*.*")))
    return video_path

# Función principal para seleccionar un video, obtener una imagen y calcular el factor de escala
def calibrate_video_scale():
    # Seleccionar el video mediante diálogo
    video_path = select_video_file()
    if not video_path:
        print("No se seleccionó ningún video.")
        return
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return
    
    # Leer el primer frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame del video.")
        cap.release()
        return
    
    # Mostrar el frame y permitir que el usuario seleccione dos puntos
    print("Selecciona dos puntos en la imagen entre los cuales quieres medir la distancia. Usa la rueda del ratón para hacer zoom.")
    points = select_points(frame)

    # Calcular la distancia en píxeles
    pixel_distance = calculate_distance(points[0], points[1])
    print(f"Distancia entre puntos seleccionados: {pixel_distance:.2f} píxeles")

    # Pedir al usuario que introduzca la distancia real en centímetros
    real_distance_cm = float(input("Introduce la distancia real entre los dos puntos seleccionados (en cm): "))

    # Calcular el factor de escala (cm por píxel)
    scale_factor = real_distance_cm / pixel_distance
    print(f"Factor de escala: {scale_factor:.4f} cm/píxel")

    # Obtener la fecha y hora actuales para el log
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Obtener el nombre del archivo de video
    video_name = os.path.basename(video_path)

    # Registrar los detalles en un archivo de log
    log_path = os.path.join(log_dir, 'calibration_log.txt')
    with open(log_path, 'a') as log_file:
        log_file.write(f"Fecha: {current_time}, Video: {video_name}, Factor de escala: {scale_factor:.4f} cm/píxel\n")

    print(f"Registro guardado en {log_path}")

    # Liberar el video
    cap.release()

    # Devolver el factor de escala para su uso posterior
    return scale_factor

# Función para aplicar la escala a una distancia en píxeles
def pixels_to_cm(pixel_distance, scale_factor):
    return pixel_distance * scale_factor

# Ejemplo de uso
if __name__ == "__main__":
    scale_factor = calibrate_video_scale()
    if scale_factor:
        print(f"Factor de escala calculado: {scale_factor} cm/píxel")
