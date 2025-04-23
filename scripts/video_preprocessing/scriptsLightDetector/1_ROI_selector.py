import cv2
import tkinter as tk
from tkinter import filedialog

def select_video_file():
    # Inicializar la ventana de selección de archivos
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter

    # Mostrar el cuadro de diálogo para seleccionar un archivo
    video_path = filedialog.askopenfilename(
        title="Seleccionar archivo de video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv")]
    )
    
    return video_path

def select_roi_from_video(video_path, display_size=(800, 600)):
    # Abrir el archivo de video
    cap = cv2.VideoCapture(video_path)
    
    # Leer el primer frame del video
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo capturar el primer frame del video.")
        cap.release()
        return None
    
    # Obtener dimensiones originales
    original_height, original_width = frame.shape[:2]
    
    # Calcular el tamaño nuevo manteniendo la relación de aspecto
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        new_width = display_size[0]
        new_height = int(display_size[0] / aspect_ratio)
    else:
        new_height = display_size[1]
        new_width = int(display_size[1] * aspect_ratio)
    
    # Redimensionar el frame para ajustarse a la ventana de visualización
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # Mostrar el frame y permitir al usuario seleccionar la ROI
    roi_resized = cv2.selectROI("Seleccionar ROI", frame_resized, fromCenter=False, showCrosshair=True)
    
    # Cerrar la ventana de visualización
    cv2.destroyAllWindows()
    
    # Liberar el objeto de captura de video
    cap.release()
    
    # Verificar si la ROI es válida
    if roi_resized == (0, 0, 0, 0):
        print("Selección de ROI no válida.")
        return None
    
    # Escalar la ROI al tamaño original
    scale_x = original_width / new_width
    scale_y = original_height / new_height
    roi = (
        int(roi_resized[0] * scale_x),
        int(roi_resized[1] * scale_y),
        int(roi_resized[2] * scale_x),
        int(roi_resized[3] * scale_y)
    )
    
    return roi

# Seleccionar archivo de video
video_path = select_video_file()

if video_path:
    # Seleccionar ROI del primer frame
    roi = select_roi_from_video(video_path)

    if roi:
        print(f"ROI seleccionada: {roi}")  # roi es una tupla (x, y, w, h)
    else:
        print("La selección de ROI no fue exitosa.")

    # Usar la ROI seleccionada para procesamiento adicional
    if roi:
        x, y, w, h = roi
        
        # Ejemplo de uso de la ROI para procesamiento en una función
        def process_video_with_roi(video_path, roi):
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Recortar el frame a la ROI seleccionada
                frame_roi = frame[y:y+h, x:x+w]
                
                # Mostrar el frame recortado
                cv2.imshow("Frame Recortado", frame_roi)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Procesar el video usando la ROI seleccionada
        process_video_with_roi(video_path, roi)
else:
    print("No se seleccionó ningún archivo de video.")
