# generar_pdf.py - Genera un PDF por cada entrada en Stimuli_information.csv con información de los estímulos y gráficos de velocidad y poses.
import os
import pandas as pd
from fpdf import FPDF
from math import sqrt
import matplotlib.pyplot as plt

# Directorios
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
segmented_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_segmentados.csv'
plots_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos\plot-poses'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos'
output_pdf_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\pdfs'
log_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\logs'
font_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\fonts\Arial-Unicode-Regular.ttf'

# Crear el directorio de logs si no existe
os.makedirs(log_dir, exist_ok=True)

# Archivos de log
log_pdf_path = os.path.join(log_dir, 'log_archivos_pdf.txt')
log_omitidos_path = os.path.join(log_dir, 'log_archivos_omitidos.txt')

# Inicializar logs
with open(log_pdf_path, 'w') as log_pdf, open(log_omitidos_path, 'w') as log_omitidos:
    log_pdf.write("Archivos PDF generados:\n")
    log_omitidos.write("Archivos omitidos por valor 'Descartar = Sí':\n")

# Leer los archivos CSV
print("Leyendo Stimuli_information.csv...")
stimuli_info = pd.read_csv(stimuli_info_path)
print("Leyendo informacion_archivos_segmentados.csv...")
segmented_info = pd.read_csv(segmented_info_path)

# Filtrar las entradas que deben ser descartadas
omitidos = stimuli_info[stimuli_info['Descartar'].isin(['Sí', 'SÃ­'])]  # Las que se van a omitir
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']  # Las que no se omiten (solo 'No')

# Guardar en el log los omitidos
with open(log_omitidos_path, 'a') as log_omitidos:
    for _, row in omitidos.iterrows():
        log_omitidos.write(f"ID: {row['Cámara Lateral']}, Descartar: {row['Descartar']}\n")

# Asegurarse de que el directorio para los PDFs existe
os.makedirs(output_pdf_dir, exist_ok=True)

# Definir columnas relacionadas a los parámetros del estímulo que queremos mostrar
columnas_estimulo = ['Día experimental', 'Hora', 'Amplitud (μA)', 'Duración (ms)', 
                     'Forma del Pulso', 'Frecuencia (Hz)', 'Top of the cortex (mm)', 
                     'Profundidad electrodo (mm)', 'Movimiento evocado', 'Cabeza Libre (0 No, 1 Si)']

# Función para calcular la distancia entre dos puntos
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Función para calcular la velocidad de una parte del cuerpo
def calcular_velocidades(csv_path):
    # Leer el archivo CSV con encabezado multinivel
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    
    # Lista de partes del cuerpo que nos interesan
    body_parts = ['Muñeca', 'Codo', 'Hombro']
    
    # Filtrar por likelihood > 0.5
    velocidades = {}
    for part in body_parts:
        x_col = (df.columns[1][0], part, 'x')
        y_col = (df.columns[1][0], part, 'y')
        likelihood_col = (df.columns[1][0], part, 'likelihood')

        df_filtered = df[df[likelihood_col] > 0.2]  # Likelihood ajustado a 0.2

        # Calcular las velocidades
        velocidad_part = []
        for i in range(1, len(df_filtered)):
            x1, y1 = df_filtered.iloc[i - 1][x_col], df_filtered.iloc[i - 1][y_col]
            x2, y2 = df_filtered.iloc[i][x_col], df_filtered.iloc[i][y_col]
            distancia = calcular_distancia(x1, y1, x2, y2)
            
            # Supongamos que Δt entre frames es constante (100 fps -> Δt = 1/100 segundos)
            delta_t = 1 / 100  
            velocidad = distancia / delta_t
            velocidad_part.append(velocidad)
        
        velocidades[part] = velocidad_part
    
    return velocidades

# Función para generar gráficos de velocidad
def generar_grafico_velocidad(velocidades, segmento):
    plt.figure(figsize=(11.69, 8.27))  # Ajuste para tamaño A4 en pulgadas
    colors = ['blue', 'orange', 'green']  # Colores para las partes del cuerpo
    for i, (part, velocidad) in enumerate(velocidades.items()):
        if len(velocidad) > 0:
            plt.plot(velocidad, label=f'Velocidad de {part}', color=colors[i])
            # Calcular la media de las velocidades
            media_velocidad = sum(velocidad) / len(velocidad)
            plt.axhline(media_velocidad, linestyle='--', color=colors[i], label=f'Media {part}')
    
    # Añadir línea en el frame 100 si hay suficientes frames
    plt.axvline(100, color='blue', linestyle='--', label='1 segundo (frame 100)')

    plt.title(f'Velocidad a lo largo del tiempo - Segmento {segmento}')
    plt.xlabel('Frames')
    plt.ylabel('Velocidad (unidades/segundo)')
    plt.legend()
    
    # Guardar la imagen del gráfico temporalmente
    graph_image_path = f"temp_graph_{segmento}.png"
    plt.savefig(graph_image_path)
    plt.close()
    
    return graph_image_path

# Función para buscar el archivo CSV que coincida con las dos partes del nombre (camara_lateral y nombre_segmento)
def encontrar_csv(camara_lateral, nombre_segmento):
    for file_name in os.listdir(csv_folder):
        if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
            return os.path.join(csv_folder, file_name)
    return None

# Función para generar un PDF por cada entrada
def generar_pdf_por_fila(index, row):
    print(f"\nGenerando PDF para la fila {index}, Cámara Lateral: {row['Cámara Lateral']}")
    
    camara_lateral = row['Cámara Lateral']
    
    if pd.notna(camara_lateral):
        # Buscar coincidencia en 'informacion_archivos_segmentados.csv'
        matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
        
        if not matching_segment.empty:
            # Ordenar los videos segmentados en base a 'NumeroOrdinal'
            matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()

            # Agregar fuente y establecer tamaño pequeño
            pdf.add_font("ArialUnicode", "", font_path, uni=True)
            pdf.set_font("ArialUnicode", size=10)

            # Mostrar solo las columnas relacionadas a los parámetros del estímulo
            info_estimulo = row[columnas_estimulo].to_string()
            pdf.multi_cell(0, 10, f"Información del estímulo:\n{info_estimulo}")

            segmentos_tomados = []

            for _, segment_row in matching_segment_sorted.iterrows():
                nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                segmentos_tomados.append(nombre_segmento)
                
                # Buscar el archivo CSV y calcular las velocidades
                csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                if csv_path:
                    velocidades = calcular_velocidades(csv_path)

                    if any(len(v) > 0 for v in velocidades.values()):  # Solo si hay velocidades no vacías
                        # Generar el gráfico de velocidades
                        graph_image_path = generar_grafico_velocidad(velocidades, nombre_segmento)
                        
                        # Agregar el nombre del segmento al inicio de la página
                        pdf.add_page()
                        pdf.set_font("ArialUnicode", size=12)
                        pdf.multi_cell(0, 10, f"Segmento: {nombre_segmento} (Número Ordinal: {segment_row['NumeroOrdinal']})")

                        # Colocar el gráfico de velocidad en el PDF (tamaño completo)
                        pdf.image(graph_image_path, x=5, y=20, w=280)  # Ajustar para que ocupe casi toda la página

                        # Eliminar la imagen temporal después de insertarla
                        os.remove(graph_image_path)

                # Buscar y agregar las imágenes del segmento (plot poses)
                for root, dirs, _ in os.walk(plots_folder):
                    for dir_name in dirs:
                        if camara_lateral in dir_name and nombre_segmento in dir_name:
                            image_dir = os.path.join(root, dir_name)

                            # Obtener las imágenes y ordenarlas
                            images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
                            images.sort()  # Ordenarlas por nombre para que se añadan en orden

                            # Añadir imágenes al PDF ocupando casi toda la página
                            for i, image_file in enumerate(images):
                                image_path = os.path.join(image_dir, image_file)
                                pdf.add_page()  # Nueva página para cada imagen
                                pdf.image(image_path, x=5, y=20, w=280)  # Ajustar para ocupar toda la hoja
    
            # Guardar el PDF
            safe_filename = camara_lateral.replace("/", "-")
            output_pdf_path = os.path.join(output_pdf_dir, f'{safe_filename}_stimuli.pdf')
            pdf.output(output_pdf_path)
            print(f'PDF generado: {output_pdf_path}')

            # Guardar en el log de PDFs generados
            with open(log_pdf_path, 'a') as log_pdf:
                log_pdf.write(f"PDF: {output_pdf_path}, Segmentos: {', '.join(segmentos_tomados)}\n")

# Generar PDF por cada fila en Stimuli_information
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row)
