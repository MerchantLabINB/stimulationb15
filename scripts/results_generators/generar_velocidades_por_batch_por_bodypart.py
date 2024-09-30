import os
import sys
import pandas as pd
import numpy as np
from math import sqrt
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging  # Import logging module

# Set up logging
log_file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\processing_log.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the path to Stimulation.py
stimulation_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\scripts\GUI_pattern_generator'
sys.path.append(stimulation_path)

# Import the estimulo function from Stimulation.py
from Stimulation import estimulo

# Directories
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
segmented_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_segmentados.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos'
output_pdf_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\pdfs'
font_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\fonts\Arial-Unicode-Regular.ttf'

# Load CSV files
stimuli_info = pd.read_csv(stimuli_info_path)
segmented_info = pd.read_csv(segmented_info_path)

# Body parts
body_parts = ['Muñeca', 'Codo', 'Hombro', 'Frente', 'NudilloCentral', 'DedoMedio', 'Braquiradial', 'Bicep']

# Function to calculate distance between two points
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to find corresponding CSV file based on camera and segment name
def encontrar_csv(camara_lateral, nombre_segmento):
    try:
        for file_name in os.listdir(csv_folder):
            if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
                return os.path.join(csv_folder, file_name)
        logging.warning(f'CSV file not found for camera: {camara_lateral}, segment: {nombre_segmento}')
        return None
    except Exception as e:
        logging.error(f'Error accessing CSV files: {e}')
        return None

# Smoothing function using a moving average
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to calculate velocities for each body part with smoothing
def calcular_velocidades(csv_path):
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        velocidades = {}

        for part in body_parts:
            x_col = (df.columns[1][0], part, 'x')
            y_col = (df.columns[1][0], part, 'y')
            likelihood_col = (df.columns[1][0], part, 'likelihood')
            df_filtered = df[df[likelihood_col] > 0.1]

            if df_filtered.empty:
                velocidades[part] = []
                continue

            velocidad_part = []
            for i in range(1, len(df_filtered)):
                x1, y1 = df_filtered.iloc[i - 1][x_col], df_filtered.iloc[i - 1][y_col]
                x2, y2 = df_filtered.iloc[i][x_col], df_filtered.iloc[i][y_col]
                distancia = calcular_distancia(x1, y1, x2, y2)
                delta_t = 1 / 100  # 100 fps
                velocidad = distancia / delta_t
                velocidad_part.append(velocidad)

            # Apply smoothing (moving average) to the velocity data
            if len(velocidad_part) > 5:  # Only smooth if there's enough data
                velocidad_part = moving_average(velocidad_part, window_size=5)

            velocidades[part] = velocidad_part

        return velocidades
    except Exception as e:
        logging.error(f'Error calculating velocities for CSV: {csv_path}, Error: {e}')
        return {}

# Convert time from microseconds to frames
def us_to_frames(duracion_us):
    return duracion_us / 10000  # 1 frame = 10,000 µs

# Function to generate the stimulus from parameters using Stimulation.py's logic
def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    try:
        forma = forma.strip().lower()  # Ensure lowercase
        print(f"Generating stimulus with shape: {forma}, amplitude: {amplitud}, duration: {duracion}, frequency: {frecuencia}, pulse duration: {duracion_pulso}, compensate: {compensar}")

        # Verify valid parameters
        if duracion <= 0 or frecuencia <= 0 or duracion_pulso <= 0:
            logging.error(f"Invalid parameters: duration={duracion}, frequency={frecuencia}, pulse_duration={duracion_pulso}")
            return [], []

        # Generate stimulus using the Stimulation.py function
        lista_amplitud, lista_tiempo = estimulo(
            forma=forma, amplitud=amplitud, duracion=duracion,
            frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
        )

        # Ensure correct stimulus generation
        if not lista_amplitud or not lista_tiempo:
            logging.error(f"Invalid stimulus with parameters: shape={forma}, amplitude={amplitud}, duration={duracion}, frequency={frecuencia}, pulse_duration={duracion_pulso}, compensate={compensar}")
            return [], []

        # Convert all stimulus times (in µs) to frames
        lista_tiempo = [us_to_frames(tiempo) for tiempo in lista_tiempo]

        return lista_amplitud, lista_tiempo
    except Exception as e:
        logging.error(f'Error generating stimulus: {e}')
        return [], []

# Function to plot stimulus and velocities with fixed x-axis of 400 frames and shaded stimulus area
def plot_stimulus_with_velocities(velocidades, amplitude_list, duration_list, segmento, global_max_velocity):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={'height_ratios': [3, 1]})

    color_map = plt.get_cmap('tab10')
    body_parts = list(velocidades.keys())

    # Gráfico de velocidades
    for i, (part, vel) in enumerate(velocidades.items()):
        if len(vel) > 0:  # Explicitly check if the velocity list is not empty
            ax1.plot(vel, label=f'Velocidad de {part}', color=color_map(i % 10))
            ax1.axhline(np.mean(vel), linestyle='--', color=color_map(i % 10), label=f'Media {part}')

    ax1.axvline(0, color='red', linestyle='--', label='Inicio (frame 0)')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Velocidad (unidades/segundo)')
    ax1.set_ylim(0, 2000)
    ax1.set_xlim(0, 400)  # Fijar el límite superior en 400 frames

    # Verificar si se pueden plotear los datos del estímulo
    if not amplitude_list or not duration_list:
        logging.warning(f"Advertencia: No hay datos válidos para ploteo de estímulo en segmento {segmento}.")
        return "", 0, 0  # Return empty and default frames if stimulus is invalid

    # Ploteo de estímulo
    x_vals = [100]
    y_vals = [0]
    start_frame = 100
    current_frame = start_frame

    for amp, dur in zip(amplitude_list, duration_list):
        frames_to_add = dur  # Time is now already in frames
        next_frame = current_frame + frames_to_add
        x_vals.extend([current_frame, next_frame])
        y_vals.extend([amp / 1000, amp / 1000])
        current_frame = next_frame

    ax2.step(x_vals, y_vals, color='blue', where='post', label='Estimulación Bifásica', linewidth=0.7)
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Amplitud (microamperios)')
    ax2.set_xlim(0, 400)  # Fijar el límite superior en 400 frames

    # Sombrear la región del estímulo en ambas gráficas
    ax1.axvspan(start_frame, current_frame, color='purple', alpha=0.3, label='Zona de estímulo')
    ax2.axvspan(start_frame, current_frame, color='purple', alpha=0.3, label='Zona de estímulo')

    graph_image_path = f"temp_graph_{segmento}.png"
    plt.savefig(graph_image_path)
    plt.close()

    return graph_image_path, start_frame, current_frame  # Return frame boundaries to ensure shading consistency

# Function to plot trajectories of body parts with fixed x-axis of 400 frames and shaded stimulus area
def plot_trajectories_over_frames(csv_path, body_parts, start_frame, end_frame):
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size plot
    color_map = plt.get_cmap('tab10')

    for i, part in enumerate(body_parts):
        x_col = (df.columns[1][0], part, 'x')
        y_col = (df.columns[1][0], part, 'y')
        likelihood_col = (df.columns[1][0], part, 'likelihood')

        df_filtered = df[df[likelihood_col] > 0.1]
        if not df_filtered.empty:
            frames = range(len(df_filtered))
            x_vals = df_filtered[x_col].values
            y_vals = df_filtered[y_col].values

            ax.plot(frames, x_vals, label=f'{part} (x)', color=color_map(i % 10), linestyle='-')
            ax.plot(frames, y_vals, label=f'{part} (y)', color=color_map(i % 10), linestyle='--')

    # Sombrear la región del estímulo usando start_frame y end_frame
    ax.axvspan(start_frame, end_frame, color='purple', alpha=0.3, label='Zona de estímulo')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Coordenadas (px)')
    ax.set_xlim(0, 400)  # Fijar el límite superior en 400 frames
    ax.set_title('Trayectorias (x e y) a lo largo del tiempo')
    ax.legend(loc='upper right')

    trajectory_image_path = "temp_trajectory_over_frames_plot.png"
    plt.savefig(trajectory_image_path)
    plt.close()

    return trajectory_image_path

# Function to generate PDF with consistent shaded areas
def generar_pdf_por_fila(index, row, global_max_velocity):
    try:
        print(f"\nGenerando PDF para la fila {index}, Cámara Lateral: {row['Cámara Lateral']}")
        camara_lateral = row['Cámara Lateral']

        if pd.notna(camara_lateral):
            matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
            if not matching_segment.empty:
                matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.add_font("ArialUnicode", "", font_path, uni=True)
                pdf.set_font("ArialUnicode", size=10)
                info_estimulo = row[['Día experimental', 'Hora', 'Amplitud (μA)', 'Duración (ms)', 'Frecuencia (Hz)', 'Movimiento evocado', 'Forma del Pulso']].to_string()
                pdf.multi_cell(0, 10, f"Información del estímulo:\n{info_estimulo}")

                for _, segment_row in matching_segment_sorted.iterrows():
                    nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                    csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                    if csv_path:
                        velocidades = calcular_velocidades(csv_path)
                        if any(len(v) > 0 for v in velocidades.values()):
                            amplitude_list, duration_list = generar_estimulo_desde_parametros(row['Forma del Pulso'],
                                                                                              row['Amplitud (μA)'] * 1000,
                                                                                              row['Duración (ms)'] * 1000,
                                                                                              row['Frecuencia (Hz)'],
                                                                                              200, compensar=True)
                            # Generate velocity and stimulus plots
                            graph_image_path, start_frame, end_frame = plot_stimulus_with_velocities(velocidades, amplitude_list, duration_list, nombre_segmento, global_max_velocity)

                            # Generate trajectory plots with consistent shading
                            trajectory_image_path = plot_trajectories_over_frames(csv_path, body_parts, start_frame, end_frame)

                            # Add images to PDF
                            pdf.add_page()
                            pdf.set_font("ArialUnicode", size=12)
                            pdf.multi_cell(0, 10, f"Segmento: {nombre_segmento} (Número Ordinal: {segment_row['NumeroOrdinal']})")
                            pdf.image(graph_image_path, x=10, y=20, w=190)
                            pdf.image(trajectory_image_path, x=10, y=160, w=190)

                            # Remove temporary files
                            os.remove(graph_image_path)
                            os.remove(trajectory_image_path)

                safe_filename = camara_lateral.replace("/", "-")
                output_pdf_path = os.path.join(output_pdf_dir, f'{safe_filename}_stimuli.pdf')
                pdf.output(output_pdf_path)
                print(f'PDF generado: {output_pdf_path}')
            else:
                logging.warning(f"No matching segment found for camera: {camara_lateral}")
        else:
            logging.warning(f"Camera lateral information missing for row {index}")
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")

# Main loop to calculate global maximum velocity for plotting
global_max_velocity = 0
for index, row in stimuli_info.iterrows():
    camara_lateral = row['Cámara Lateral']
    if pd.notna(camara_lateral):
        matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
        if not matching_segment.empty:
            matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')
            for _, segment_row in matching_segment_sorted.iterrows():
                nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                if csv_path:
                    velocidades = calcular_velocidades(csv_path)
                    for part, vel in velocidades.items():
                        if len(vel) > 0:  # Ensure this check looks at the length of the velocity list
                            if np.max(vel) > global_max_velocity:
                                global_max_velocity = np.max(vel)

# Generate PDF for each row in Stimuli_information and log any failures
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row, global_max_velocity)
