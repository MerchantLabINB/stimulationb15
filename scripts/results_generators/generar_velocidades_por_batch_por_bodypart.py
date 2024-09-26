import os
import sys
import pandas as pd
import numpy as np
from math import sqrt
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# Function to obtain stimulus parameters
def obtener_parametros_estimulo(fila_stimuli):
    forma = fila_stimuli['Forma del Pulso']
    amplitud = fila_stimuli['Amplitud (µA)'] * 1000  # Convert to nanoamperios
    duracion = fila_stimuli['Duración (ms)'] * 1000  # Convert to microseconds
    frecuencia = fila_stimuli['Frecuencia (Hz)']
    duracion_pulso = 200
    compensar = True
    return forma, amplitud, duracion, frecuencia, duracion_pulso, compensar

# Function to generate the stimulus from parameters using Stimulation.py's logic
def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    print(f"Generando estímulo con forma: {forma}, amplitud: {amplitud}, duración: {duracion}, frecuencia: {frecuencia}, duración pulso: {duracion_pulso}, compensar: {compensar}")
    lista_amplitud, lista_tiempo = estimulo(
        forma=forma, amplitud=amplitud, duracion=duracion,
        frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
    )
    return lista_amplitud, lista_tiempo

# Function to plot trajectories of body parts
def plot_trajectories(csv_path, body_parts):
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size plot
    color_map = plt.get_cmap('tab10')  # Color map for distinct colors

    for i, part in enumerate(body_parts):
        x_col = (df.columns[1][0], part, 'x')
        y_col = (df.columns[1][0], part, 'y')
        likelihood_col = (df.columns[1][0], part, 'likelihood')

        # Filter out rows where likelihood is too low
        df_filtered = df[df[likelihood_col] > 0.1]
        if not df_filtered.empty:
            x_vals = df_filtered[x_col].values
            y_vals = df_filtered[y_col].values
            ax.plot(x_vals, y_vals, label=f'Trayectoria de {part}', color=color_map(i % 10))

    ax.set_xlabel('Frames')
    ax.set_ylabel('Posición (px)')
    ax.set_title('Trayectorias de las partes del cuerpo')
    ax.legend(loc='upper right')
    
    trajectory_image_path = "temp_trajectory_plot.png"
    plt.savefig(trajectory_image_path)
    plt.close()
    
    return trajectory_image_path

def plot_trajectories_over_frames(csv_path, body_parts):
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size plot
    color_map = plt.get_cmap('tab10')  # Color map for distinct colors

    for i, part in enumerate(body_parts):
        x_col = (df.columns[1][0], part, 'x')
        y_col = (df.columns[1][0], part, 'y')
        likelihood_col = (df.columns[1][0], part, 'likelihood')

        # Filter out rows where likelihood is too low
        df_filtered = df[df[likelihood_col] > 0.1]
        if not df_filtered.empty:
            frames = range(len(df_filtered))  # Frames will be the x-axis
            x_vals = df_filtered[x_col].values
            y_vals = df_filtered[y_col].values

            # Plot both x and y coordinates against frames
            ax.plot(frames, x_vals, label=f'{part} (x)', color=color_map(i % 10), linestyle='-')
            ax.plot(frames, y_vals, label=f'{part} (y)', color=color_map(i % 10), linestyle='--')

    ax.set_xlabel('Frames')
    ax.set_ylabel('Coordenadas (px)')
    ax.set_title('Trayectorias (x e y) a lo largo del tiempo')
    ax.legend(loc='upper right')
    
    trajectory_image_path = "temp_trajectory_over_frames_plot.png"
    plt.savefig(trajectory_image_path)
    plt.close()
    
    return trajectory_image_path

# Function to calculate velocities for each body part
def calcular_velocidades(csv_path):
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

        if velocidad_part:
            Q1, Q3 = np.percentile(velocidad_part, [25, 75])
            IQR = Q3 - Q1
            velocidad_filtered = [v for v in velocidad_part if Q1 - 1.5 * IQR <= v <= Q3 + 1.5 * IQR]
            velocidades[part] = velocidad_filtered
        else:
            velocidades[part] = []

    return velocidades

# Function to find corresponding CSV file based on camera and segment name
def encontrar_csv(camara_lateral, nombre_segmento):
    for file_name in os.listdir(csv_folder):
        if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
            return os.path.join(csv_folder, file_name)
    return None

# Adjusted code for generating stimulus and including pulse shape in the PDF

# Function to plot stimulus and velocities
def plot_stimulus_with_velocities(velocidades, amplitude_list, duration_list, segmento, global_max_velocity):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={'height_ratios': [3, 1]})

    # Dynamically generate a colormap with enough distinct colors
    color_map = plt.get_cmap('tab10')  # 'tab10' provides 10 distinct colors; use other maps like 'tab20' if needed
    body_parts = list(velocidades.keys())  # Get the body parts from the velocities dictionary

    # Plot velocities for each body part
    for i, (part, vel) in enumerate(velocidades.items()):
        if vel:  # Ensure there is valid data to plot
            ax1.plot(vel, label=f'Velocidad de {part}', color=color_map(i % 10))  # Use modulo to cycle through colors
            ax1.axhline(np.mean(vel), linestyle='--', color=color_map(i % 10), label=f'Media {part}')

    ax1.axvline(0, color='red', linestyle='--', label='Inicio (frame 0)')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Velocidad (unidades/segundo)')
    ax1.set_ylim(0, min(global_max_velocity, 5000))  # Set Y limit up to 5000 if max_velocity exceeds it
    ax1.set_xlim(0, max(len(vel) for vel in velocidades.values()))  # Set x limit based on the max length of velocity data

    # Stimulus starting at frame 100
    x_vals = [100]  
    y_vals = [0]    
    current_frame = 100  
    start_frame = current_frame  
    end_frame = current_frame   

    for amp, dur in zip(amplitude_list, duration_list):
        frames_to_add = dur / 10000
        next_frame = current_frame + frames_to_add
        x_vals.extend([current_frame, next_frame])
        y_vals.extend([amp / 1000, amp / 1000])
        current_frame = next_frame
    end_frame = current_frame

    ax2.step(x_vals, y_vals, color='purple', where='post', label='Estimulación Bifásica', linewidth=0.7)
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Amplitud (microamperios)')
    ax2.set_xlim(0, max(len(vel) for vel in velocidades.values()))

    ax1.axvline(start_frame, color='purple', linestyle='--', label='Inicio del estímulo', linewidth=0.7)
    ax1.axvline(end_frame, color='purple', linestyle='--', label='Fin del estímulo', linewidth=0.7)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=min(y_vals))

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    graph_image_path = f"temp_graph_{segmento}.png"
    plt.savefig(graph_image_path)
    plt.close()

    return graph_image_path

# PDF generation function with added trajectories plot
# Function to generate the PDF, velocities, stimulus, and trajectories aligned
# Function to generate the PDF, velocities, stimulus, and trajectories aligned vertically
def generar_pdf_por_fila(index, row, global_max_velocity):
    print(f"\nGenerating PDF for row {index}, Cámara Lateral: {row['Cámara Lateral']}")
    camara_lateral = row['Cámara Lateral']

    if pd.notna(camara_lateral):
        matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
        if not matching_segment.empty:
            matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

            pdf = FPDF(orientation='P', unit='mm', format='A4')  # Vertical orientation (Portrait)
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
                        amplitude_list, duration_list = estimulo(row['Forma del Pulso'], 
                                                                row['Amplitud (μA)'] * 1000,
                                                                row['Duración (ms)'] * 1000,
                                                                row['Frecuencia (Hz)'], 
                                                                200)
                        # Generate velocities and stimulus plot
                        graph_image_path = plot_stimulus_with_velocities(velocidades, amplitude_list, duration_list, nombre_segmento, global_max_velocity)

                        # Generate trajectory plot over frames
                        trajectory_image_path = plot_trajectories_over_frames(csv_path, body_parts)

                        # Add all plots to a single page, arranged vertically
                        pdf.add_page()  # New page for each video segment
                        pdf.set_font("ArialUnicode", size=12)
                        pdf.multi_cell(0, 10, f"Segmento: {nombre_segmento} (Número Ordinal: {segment_row['NumeroOrdinal']})")

                        # Ensure the images fit in a vertical layout on one page
                        pdf.image(graph_image_path, x=10, y=20, w=190)  # Adjust size to fit page width (A4 size: 210mm)
                        pdf.image(trajectory_image_path, x=10, y=160, w=190)  # Align second image below the first one

                        # Clean up temporary image files
                        os.remove(graph_image_path)
                        os.remove(trajectory_image_path)

            safe_filename = camara_lateral.replace("/", "-")
            output_pdf_path = os.path.join(output_pdf_dir, f'{safe_filename}_stimuli.pdf')
            pdf.output(output_pdf_path)
            print(f'PDF generated: {output_pdf_path}')

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
                        if vel:
                            Q1, Q3 = np.percentile(vel, [25, 75])
                            if Q3 > global_max_velocity:
                                global_max_velocity = Q3

# Generate PDF for each row in Stimuli_information
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row, global_max_velocity)