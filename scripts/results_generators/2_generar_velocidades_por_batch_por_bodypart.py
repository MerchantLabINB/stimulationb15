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

# Import the estimulo function from Stimulation
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

# Function to calculate distance between two points
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to obtain stimulus parameters
def obtener_parametros_estimulo(fila_stimuli):
    forma = fila_stimuli['Forma del Pulso']
    amplitud = fila_stimuli['Amplitud (μA)'] * 1000  # Convertir a nanoamperios
    duracion = fila_stimuli['Duración (ms)'] * 1000  # Convertir a microsegundos
    frecuencia = fila_stimuli['Frecuencia (Hz)']
    duracion_pulso = 200
    compensar = True
    return forma, amplitud, duracion, frecuencia, duracion_pulso, compensar

# Function to generate the stimulus from parameters using Stimulation.py's logic
def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    lista_amplitud, lista_tiempo = estimulo(
        forma=forma, amplitud=amplitud, duracion=duracion,
        frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
    )
    return lista_amplitud, lista_tiempo

# Function to calculate velocities for each body part
def calcular_velocidades(csv_path):
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    body_parts = ['Muñeca', 'Codo', 'Hombro', 'Frente', 'NudilloCentral', 'DedoMedio', 'Braquiradial', 'Bicep']
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

# Function to plot velocities per body part with trials in chronological order using color gradient
def plot_velocities_per_bodypart(velocidades_list, amplitude_list, duration_list, body_part, max_velocity_length):
    num_trials = len(velocidades_list)
    colors = plt.cm.viridis(np.linspace(0, 1, num_trials))  # Use 'viridis' colormap

    # Calculate the third quartile (Q3) for all velocities
    all_velocities = []
    for vel in velocidades_list:
        all_velocities.extend(vel)
    if all_velocities:
        Q3 = np.percentile(all_velocities, 75)
    else:
        Q3 = 0  # In case there are no velocities

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), gridspec_kw={'height_ratios': [3, 1]})

    # Plot velocities for each trial
    for idx, vel in enumerate(velocidades_list):
        ax1.plot(vel, label=f'Trial {idx+1}', color=colors[idx])

    ax1.set_xlabel('Frames')
    ax1.set_ylabel(f'Velocidad de {body_part} (unidades/segundo)')
    ax1.set_ylim(0, Q3)
    ax1.set_xlim(0, max_velocity_length)

    # Generate data for plotting the stimulus using amplitude_list and duration_list
    x_vals = [100]
    y_vals = [0]
    current_frame = 100

    for amp, dur in zip(amplitude_list, duration_list):
        frames_to_add = dur / 10000
        next_frame = current_frame + frames_to_add
        x_vals.extend([current_frame, next_frame])
        y_vals.extend([amp / 1000, amp / 1000])
        current_frame = next_frame

    ax2.step(x_vals, y_vals, color='purple', where='post', label='Estimulación', linewidth=0.7)
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Amplitud (µA)')
    ax2.set_xlim(0, max_velocity_length)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    graph_image_path = f"temp_graph_{body_part}.png"
    plt.savefig(graph_image_path)
    plt.close()

    return graph_image_path

# Function to find corresponding CSV file based on camera and segment name
def encontrar_csv(camara_lateral, nombre_segmento):
    for file_name in os.listdir(csv_folder):
        if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
            return os.path.join(csv_folder, file_name)
    return None

# PDF generation function
def generar_pdf_por_fila(index, row):
    print(f"\nGenerating PDF for row {index}, Cámara Lateral: {row['Cámara Lateral']}")
    camara_lateral = row['Cámara Lateral']

    if pd.notna(camara_lateral):
        matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
        if not matching_segment.empty:
            matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.add_font("ArialUnicode", "", font_path, uni=True)
            pdf.set_font("ArialUnicode", size=10)
            info_estimulo = row[['Día experimental', 'Hora', 'Amplitud (μA)', 'Duración (ms)', 'Frecuencia (Hz)', 'Movimiento evocado']].to_string()
            pdf.multi_cell(0, 10, f"Información del estímulo:\n{info_estimulo}")

            # Initialize velocities per body part
            body_parts = ['Muñeca', 'Codo', 'Hombro', 'Frente', 'NudilloCentral', 'DedoMedio', 'Braquiradial', 'Bicep']
            velocidades_por_bodypart = {part: [] for part in body_parts}

            # For storing the max length of velocities for x-axis limits
            max_velocity_length = 0

            # Collect velocities from all segments
            for _, segment_row in matching_segment_sorted.iterrows():
                nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                if csv_path:
                    velocidades = calcular_velocidades(csv_path)
                    if any(len(v) > 0 for v in velocidades.values()):
                        for part in body_parts:
                            if velocidades[part]:
                                velocidades_por_bodypart[part].append(velocidades[part])
                                if len(velocidades[part]) > max_velocity_length:
                                    max_velocity_length = len(velocidades[part])

            # Generate stimulus parameters
            forma, amplitud, duracion, frecuencia, duracion_pulso, compensar = obtener_parametros_estimulo(row)
            amplitude_list, duration_list = generar_estimulo_desde_parametros(
                forma, amplitud, duracion, frecuencia, duracion_pulso, compensar
            )

            # For each body part, create a plot
            for part in body_parts:
                if velocidades_por_bodypart[part]:
                    graph_image_path = plot_velocities_per_bodypart(
                        velocidades_por_bodypart[part],
                        amplitude_list, duration_list,
                        part, max_velocity_length
                    )
                    pdf.add_page()
                    pdf.set_font("ArialUnicode", size=12)
                    pdf.multi_cell(0, 10, f"Parte del cuerpo: {part}")
                    pdf.image(graph_image_path, x=5, y=20, w=280)
                    os.remove(graph_image_path)

            safe_filename = camara_lateral.replace("/", "-")
            output_pdf_path = os.path.join(output_pdf_dir, f'{safe_filename}_stimuli.pdf')
            pdf.output(output_pdf_path)
            print(f'PDF generated: {output_pdf_path}')

# Generate PDF for each row in Stimuli_information
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row)
