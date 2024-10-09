import os
import sys
import pandas as pd
import numpy as np
from math import sqrt
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging  # Import logging module
from PIL import Image

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

def plot_stimulus_with_velocities(velocidades, amplitude_list, duration_list, segmento, global_max_velocity, forma, amplitud, duracion, frecuencia):
    from matplotlib.ticker import MultipleLocator
    # Creamos los subplots sin compartir el eje x
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    color_map = plt.get_cmap('tab10')
    body_parts = list(velocidades.keys())

    # Gráfico de velocidades
    for i, (part, vel) in enumerate(velocidades.items()):
        if len(vel) > 0:
            # Usamos iniciales para las etiquetas y líneas más delgadas
            ax1.plot(vel, label=f'V_{part}', color=color_map(i % 10), linewidth=1)
            ax1.axhline(np.mean(vel), linestyle='--', color=color_map(i % 10), label=f'M_{part}', linewidth=1)

    # Dibujamos la línea vertical en x=0 sin etiqueta
    ax1.axvline(0, color='red', linestyle='--', linewidth=1)  # Sin etiqueta

    # Configuración del eje x para el gráfico de velocidades
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Velocidad (unidades/segundo)')
    ax1.set_ylim(0, 2500)
    ax1.set_xlim(0, 400)

    # Añadir marcas menores en el eje x cada 10 frames (100 ms)
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.tick_params(axis='x', which='minor', length=4)

    # Colocar la leyenda dentro del área del gráfico
    ax1.legend(loc='upper right', fontsize='small', ncol=2, framealpha=0.5)

    if not amplitude_list or not duration_list:
        logging.warning(f"Advertencia: No hay datos válidos para ploteo de estímulo en segmento {segmento}.")
        return "", 0, 0

    # Generar x_vals y y_vals para el estímulo
    x_vals = [100]
    y_vals = [0]
    start_frame = 100
    current_frame = start_frame

    for amp, dur in zip(amplitude_list, duration_list):
        frames_to_add = dur
        next_frame = current_frame + frames_to_add
        x_vals.extend([current_frame, next_frame])
        y_vals.extend([amp / 1000, amp / 1000])
        current_frame = next_frame

    # Convertir x_vals a milisegundos para el gráfico de estímulo
    x_vals_ms = [x * 10 for x in x_vals]
    start_time_ms = start_frame * 10
    end_time_ms = current_frame * 10

    # Gráfico de estímulo
    ax2.step(x_vals_ms, y_vals, color='blue', where='post', linewidth=1)
    ax2.set_xlabel('Milisegundos (ms)')
    ax2.set_ylabel('Amplitud (μA)')
    ax2.set_xlim(0, 4000)
    ax2.set_ylim(-160, 160)
    ax2.set_yticks([-160, -80, 0, 80, 160])

    # Añadir línea horizontal en y=0
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Añadir texto con los parámetros del estímulo
    estimulo_params_text = f"Forma: {forma}, Amplitud: {amplitud} μA, Duración: {duracion} ms, Frecuencia: {frecuencia} Hz"
    ax2.text(0.95, 0.95, estimulo_params_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Sombrear la región del estímulo
    ax1.axvspan(start_frame, current_frame, color='blue', alpha=0.3)
    ax2.axvspan(start_time_ms, end_time_ms, color='blue', alpha=0.3)

    # Añadir marcas menores en el eje x de ax2 cada 100 ms
    ax2.xaxis.set_minor_locator(MultipleLocator(100))
    ax2.tick_params(axis='x', which='minor', length=4)

    # Ajustamos el espaciado entre los dos gráficos
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout(pad=2.0)

    # Guardar la imagen
    graph_image_path = f"temp_graph_{segmento}.png"
    plt.savefig(graph_image_path, bbox_inches='tight')
    plt.close()

    end_frame = current_frame

    return graph_image_path, start_frame, end_frame

def plot_trajectories_over_frames(csv_path, body_parts, start_frame, end_frame):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Ajustamos el tamaño de la figura
    fig, ax = plt.subplots(figsize=(10, 6))

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

            # Usamos iniciales para las etiquetas y líneas más delgadas
            ax.plot(frames, x_vals, label=f'X_{part}', color=color_map(i % 10), linestyle='-', linewidth=1)
            ax.plot(frames, y_vals, label=f'Y_{part}', color=color_map(i % 10), linestyle='--', linewidth=1)

    # Sombrear la región del estímulo
    ax.axvspan(start_frame, end_frame, color='blue', alpha=0.3)

    ax.set_xlabel('Frames')
    ax.set_ylabel('Coordenadas (px)')
    ax.set_xlim(0, 400)
    ax.set_title('Trayectorias (x e y) a lo largo del tiempo')

    # Colocar la leyenda dentro del área del gráfico
    ax.legend(loc='upper right', fontsize='small', ncol=2, framealpha=0.5)

    # Añadir marcas menores en el eje x cada 10 frames
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='x', which='minor', length=4)

    # Agregar eje x secundario en milisegundos en la parte inferior
    def ms_from_frames(x):
        return x * 10  # 1 frame = 10 ms

    def frames_from_ms(x):
        return x / 10

    ax_secondary = ax.secondary_xaxis('bottom', functions=(ms_from_frames, frames_from_ms))
    ax_secondary.set_xlabel("Milisegundos (ms)")
    ax_secondary.set_xlim(ms_from_frames(0), ms_from_frames(400))
    ax_secondary.spines['bottom'].set_position(('outward', 40))
    ax_secondary.xaxis.set_minor_locator(MultipleLocator(100))
    ax_secondary.tick_params(axis='x', which='minor', length=4)

    # Remover etiquetas del eje x en la parte superior
    ax.tick_params(axis='x', which='both', labeltop=False)

    trajectory_image_path = "temp_trajectory_over_frames_plot.png"
    plt.savefig(trajectory_image_path, bbox_inches='tight')
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
                            graph_image_path, start_frame, end_frame = plot_stimulus_with_velocities(
                                velocidades, amplitude_list, duration_list, nombre_segmento, global_max_velocity,
                                forma=row['Forma del Pulso'], amplitud=row['Amplitud (μA)'], duracion=row['Duración (ms)'], frecuencia=row['Frecuencia (Hz)']
                            )

                            # Generate trajectory plots with consistent shading
                            trajectory_image_path = plot_trajectories_over_frames(csv_path, body_parts, start_frame, end_frame)

                            # Add images to PDF
                            pdf.add_page()
                            pdf.set_font("ArialUnicode", size=12)
                            pdf.multi_cell(0, 10, f"Segmento: {nombre_segmento} (Número Ordinal: {segment_row['NumeroOrdinal']})")
                            # Añadir imágenes al PDF con el mismo ancho
                            pdf.image(graph_image_path, x=10, y=20, w=190)
                            pdf.image(trajectory_image_path, x=10, y=150, w=190)

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
                        if len(vel) > 0:
                            if np.max(vel) > global_max_velocity:
                                global_max_velocity = np.max(vel)

# Generate PDF for each row in Stimuli_information and log any failures
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row, global_max_velocity)
