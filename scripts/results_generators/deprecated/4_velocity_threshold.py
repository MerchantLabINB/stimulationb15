# Importaciones y configuración inicial
import os
import sys
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import seaborn as sns

from scipy.signal import savgol_filter
import re
import shutil

# Configuración del logging
log_file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\processing_log.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Añadir la ruta a Stimulation.py
stimulation_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\scripts\GUI_pattern_generator'
sys.path.append(stimulation_path)

# Importar la función estimulo de Stimulation.py
from Stimulation import estimulo

# Directorios
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
segmented_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_segmentados.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos'
output_variability_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\variability_plots\thresholds'

# Asegurarse de que el directorio de salida existe
if not os.path.exists(output_variability_dir):
    os.makedirs(output_variability_dir)

# Cargar archivos CSV
stimuli_info = pd.read_csv(stimuli_info_path)
segmented_info = pd.read_csv(segmented_info_path)

# Filtrar entradas donde 'Descartar' es 'No'
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']

# Verificar si stimuli_info no está vacío
if stimuli_info.empty:
    logging.error("El DataFrame stimuli_info está vacío después de filtrar por 'Descartar' == 'No'. Verifica el archivo CSV.")
    sys.exit("El DataFrame stimuli_info está vacío. No hay datos para procesar.")

# Imprimir las columnas de stimuli_info para verificar los nombres
print("Columnas en stimuli_info:", stimuli_info.columns)
print(stimuli_info['Dia experimental'].unique())

# Articulaciones (body parts) con nombres de paletas de colores
body_parts_colors = {
    'Frente': 'Blues',
    'Hombro': 'Oranges',
    'Codo': 'Greens',
    'Muñeca': 'Reds',
    'NudilloCentral': 'Purples',
    'DedoMedio': 'pink',      # Usamos 'pink' en minúsculas
    'Braquiradial': 'Greys',
    'Bicep': 'YlOrBr'
}

body_parts = list(body_parts_colors.keys())

def sanitize_filename(filename):
    # Reemplaza los caracteres inválidos por guiones bajos
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# Función para calcular la distancia entre dos puntos
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Función para encontrar el archivo CSV correspondiente basado en la cámara y el nombre del segmento
def encontrar_csv(camara_lateral, nombre_segmento):
    try:
        for file_name in os.listdir(csv_folder):
            if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
                csv_path = os.path.join(csv_folder, file_name)
                print(f'Archivo CSV encontrado: {csv_path}')
                return csv_path
        print(f'Archivo CSV no encontrado para la cámara: {camara_lateral}, segmento: {nombre_segmento}')
        return None
    except Exception as e:
        logging.error(f'Error al acceder a los archivos CSV: {e}')
        return None

# Función de suavizado usando Savitzky-Golay
def suavizar_datos(data, window_length=21, polyorder=3):
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

# Función para calcular velocidades y posiciones para cada articulación con suavizado
# Función para calcular velocidades y posiciones para cada articulación con suavizado
def calcular_velocidades(csv_path):
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        print(f'Columnas en {csv_path}: {df.columns}')

        velocidades = {}
        posiciones = {}

        for part in body_parts:
            x_col = (df.columns[1][0], part, 'x')
            y_col = (df.columns[1][0], part, 'y')
            likelihood_col = (df.columns[1][0], part, 'likelihood')

            valid_frames = df[df[likelihood_col] > 0.1].shape[0]
            logging.info(f'{part} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')

            df_filtered = df[df[likelihood_col] > 0.1]
            if df_filtered.empty:
                logging.warning(f'No hay datos suficientes para {part} en {csv_path} después de filtrar por likelihood.')
                velocidades[part] = np.array([])  # Empty arrays to avoid breaking logic later
                posiciones[part] = {'x': np.array([]), 'y': np.array([])}
                continue

            x = df_filtered[x_col].values
            y = df_filtered[y_col].values

            delta_x = np.diff(x)
            delta_y = np.diff(y)

            distancias = np.hypot(delta_x, delta_y)
            delta_t = 1 / 100  # 100 fps
            velocidad_part = distancias / delta_t

            velocidad_part = suavizar_datos(velocidad_part, window_length=21)

            velocidades[part] = velocidad_part
            posiciones[part] = {'x': x, 'y': y}

        return velocidades, posiciones
    except Exception as e:
        logging.error(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        return {}, {}


# Función para calcular aceleraciones
def calcular_aceleraciones(velocidades):
    aceleraciones = {}
    for part, vel in velocidades.items():
        vel = np.array(vel)
        if len(vel) < 2:
            # No se puede calcular gradiente con menos de 2 puntos
            logging.warning(f'No se puede calcular aceleración para {part} porque la velocidad tiene longitud {len(vel)}')
            aceleracion_part = np.array([])
        else:
            # Utilizar np.gradient para calcular la derivada numérica
            aceleracion_part = np.gradient(vel)
        aceleraciones[part] = aceleracion_part
    return aceleraciones

# Convertir tiempo de microsegundos a frames
def us_to_frames(duracion_us):
    return duracion_us / 10000  # 1 frame = 10,000 µs

# Función para generar el estímulo desde parámetros usando la lógica de Stimulation.py
def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    try:
        forma = forma.strip().lower()  # Asegurar minúsculas
        print(f"Generando estímulo con forma: {forma}, amplitud: {amplitud}, duración: {duracion}, frecuencia: {frecuencia}, duración del pulso: {duracion_pulso}, compensar: {compensar}")

        # Verificar parámetros válidos
        if duracion <= 0 or frecuencia <= 0 or duracion_pulso <= 0:
            logging.error(f"Parámetros inválidos: duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}")
            return [], []

        # Generar estímulo usando la función estimulo
        lista_amplitud, lista_tiempo = estimulo(
            forma=forma, amplitud=amplitud, duracion=duracion,
            frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
        )

        # Asegurar generación correcta del estímulo
        if not lista_amplitud or not lista_tiempo:
            logging.error(f"Estímulo inválido con parámetros: forma={forma}, amplitud={amplitud}, duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}, compensar={compensar}")
            return [], []

        # Convertir todos los tiempos del estímulo (en µs) a frames
        lista_tiempo = [us_to_frames(tiempo) for tiempo in lista_tiempo]

        return lista_amplitud, lista_tiempo
    except Exception as e:
        logging.error(f'Error al generar estímulo: {e}')
        return [], []

# Función para generar gráfico de grupo
def plot_group_graph(velocities_list, positions_x_list, positions_y_list,
                     threshold, amplitude_list, duration_list,
                     group_name, body_part, start_frame, current_frame, 
                     mean_vel_pre, std_vel_pre, amplitud_real, output_image_dir):  # Add output_image_dir parameter
    print(f"Generating graph for {body_part} in group {group_name}...")

    try:
        dia_experimental, forma_pulso, duracion_ms = group_name

        # Número de ensayos
        num_trials = len(velocities_list)

        # Obtener el colormap para la articulación y revertirlo para ir de oscuro a claro
        cmap_name = body_parts_colors.get(body_part, 'viridis')
        cmap = plt.get_cmap(cmap_name + '_r')  # '_r' para invertir el colormap
        colors = cmap(np.linspace(0, 1, num_trials))

        # Configuración de la figura y subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 14),
                                 gridspec_kw={'height_ratios': [4, 4, 4, 2]})
        ax1, ax2, ax3, ax4 = axes

        # Gráfico de trayectorias en ax1
        for idx, (x, y) in enumerate(zip(positions_x_list, positions_y_list)):
            frames = np.arange(len(x))
            color = colors[idx]
            ax1.plot(frames, x, color=color, linestyle='-', alpha=0.5)
            ax1.plot(frames, y, color=color, linestyle='--', alpha=0.5)

        ax1.set_title(f'Trayectorias - {body_part}')
        ax1.set_xlabel('Frames')
        ax1.set_ylabel('Coordenadas (px)')
        ax1.set_xlim(0, 400)
        ax1.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

        # Gráfico de velocidades en ax2
        passed_threshold_count = 0  # Contador de ensayos que pasan el umbral
        for idx, vel in enumerate(velocities_list):
            frames_vel = np.arange(len(vel))
            color = colors[idx]
            ax2.plot(frames_vel, vel, color=color, alpha=0.5)

            # Resaltar la parte de la curva que está por encima del umbral durante el estímulo
            within_stimulus = (frames_vel >= start_frame) & (frames_vel <= current_frame)
            above_threshold = (vel > threshold) & within_stimulus

            if np.any(above_threshold):
                passed_threshold_count += 1  # Contar este ensayo como que pasa el umbral

            ax2.plot(frames_vel[above_threshold], vel[above_threshold], color='red', linewidth=1, alpha=0.7)

        # Añadir líneas horizontales para el umbral y la media pre-estímulo
        ax2.axhline(threshold, color='k', linestyle='--', label=f'Umbral ({threshold:.2f}), {passed_threshold_count}/{num_trials} superaron')
        ax2.axhline(mean_vel_pre, color='gray', linestyle=':', label=f'Media Pre-estímulo ({mean_vel_pre:.2f})')

        ax2.set_title(f'Velocidades - {body_part}')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Velocidad (unidades/segundo)')
        ax2.set_xlim(0, 400)
        ax2.axvspan(start_frame, current_frame, color='blue', alpha=0.1)
        ax2.legend(loc='upper right', fontsize=9)

        # Gráfico de aceleraciones en ax3
        for idx, vel in enumerate(velocities_list):
            aceleracion = np.gradient(vel)
            frames_acc = np.arange(len(aceleracion))
            color = colors[idx]
            ax3.plot(frames_acc, aceleracion, color=color, alpha=0.5)

        ax3.set_title(f'Aceleraciones - {body_part}')
        ax3.set_xlabel('Frames')
        ax3.set_ylabel('Aceleración (unidades/segundo²)')
        ax3.set_xlim(0, 400)
        ax3.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

        # Gráfico del estímulo en ax4
        x_vals, y_vals = [start_frame], [0]
        current_frame_stimulus = start_frame
        for amp, dur in zip(amplitude_list, duration_list):
            next_frame = current_frame_stimulus + dur
            x_vals.extend([current_frame_stimulus, next_frame])
            y_vals.extend([amp / 1000, amp / 1000])
            current_frame_stimulus = next_frame

        ax4.step(x_vals, y_vals, color='blue', where='post', linewidth=1, 
                 label=f'Amplitud: {amplitud_real:.2f} μA')

        ax4.set_xlabel('Frames')
        ax4.set_ylabel('Amplitud (μA)')
        ax4.set_xlim(0, 400)
        ax4.set_ylim(-160, 160)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvspan(start_frame, current_frame, color='blue', alpha=0.3)

        # Añadir el texto de los parámetros del estímulo
        estimulo_params_text = f"Forma: {forma_pulso}, Duración: {duracion_ms} ms, Día: {dia_experimental}"
        ax4.text(0.95, 0.95, estimulo_params_text, transform=ax4.transAxes, fontsize=8,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Mostrar la leyenda
        ax4.legend(loc='lower right', fontsize=9)

        # Ajustar espacios entre subplots
        plt.subplots_adjust(hspace=0.4)

        # Save the image to the provided directory
        graph_image_path = os.path.join(output_image_dir, f"graph_{sanitize_filename(dia_experimental)}_{sanitize_filename(forma_pulso)}_{duracion_ms}ms_{sanitize_filename(body_part)}.png")
        plt.savefig(graph_image_path, dpi=300)
        plt.close()

        print(f"Graph for {body_part} saved at {graph_image_path}")
        return graph_image_path

    except Exception as e:
        logging.error(f"Error al generar gráfico de grupo para {body_part} en el grupo {group_name}: {e}")
        print(f"Error generating graph for {body_part}: {e}")
        return ''

# Función principal para recolectar datos y generar gráficos
def collect_velocity_threshold_data():
    total_trials = 0
    all_movement_data = []  # To collect movement-related data across trials
    thresholds_data = []  # To store thresholds for each body part
    graphs_info = []  # To store information about generated graphs

    # Group by 'Dia experimental' and 'Forma del Pulso'
    # Modify the group_by to include Duración (ms) for per-stimulus thresholding
    grouped_stimuli = stimuli_info.groupby(['Dia experimental', 'Forma del Pulso', 'Duración (ms)'])

    for group_name, group_df in grouped_stimuli:
        dia_experimental, forma_pulso, duracion_ms = group_name
        print(f'Procesando grupo: Día {dia_experimental}, Forma del Pulso {forma_pulso}, Duración {duracion_ms}ms')

        # Pre-stim velocities across body parts
        pre_stim_velocities = {part: [] for part in body_parts}

        # Iterate over all trials in the group for threshold calculation
        for index, row in group_df.iterrows():
            camara_lateral = row['Camara Lateral']
            duracion_ms = row['Duración (ms)']  # Extract 'duracion_ms' directly from the row

            if pd.notna(camara_lateral):
                matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                if not matching_segment.empty:
                    matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                    # Extract pre-stimulus velocity data for threshold calculation
                    for _, segment_row in matching_segment_sorted.iterrows():
                        nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                        csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                        if csv_path:
                            velocidades, posiciones = calcular_velocidades(csv_path)
                            
                            for part in body_parts:
                                vel = velocidades.get(part, [])
                                if len(vel) > 0:
                                    vel_pre_stim = vel[:100]  # Use first 100 frames for pre-stimulus
                                    vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]  # Remove NaN
                                    pre_stim_velocities[part].extend(vel_pre_stim)
                                else:
                                    print(f'Trial {nombre_segmento}, articulación {part}, sin datos de velocidad.')

        # Calculate threshold per body part based on all pre-stimulus velocities in this group
        thresholds = {}
        mean_std_per_part = {}

        # For each body part, compute the threshold
        for part in body_parts:
            vel_list = pre_stim_velocities[part]
            print(f'LISTA VELOCIDADES{part}: {len(vel_list)} valores de velocidad pre-estímulo recolectados.')

            if len(vel_list) < 10:
                logging.warning(f'Datos insuficientes para calcular el umbral para {part} en el grupo {group_name}. Se requieren al menos 10 valores.')
                continue

            # Calculate mean and standard deviation of pre-stim velocities
            mean_vel_pre = np.nanmean(vel_list)
            std_vel_pre = np.nanstd(vel_list)
            threshold = mean_vel_pre + 2 * std_vel_pre  # Threshold is mean + 2*std
            thresholds[part] = threshold
            mean_std_per_part[part] = (mean_vel_pre, std_vel_pre)
            print(f"Umbral para {part} en el grupo {group_name}: {threshold}")

            # Store threshold data for future analysis
            thresholds_data.append({
                'body_part': part,
                'Dia experimental': dia_experimental,
                'Forma del Pulso': forma_pulso,
                'Duración (ms)': duracion_ms,
                'threshold': threshold,
                'mean_pre_stim': mean_vel_pre,
                'std_pre_stim': std_vel_pre,
                'num_pre_stim_values': len(vel_list)
            })

        # Now, we process each body part and generate graphs where possible
        for part in body_parts:
            if part not in thresholds:
                logging.warning(f'Se omite la generación de gráficos para {part} porque no se calculó un umbral.')
                continue  # Skip this body part if no threshold was computed

            threshold = thresholds[part]
            mean_vel_pre, std_vel_pre = mean_std_per_part[part]
            max_amplitud_ensayos = group_df[group_df['Amplitud (microA)'] == group_df['Amplitud (microA)'].max()]

            # Process all trials and collect velocities for this body part
            group_velocities = {bp: [] for bp in body_parts}  # Initialize velocities for all body parts
            group_positions = {bp: {'x': [], 'y': []} for bp in body_parts}  # Initialize positions for all body parts

            movement_trials = 0  # Count trials where movement exceeds the threshold
            total_trials_part = 0  # Count total trials for this body part

            # Process trials with the maximum amplitude

            for index, row in max_amplitud_ensayos.iterrows():
                camara_lateral = row['Camara Lateral']
                duracion_ms = row['Duración (ms)']

                if pd.notna(camara_lateral):
                    matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                    if not matching_segment.empty:
                        matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                        # Generate stimulus to get 'start_frame' and 'current_frame'
                        compensar = False if duracion_ms == 1000 else True
                        amplitude_list, duration_list = generar_estimulo_desde_parametros(
                            row['Forma del Pulso'],
                            row['Amplitud (microA)'] * 1000,  # Convert to μA
                            duracion_ms * 1000,  # Convert to µs
                            row['Frecuencia (Hz)'],
                            200,  # Pulse duration in µs
                            compensar=compensar
                        )

                        # Define start and current frame based on stimulus duration
                        start_frame = 100
                        current_frame = int(start_frame + sum(duration_list))

                        for _, segment_row in matching_segment_sorted.iterrows():
                            nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                            csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                            if csv_path:
                                print(f'Procesando segmento {nombre_segmento}, CSV encontrado: {csv_path}')
                                velocidades, posiciones = calcular_velocidades(csv_path)

                                if part in velocidades:
                                    vel = velocidades[part]
                                    pos = posiciones[part]
                                    if len(vel) > 0:
                                        group_velocities[part].append(vel)
                                        group_positions[part]['x'].append(pos['x'])
                                        group_positions[part]['y'].append(pos['y'])

                                        # Count trials where velocity exceeds threshold during stimulus
                                        vel_stimulus = vel[start_frame:current_frame]
                                        if np.any(vel_stimulus > threshold):
                                            movement_trials += 1

                                        total_trials_part += 1
            if group_velocities[part]:
                graph_image_path = plot_group_graph(
                    group_velocities[part], group_positions[part]['x'], group_positions[part]['y'],
                    threshold, amplitude_list, duration_list,
                    (dia_experimental, forma_pulso, duracion_ms), part, start_frame, current_frame,
                    mean_vel_pre, std_vel_pre, row['Amplitud (microA)'],
                    output_variability_dir
                )
            # Calculate the proportion of trials where movement exceeds the threshold
            no_movement_trials = total_trials_part - movement_trials
            proportion_movement = movement_trials / total_trials_part if total_trials_part > 0 else 0

            # Store movement data
            all_movement_data.append({
                'body_part': part,
                'Dia experimental': dia_experimental,
                'Forma del Pulso': forma_pulso,
                'Duración (ms)': duracion_ms,
                'movement_trials': movement_trials,
                'total_trials': total_trials_part,
                'no_movement_trials': no_movement_trials,
                'proportion_movement': proportion_movement
            })

            # Generate graph for this body part
            if len(group_velocities[part]) == 0:
                logging.warning(f'No se generó gráfico para {part}, sin datos válidos para generar un gráfico.')
                continue

            # Set the output directory
            output_image_dir = os.path.join(output_variability_dir, 'group_graphs')
            if not os.path.exists(output_image_dir):
                os.makedirs(output_image_dir)

            # Call the plotting function to generate the graph
            graph_image_path = plot_group_graph(
                group_velocities[part], group_positions[part]['x'], group_positions[part]['y'],
                threshold, amplitude_list, duration_list,
                (dia_experimental, forma_pulso, duracion_ms), part, start_frame, current_frame,
                mean_vel_pre, std_vel_pre, row['Amplitud (microA)'],
                output_image_dir  # Pass the output directory for graph saving
            )

            if graph_image_path and os.path.exists(graph_image_path):
                output_image_path = os.path.join(output_image_dir, f'grupo_{sanitize_filename(dia_experimental)}_{sanitize_filename(forma_pulso)}_{duracion_ms}ms_{sanitize_filename(part)}.png')

                # Remove old file if it exists, then move new file to output
                if os.path.exists(output_image_path):
                    try:
                        os.remove(output_image_path)
                        print(f"Archivo anterior eliminado: {output_image_path}")
                    except Exception as e:
                        logging.error(f"Error al eliminar el archivo antiguo: {output_image_path}, Error: {e}")
                        print(f"Error al eliminar archivo antiguo: {output_image_path}")

                try:
                    shutil.move(graph_image_path, output_image_path)
                    print(f'Gráfico guardado: {output_image_path}')

                    # Append graph info for later analysis
                    graphs_info.append({
                        'Dia experimental': dia_experimental,
                        'Forma del Pulso': forma_pulso,
                        'Duración (ms)': duracion_ms,
                        'body_part': part,
                        'graph_path': output_image_path
                    })
                except Exception as e:
                    logging.error(f"Error al mover el archivo nuevo: {graph_image_path} a {output_image_path}, Error: {e}")
                    print(f"Error al mover archivo: {graph_image_path}")

    # Create a DataFrame from collected movement data
    counts_df = pd.DataFrame(all_movement_data)
    counts_df.to_csv(os.path.join(output_variability_dir, 'movement_counts_summary.csv'), index=False)

    # Create and save the thresholds DataFrame
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_df.to_csv(os.path.join(output_variability_dir, 'thresholds_summary.csv'), index=False)

    # Save the graph information to a CSV file
    graphs_df = pd.DataFrame(graphs_info)
    graphs_df.to_csv(os.path.join(output_variability_dir, 'graphs_generated.csv'), index=False)
    print(f"Información de los gráficos guardada en {os.path.join(output_variability_dir, 'graphs_generated.csv')}")

    return counts_df

# Función para analizar los mejores bodyparts y estímulos
def analyze_best_bodyparts_and_stimuli(counts_df):
    # Create a column to identify the stimulus
    counts_df['Estímulo'] = counts_df['Forma del Pulso'] + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

    # Sort by proportion of movement to identify the best body parts and stimuli
    sorted_df = counts_df.sort_values(by='proportion_movement', ascending=False)

    # Display top 5 body parts with the highest proportion of movement
    top_bodyparts = sorted_df.groupby('body_part')['proportion_movement'].mean().sort_values(ascending=False)
    print("Top articulaciones con mayor proporción de movimiento:")
    print(top_bodyparts.head(5))

    # Display top 5 stimuli with the highest proportion of movement
    top_stimuli = sorted_df.groupby('Estímulo')['proportion_movement'].mean().sort_values(ascending=False)
    print("\nTop estímulos con mayor proporción de movimiento:")
    print(top_stimuli.head(5))

    # Save results to CSV files
    top_bodyparts.to_csv(os.path.join(output_variability_dir, 'top_bodyparts.csv'))
    top_stimuli.to_csv(os.path.join(output_variability_dir, 'top_stimuli.csv'))

# Función para generar el heatmap
def plot_heatmap(counts_df):
    # Crear una columna para identificar el estímulo
    counts_df['Estímulo'] = counts_df['Forma del Pulso'] + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

    # Pivotear los datos para el heatmap de proporción
    pivot_prop = counts_df.pivot_table(
        index='body_part',
        columns=['Dia experimental', 'Estímulo'],
        values='proportion_movement',
        aggfunc='mean'
    )

    # Pivotear los datos para los counts
    pivot_movement = counts_df.pivot_table(
        index='body_part',
        columns=['Dia experimental', 'Estímulo'],
        values='movement_trials',
        aggfunc='sum'
    )

    pivot_total = counts_df.pivot_table(
        index='body_part',
        columns=['Dia experimental', 'Estímulo'],
        values='total_trials',
        aggfunc='sum'
    )

    # Asegurar que los pivotes tengan los mismos índices y columnas
    common_index = pivot_prop.index.union(pivot_movement.index).union(pivot_total.index)
    common_columns = pivot_prop.columns.union(pivot_movement.columns).union(pivot_total.columns)

    pivot_prop = pivot_prop.reindex(index=common_index, columns=common_columns)
    pivot_movement = pivot_movement.reindex(index=common_index, columns=common_columns)
    pivot_total = pivot_total.reindex(index=common_index, columns=common_columns)

    # Crear una matriz de anotaciones con 'movement_trials/total_trials'
    annot_matrix = pivot_movement.fillna(0).astype(int).astype(str) + '/' + pivot_total.fillna(0).astype(int).astype(str)

    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_prop, annot=annot_matrix, fmt='', cmap='viridis')
    plt.title('Proporción de Movimiento Detectado por Articulación, Día y Estímulo')
    plt.xlabel('Día Experimental y Estímulo')
    plt.ylabel('Articulación')
    plt.tight_layout()
    plt.savefig(os.path.join(output_variability_dir, 'heatmap_bodypart_day_stimulus.png'))
    print('Gráfico heatmap_bodypart_day_stimulus.png guardado.')
    plt.close()

# Código principal
if __name__ == "__main__":
    # Llamar a collect_velocity_threshold_data
    counts_df = collect_velocity_threshold_data()
    print("Counts DataFrame after collection:", counts_df.shape)
    print(counts_df.head())

    # Analizar los mejores bodyparts y estímulos
    analyze_best_bodyparts_and_stimuli(counts_df)

    # Generar heatmap
    plot_heatmap(counts_df)