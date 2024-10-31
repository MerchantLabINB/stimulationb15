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
import glob  # Importar el módulo glob

# Configuración del logging
refactored_log_file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_processing_log.txt'
logging.basicConfig(filename=refactored_log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Añadir la ruta a Stimulation.py
stimulation_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\scripts\GUI_pattern_generator'
sys.path.append(stimulation_path)

# Importar la función estimulo de Stimulation.py
from Stimulation import estimulo

# Directorios
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
segmented_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_segmentados.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\xv_lat-Br-2024-10-02\videos'
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_variability_plots'

# Asegurarse de que el directorio de salida existe
if not os.path.exists(output_comparisons_dir):
    os.makedirs(output_comparisons_dir)

# Cargar archivos CSV
stimuli_info = pd.read_csv(stimuli_info_path)
segmented_info = pd.read_csv(segmented_info_path)

# Filtrar entradas donde 'Descartar' es 'No'
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']

# Normalizar 'Forma del Pulso' a minúsculas para evitar problemas de coincidencia
stimuli_info['Forma del Pulso'] = stimuli_info['Forma del Pulso'].str.lower()

# Verificar si stimuli_info no está vacío
if stimuli_info.empty:
    logging.error("El DataFrame stimuli_info está vacío después de filtrar por 'Descartar' == 'No'. Verifica el archivo CSV.")
    sys.exit("El DataFrame stimuli_info está vacío. No hay datos para procesar.")

# Articulaciones (body parts) con nombres de paletas de colores
body_parts_colors = {
    'Frente': 'Blues',
    'Hombro': 'Oranges',
    'Codo': 'Greens',
    'Muñeca': 'Reds',
    'NudilloCentral': 'Purples',
    'DedoMedio': 'pink',  # Usamos 'pink' en minúsculas
    'Braquiradial': 'Greys',
    'Bicep': 'YlOrBr'
}

# Diccionario de colores específicos para cada articulación
body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muñeca': 'red',
    'NudilloCentral': 'purple',
    'DedoMedio': 'pink',
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}

body_parts = list(body_parts_colors.keys())

def sanitize_filename(filename):
    # Reemplaza los caracteres inválidos por guiones bajos
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# Función para calcular la distancia entre dos puntos
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Función modificada para encontrar el archivo CSV correspondiente basado en la cámara y el nombre del segmento
def encontrar_csv(camara_lateral, nombre_segmento):
    try:
        # Extract digits from the segment name
        match = re.search(r'segment_(\d+_\d+)', nombre_segmento)
        if match:
            digits = match.group(1)
            pattern = f"{camara_lateral}_{digits}*filtered.csv"
            search_pattern = os.path.join(csv_folder, pattern)
            matching_files = glob.glob(search_pattern)
            print(f'Searching for CSV files with pattern: {search_pattern}')
            if matching_files:
                csv_path = matching_files[0]
                print(f'Archivo CSV filtrado encontrado: {csv_path}')
                return csv_path
            else:
                print(f'Archivo CSV filtrado no encontrado para la cámara: {camara_lateral}, segmento: {nombre_segmento}')
                return None
        else:
            print(f'No se pudieron extraer los dígitos del nombre del segmento: {nombre_segmento}')
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
def calcular_velocidades(csv_path):
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])

        # Flatten the MultiIndex columns into single-level column names
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        # print(f"Columns in df: {df.columns}")

        # Remove the 'scorer_bodyparts_coords' column if it exists
        if 'scorer_bodyparts_coords' in df.columns:
            df = df.drop(columns=['scorer_bodyparts_coords'])

        # Adjust the body_parts to match columns
        body_parts_adjusted = [part.replace('ñ', 'n').replace(' ', '_') for part in body_parts]

        velocidades = {}
        posiciones = {}

        for part_original, part in zip(body_parts, body_parts_adjusted):
            # Initialize column variables
            x_col = None
            y_col = None
            likelihood_col = None

            # Find columns that contain the body part name and end with '_x', '_y', or '_likelihood'
            for col in df.columns:
                if col.endswith('_x') and part in col:
                    x_col = col
                elif col.endswith('_y') and part in col:
                    y_col = col
                elif col.endswith('_likelihood') and part in col:
                    likelihood_col = col

            # Check if all necessary columns are found
            if not x_col or not y_col or not likelihood_col:
                print(f"Columns for {part_original} are incomplete.")
                continue

            print(f"Using columns for {part_original}: x_col={x_col}, y_col={y_col}, likelihood_col={likelihood_col}")

            # Filter rows based on likelihood
            df_filtered = df[df[likelihood_col] > 0.1]
            valid_frames = len(df_filtered)
            logging.info(f'{part_original} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')

            if df_filtered.empty:
                logging.warning(f'No hay datos suficientes para {part_original} en {csv_path} después de filtrar por likelihood.')
                velocidades[part_original] = np.array([])
                posiciones[part_original] = {'x': np.array([]), 'y': np.array([])}
                continue

            # Extract x and y positions
            x = df_filtered[x_col].values
            y = df_filtered[y_col].values

            # Calculate velocities
            delta_x = np.diff(x)
            delta_y = np.diff(y)
            distancias = np.hypot(delta_x, delta_y)
            delta_t = 1 / 100  # Assuming 100 fps
            velocidad_part = distancias / delta_t

            # Smooth velocities
            velocidad_part = suavizar_datos(velocidad_part, window_length=21)

            # Store velocities and positions
            velocidades[part_original] = velocidad_part
            posiciones[part_original] = {'x': x, 'y': y}

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

# Definir los grupos de estímulos según tus especificaciones
stimulus_groups = [
    {
        'name': 'Rectangulares',
        'stimuli': [
            {'Forma del Pulso': 'rectangular', 'Duración (ms)': 500},
            {'Forma del Pulso': 'rectangular', 'Duración (ms)': 1000}
        ]
    },
    {
        'name': 'Rombos',
        'stimuli': [
            {'Forma del Pulso': 'rombo', 'Duración (ms)': 500},
            {'Forma del Pulso': 'rombo', 'Duración (ms)': 750},
            {'Forma del Pulso': 'rombo', 'Duración (ms)': 1000}
        ]
    },
    {
        'name': 'Rampas_y_Triple_Rombo',
        'stimuli': [
            {'Forma del Pulso': 'rampa ascendente', 'Duración (ms)': None},
            {'Forma del Pulso': 'rampa descendente', 'Duración (ms)': None},
            {'Forma del Pulso': 'triple rombo', 'Duración (ms)': None}
        ]
    }
]

# Función para generar gráficos de grupo con múltiples estímulos
def plot_group_graphs(group_velocities_dict, group_positions_dict,
                      threshold_dict, amplitude_list_dict, duration_list_dict,
                      group_name, body_part, dia_experimental,
                      start_frame_dict, current_frame_dict,
                      mean_vel_pre_dict, std_vel_pre_dict, amplitud_real_dict,
                      y_max_velocity_dict, output_pdf_path,
                      trial_indices_per_stimulus,
                      all_movement_data,
                      movement_ranges_dict,
                      form_dict,
                      duration_ms_dict,
                      frequency_dict):
    num_stimuli = len(group_velocities_dict)

    # Ajustar el tamaño de la figura basado en el número de estímulos
    fig_width = max(10, num_stimuli * 8)  # Ajuste dinámico del ancho
    fig_height = 20

    fig, axes = plt.subplots(5, num_stimuli, figsize=(fig_width, fig_height),
                             gridspec_kw={'height_ratios': [4, 4, 4, 2, 2]})
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Asegurar que axes es un arreglo 2D
    if num_stimuli == 1:
        axes = np.array([axes]).reshape(5, 1)

    # Calcular el límite máximo del eje y para los gráficos de velocidades (ax2) y duraciones de movimiento (ax5)
    group_y_max_velocity = 0
    for stimulus_key in group_velocities_dict.keys():
        if y_max_velocity_dict.get(stimulus_key):
            group_y_max_velocity = max(group_y_max_velocity, y_max_velocity_dict[stimulus_key])

    # Establecer un límite superior común para las velocidades
    group_y_max_velocity = max(group_y_max_velocity, 1)  # Asegurar al menos 1

    # Calcular el límite máximo del eje y para los gráficos de duraciones de movimiento (ax5)
    group_max_y_lim = 0
    for movement_ranges in movement_ranges_dict.values():
        if movement_ranges:
            ensayos = [mr['Ensayo'] for mr in movement_ranges]
            if ensayos:
                group_max_y_lim = max(group_max_y_lim, max(ensayos))

    # Asegurar que el límite máximo del eje y sea al menos 1
    group_max_y_lim = max(group_max_y_lim, 1)

    for idx, (stimulus_key, velocities_list) in enumerate(group_velocities_dict.items()):
        positions_x_list = group_positions_dict[stimulus_key]['x']
        positions_y_list = group_positions_dict[stimulus_key]['y']
        threshold = threshold_dict[stimulus_key]
        amplitude_list = amplitude_list_dict[stimulus_key]
        duration_list = duration_list_dict[stimulus_key]
        start_frame = start_frame_dict[stimulus_key]
        current_frame = current_frame_dict[stimulus_key]
        mean_vel_pre = mean_vel_pre_dict[stimulus_key]
        std_vel_pre = std_vel_pre_dict[stimulus_key]
        amplitud_real = amplitud_real_dict[stimulus_key]
        y_max_velocity = y_max_velocity_dict[stimulus_key]
        trial_indices = trial_indices_per_stimulus[stimulus_key]
        stim_label = stimulus_key

        # Obtener los parámetros del estímulo
        form = form_dict[stimulus_key]
        duration_ms = duration_ms_dict[stimulus_key]
        frequency = frequency_dict[stimulus_key]

        ax1 = axes[0, idx]
        ax2 = axes[1, idx]
        ax3 = axes[2, idx]
        ax4 = axes[3, idx]
        ax5 = axes[4, idx]  # Para el gráfico de duración de movimiento

        # Obtener el mapa de colores para la articulación
        cmap_name = body_parts_colors.get(body_part, 'viridis')
        cmap = plt.get_cmap(cmap_name)

        # Generar colores basados en el número de ensayos
        num_trials = len(velocities_list)
        colors = cmap(np.linspace(0.2, 1, num_trials))  # Evitar colores muy claros

        # Obtener el color específico para la articulación
        body_part_color = body_parts_specific_colors.get(body_part, 'blue')

        # Graficar desplazamiento en ax1
        for idx_trial, (x, y) in enumerate(zip(positions_x_list, positions_y_list)):
            frames = np.arange(len(x))
            color = colors[idx_trial]
            displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)  # Distancia euclidiana
            ax1.plot(frames, displacement, color=color, linestyle='-', alpha=0.5)

        ax1.set_title(f'Desplazamiento - {stim_label}', fontsize=12)
        ax1.set_xlabel('Fotogramas')
        ax1.set_ylabel('Desplazamiento (px)')
        ax1.set_xlim(0, 400)
        ax1.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

        # Graficar velocidades en ax2
        for idx_trial, vel in enumerate(velocities_list):
            frames_vel = np.arange(len(vel))
            color = colors[idx_trial]
            ax2.plot(frames_vel, vel, color=color, alpha=0.5)

            # Resaltar segmentos por encima del umbral durante el estímulo
            within_stimulus = (frames_vel >= start_frame) & (frames_vel <= current_frame)
            above_threshold = (vel > threshold) & within_stimulus

            indices_above = frames_vel[above_threshold]

            if len(indices_above) > 0:
                # Encontrar segmentos continuos
                segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0]+1)
                for segment in segments:
                    ax2.plot(segment, vel[segment], color='red', linewidth=1, alpha=0.7)

        # Añadir líneas horizontales para el umbral y la media pre-estímulo
        ax2.axhline(threshold, color='k', linestyle='--', label=f'Umbral ({threshold:.2f})')
        ax2.axhline(mean_vel_pre, color='gray', linestyle=':', label=f'Media pre-estímulo ({mean_vel_pre:.2f})')

        # Añadir líneas de cuadrícula verticales cada 10 fotogramas (100 ms)
        grid_interval = 10
        grid_frames = np.arange(start_frame, current_frame, grid_interval)
        for gf in grid_frames:
            ax2.axvline(gf, color='lightgray', linestyle='--', linewidth=0.5)

        # Establecer límites y ticks fijos para el eje Y en ax2
        ax2.set_ylim(0, group_y_max_velocity)
        ax2.set_yticks(np.linspace(0, group_y_max_velocity, num=6))  # 5 intervalos

        # Obtener datos de movimiento para este estímulo
        movimiento_datos = [d for d in all_movement_data if d['body_part'] == body_part and d['Dia experimental'] == dia_experimental and f"{d['Forma del Pulso'].capitalize()}_{d['Duración (ms)'] if d['Duración (ms)'] else ''}ms" == stim_label and d['Amplitud (microA)'] in amplitud_real]
        if movimiento_datos:
            movimiento_info = movimiento_datos[0]
            movimiento_texto = f"Ensayos con movimiento: {movimiento_info['movement_trials']}/{movimiento_info['total_trials']}"
        else:
            movimiento_texto = ""

        ax2.set_title(f'Velocidades - {stim_label}', fontsize=12)
        ax2.set_xlabel('Fotogramas')
        ax2.set_ylabel('Velocidad (unidades/fotograma)')
        ax2.set_xlim(0, 400)
        ax2.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

        # Añadir conteo de ensayos en la leyenda
        ax2.legend(loc='upper right', fontsize=9, title=movimiento_texto)

        # Graficar aceleraciones en ax3
        for idx_trial, vel in enumerate(velocities_list):
            vel = np.array(vel)
            if len(vel) < 2:
                logging.warning(f'No se puede calcular la aceleración para este ensayo porque la velocidad tiene longitud {len(vel)}')
                aceleracion = np.zeros_like(vel)
            else:
                aceleracion = np.gradient(vel)
            frames_acc = np.arange(len(aceleracion))
            color = colors[idx_trial]
            if len(aceleracion) > 0:
                ax3.plot(frames_acc, aceleracion, color=color, alpha=0.5)

        ax3.set_title(f'Aceleraciones - {stim_label}', fontsize=12)
        ax3.set_xlabel('Fotogramas')
        ax3.set_ylabel('Aceleración (unidades/fotograma²)')
        ax3.set_xlim(0, 400)
        ax3.set_ylim(-group_y_max_velocity * 0.1, group_y_max_velocity * 0.1)  # Ajuste dinámico basado en y_max_velocity
        ax3.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

        # Graficar estímulo en ax4
        x_vals, y_vals = [start_frame], [0]
        current_frame_stimulus = start_frame
        for amp, dur in zip(amplitude_list, duration_list):
            next_frame = current_frame_stimulus + dur
            x_vals.extend([current_frame_stimulus, next_frame])
            y_vals.extend([amp / 1000, amp / 1000])  # Convertir a μA
            current_frame_stimulus = next_frame

        ax4.step(x_vals, y_vals, color='blue', where='post', linewidth=1,
                 label=f'Amplitud(es): {amplitud_real} μA')

        # Añadir líneas punteadas en los valores máximos y mínimos de amplitud
        fixed_max_amp = 160  # Valor fijo máximo
        fixed_min_amp = -160    # Valor fijo mínimo

        ax4.axhline(fixed_max_amp, color='darkblue', linestyle=':', linewidth=1)
        ax4.axhline(fixed_min_amp, color='darkblue', linestyle=':', linewidth=1)

        # Establecer ticks fijos en el eje Y
        ax4.set_yticks([fixed_min_amp, fixed_max_amp])

        # Cambiar el color de las etiquetas de los ticks en max_amp y min_amp
        yticks = ax4.get_yticks()
        yticklabels = ax4.get_yticklabels()
        for tick, label in zip(yticks, yticklabels):
            if tick == fixed_min_amp or tick == fixed_max_amp:
                label.set_color('darkblue')

        ax4.set_xlabel('Fotogramas')
        ax4.set_ylabel('Amplitud (μA)')
        ax4.set_xlim(0, 400)
        # Establecer límites fijos del eje Y

        ax4.set_ylim(-160,160)
 
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axvspan(start_frame, current_frame, color='blue', alpha=0.3)

        # Añadir texto de parámetros del estímulo
        estimulo_params_text = f"Forma: {form}\nDuración: {duration_ms} ms\nFrecuencia: {frequency} Hz"
        ax4.text(0.95, 0.95, estimulo_params_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Mostrar leyenda
        ax4.legend(loc='lower right', fontsize=9)

        # Crear gráfico de duración de movimiento en ax5
        movement_ranges = movement_ranges_dict.get(stimulus_key, [])
        if movement_ranges:
            # Graficar líneas negras para cada segmento de movimiento
            total_durations_per_trial = {}
            for mr in movement_ranges:
                ensayo = mr['Ensayo']
                inicio = mr['Inicio Movimiento (Frame)']
                fin = mr['Fin Movimiento (Frame)']
                duracion_mov = fin - inicio

                # Sumar duraciones por ensayo
                if ensayo not in total_durations_per_trial:
                    total_durations_per_trial[ensayo] = duracion_mov
                else:
                    total_durations_per_trial[ensayo] += duracion_mov

                ax5.hlines(y=ensayo, xmin=inicio, xmax=fin, color='black', linewidth=2)

                # Obtener las velocidades para este ensayo
                idx_trial = ensayo - 1  # Asumiendo que 'Ensayo' empieza desde 1
                vel = velocities_list[idx_trial]
                # Encontrar el punto de velocidad máxima dentro del segmento
                segment_velocities = vel[inicio:fin+1]
                if len(segment_velocities) > 0:
                    max_vel_index = np.argmax(segment_velocities)
                    max_vel_frame = inicio + max_vel_index
                    max_vel_value = segment_velocities[max_vel_index]
                    # Marcar el punto de velocidad máxima con color de la articulación y tamaño pequeño
                    ax5.plot(max_vel_frame, ensayo, marker='o', markersize=2, color=body_part_color)
                    # Añadir marca en el eje x
                    ax5.axvline(max_vel_frame, color=body_part_color, linestyle=':', linewidth=0.1)

            # Calcular duración media sumando segmentos por ensayo
            total_durations = list(total_durations_per_trial.values())
            mean_duration = np.mean(total_durations)

            # Mostrar la duración del estímulo encima del gráfico
            stimulus_duration_frames = current_frame - start_frame

            # Ajustar el eje y para mostrar números enteros y comenzar en 1
            ax5.set_ylim(group_max_y_lim + 1, 0)  # Invertir el eje Y para que empiece en 1
            ax5.set_yticks(np.arange(1, group_max_y_lim + 1, max(1, group_max_y_lim // 10)))  # Más ticks segmentados

            ax5.set_title(f'Duraciones de Movimiento por Ensayo', fontsize=12)
            ax5.set_xlabel('Fotogramas')
            ax5.set_ylabel('Ensayo')
            ax5.set_xlim(0, 400)

            # Dibujar una banda vertical para la duración del estímulo
            ax5.axvspan(start_frame, current_frame, color='blue', alpha=0.1, label='Duración del estímulo')

            # Anotar el inicio y fin del estímulo en el eje x
            ax5.text(start_frame, group_max_y_lim + 0.5, f'Inicio: {start_frame}', fontsize=8, rotation=90, verticalalignment='bottom')
            ax5.text(current_frame, group_max_y_lim + 0.5, f'Fin: {current_frame}', fontsize=8, rotation=90, verticalalignment='bottom')

            # Dibujar líneas verticales en inicio y fin del estímulo
            ax5.axvline(start_frame, color='blue', linestyle='--', linewidth=1)
            ax5.axvline(current_frame, color='blue', linestyle='--', linewidth=1)

            # Añadir líneas de cuadrícula verticales cada 10 fotogramas (100 ms)
            grid_interval = 10
            grid_frames = np.arange(start_frame, current_frame, grid_interval)
            for gf in grid_frames:
                ax5.axvline(gf, color='lightgray', linestyle='--', linewidth=0.5)

            ax5.axvline(mean_duration, color='red', linestyle='--', label=f'Duración Media: {mean_duration:.2f} frames')

            # Mover la leyenda a la esquina inferior derecha
            ax5.legend(loc='lower right', fontsize=9)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, 'No se detectaron movimientos que excedan el umbral.',
                     horizontalalignment='center', verticalalignment='center', fontsize=10)

    # Título principal
    main_title = f'{group_name} - {body_part} - Día {dia_experimental}'
    fig.suptitle(main_title, fontsize=18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar para el título superior

    # Guardar el gráfico en PDF
    #plt.savefig(output_pdf_path, format='pdf')
    #print(f'Gráfico PDF guardado en {output_pdf_path}')

    # Guardar el gráfico en PNG con alta resolución
    output_png_path = output_pdf_path.replace('.pdf', '.png')
    plt.savefig(output_png_path, format='png', dpi=300)
    print(f'Gráfico PNG guardado en {output_png_path}')

    plt.close()

def collect_velocity_threshold_data():
    total_trials = 0
    all_movement_data = []  # Para recolectar datos de movimiento a través de los ensayos
    thresholds_data = []  # Para almacenar los umbrales para cada articulación
    processed_combinations = set()  # Para verificar las combinaciones procesadas

    grouped_by_day = stimuli_info.groupby('Dia experimental')

    for dia_experimental, day_df in grouped_by_day:
        print(f'Procesando Día Experimental: {dia_experimental}')
        print(f'Número de ensayos en day_df: {len(day_df)}')

        # Normalizar 'Forma del Pulso' en day_df
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()

        for group in stimulus_groups:
            group_name = group['name']
            stimuli_in_group = group['stimuli']
            print(f'Procesando grupo: {group_name}')

            for part in body_parts:
                # Añadir registro de depuración para verificar las combinaciones
                logging.info(f'Procesando: Día {dia_experimental}, Grupo {group_name}, Articulación {part}')
                processed_combinations.add((dia_experimental, group_name, part))

                group_velocities_dict = {}
                group_positions_dict = {}
                threshold_dict = {}
                mean_vel_pre_dict = {}
                std_vel_pre_dict = {}
                amplitude_list_dict = {}
                duration_list_dict = {}
                start_frame_dict = {}
                current_frame_dict = {}
                amplitud_real_dict = {}
                y_max_velocity_dict = {}
                trial_indices_per_stimulus = {}
                movement_ranges_dict = {}

                # Nuevos diccionarios para almacenar los parámetros del estímulo
                form_dict = {}
                duration_ms_dict = {}
                frequency_dict = {}

                for stim in stimuli_in_group:
                    forma_pulso = stim['Forma del Pulso'].lower()
                    duracion_ms = stim.get('Duración (ms)', None)

                    if duracion_ms is not None:
                        stim_df = day_df[(day_df['Forma del Pulso'].str.lower() == forma_pulso) & (day_df['Duración (ms)'] == duracion_ms)]
                    else:
                        stim_df = day_df[(day_df['Forma del Pulso'].str.lower() == forma_pulso)]

                    if stim_df.empty:
                        continue  # No hay datos para este estímulo

                    # Velocidades pre-estímulo para todas las articulaciones
                    pre_stim_velocities = {bp: [] for bp in body_parts}

                    # Iterar sobre todos los ensayos para el cálculo del umbral
                    for index, row in stim_df.iterrows():
                        camara_lateral = row['Camara Lateral']

                        if pd.notna(camara_lateral):
                            matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                            if not matching_segment.empty:
                                matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                                # Extraer datos de velocidad pre-estímulo
                                for _, segment_row in matching_segment_sorted.iterrows():
                                    nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                                    csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                                    if csv_path:
                                        velocidades, posiciones = calcular_velocidades(csv_path)

                                        for bp in body_parts:
                                            vel = velocidades.get(bp, [])
                                            if len(vel) > 0:
                                                vel_pre_stim = vel[:100]  # Primeros 100 frames para pre-estímulo
                                                vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]  # Eliminar NaN
                                                pre_stim_velocities[bp].extend(vel_pre_stim)

                    # Calcular el umbral basado en las velocidades pre-estímulo
                    vel_list = pre_stim_velocities[part]

                    if len(vel_list) < 10:
                        logging.warning(f'Datos insuficientes para calcular el umbral para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms}')
                        continue

                    # Calcular la media y desviación estándar de las velocidades pre-estímulo
                    mean_vel_pre = np.nanmean(vel_list)
                    std_vel_pre = np.nanstd(vel_list)
                    threshold = mean_vel_pre + 2 * std_vel_pre  # Umbral es media + 2*desviación estándar

                    # Registrar los cálculos de umbral para verificación
                    logging.info(f'Umbral calculado para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms}ms: Media={mean_vel_pre:.4f}, Desviación Estándar={std_vel_pre:.4f}, Umbral={threshold:.4f}')

                    # Analizar las amplitudes disponibles para este estímulo
                    amplitudes = stim_df['Amplitud (microA)'].unique()
                    amplitude_movement_counts = {}

                    for amplitude in amplitudes:
                        # Filtrar ensayos a esta amplitud
                        amplitude_trials = stim_df[stim_df['Amplitud (microA)'] == amplitude]

                        movement_trials = 0
                        total_trials_part = 0
                        max_velocities = []

                        for index, row in amplitude_trials.iterrows():
                            camara_lateral = row['Camara Lateral']
                            duracion_ms = row['Duración (ms)']

                            if pd.notna(camara_lateral):
                                matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                                if not matching_segment.empty:
                                    matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                                    # Generar estímulo para obtener 'start_frame' y 'current_frame'
                                    compensar = False if duracion_ms == 1000 else True
                                    amplitude_list, duration_list = generar_estimulo_desde_parametros(
                                        row['Forma del Pulso'],
                                        amplitude * 1000,  # Convertir a μA
                                        duracion_ms * 1000 if duracion_ms else 1000000,  # Convertir a µs
                                        row['Frecuencia (Hz)'],
                                        200,  # Duración del pulso en µs
                                        compensar=compensar
                                    )

                                    # Definir start y current frame basado en la duración del estímulo
                                    start_frame = 100
                                    current_frame = int(start_frame + sum(duration_list))

                                    for _, segment_row in matching_segment_sorted.iterrows():
                                        nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                                        csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                                        if csv_path:
                                            velocidades, posiciones = calcular_velocidades(csv_path)

                                            if part in velocidades:
                                                vel = velocidades[part]
                                                if len(vel) > 0:
                                                    # Contar ensayos donde la velocidad excede el umbral durante el estímulo
                                                    vel_stimulus = vel[start_frame:current_frame]
                                                    if np.any(vel_stimulus > threshold):
                                                        movement_trials += 1

                                                    total_trials_part += 1

                                                    # Colectar máximos de velocidad
                                                    max_vel = np.max(vel)
                                                    max_velocities.append(max_vel)

                        # Calcular proporción de ensayos donde el movimiento excede el umbral
                        proportion_movement = movement_trials / total_trials_part if total_trials_part > 0 else 0

                        # Guardar los conteos de movimiento para esta amplitud
                        amplitude_movement_counts[amplitude] = {
                            'movement_trials': movement_trials,
                            'total_trials': total_trials_part,
                            'max_velocities': max_velocities,
                            'proportion_movement': proportion_movement
                        }

                        # Almacenar datos de movimiento para todas las amplitudes
                        all_movement_data.append({
                            'body_part': part,
                            'Dia experimental': dia_experimental,
                            'Forma del Pulso': forma_pulso,
                            'Duración (ms)': duracion_ms,
                            'Amplitud (microA)': amplitude,
                            'movement_trials': movement_trials,
                            'total_trials': total_trials_part,
                            'no_movement_trials': total_trials_part - movement_trials,
                            'proportion_movement': proportion_movement
                        })

                    # Ahora, seleccionar todas las amplitudes con mayor 'proportion_movement'
                    if amplitude_movement_counts:
                        # Encontrar el valor máximo de 'proportion_movement'
                        max_proportion = max([data['proportion_movement'] for data in amplitude_movement_counts.values()])

                        # Obtener todas las amplitudes que tienen la proporción máxima
                        selected_amplitudes = [amp for amp, data in amplitude_movement_counts.items() if data['proportion_movement'] == max_proportion]

                        # Combinar los ensayos de estas amplitudes
                        selected_amplitude_data_list = [amplitude_movement_counts[amp] for amp in selected_amplitudes]

                        # Unir los ensayos y datos de movimiento
                        movement_trials = sum(data['movement_trials'] for data in selected_amplitude_data_list)
                        total_trials_part = sum(data['total_trials'] for data in selected_amplitude_data_list)
                        max_velocities = sum([data['max_velocities'] for data in selected_amplitude_data_list], [])
                        proportion_movement = movement_trials / total_trials_part if total_trials_part > 0 else 0

                        # Para fines de graficado, combinamos los ensayos de las amplitudes seleccionadas
                        selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amplitudes)]

                        # Imprimir las amplitudes seleccionadas para verificar
                        print(f"Amplitudes seleccionadas para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms} ms: {selected_amplitudes} μA con proporción de movimiento: {proportion_movement:.2f}")
                    else:
                        continue  # No hay datos para este estímulo y articulación

                    # Calcular y_max_velocity como la media más la desviación estándar de los máximos
                    if max_velocities:
                        y_max_velocity = np.mean(max_velocities) + np.std(max_velocities)
                    else:
                        y_max_velocity = None

                    # Obtener la frecuencia para este estímulo
                    frequencies = selected_trials['Frecuencia (Hz)'].unique()
                    if len(frequencies) == 1:
                        frequency = frequencies[0]
                    else:
                        logging.warning(f'Múltiples frecuencias encontradas para el estímulo {forma_pulso} {duracion_ms} ms. Usando la primera.')
                        frequency = frequencies[0]

                    # Colectar velocidades y posiciones para esta articulación
                    group_velocities = []
                    group_positions = {'x': [], 'y': []}
                    group_trial_indices = []
                    trial_counter = 0
                    movement_ranges = []  # Para almacenar los rangos de movimiento

                    for index, row in selected_trials.iterrows():
                        camara_lateral = row['Camara Lateral']
                        duracion_ms = row['Duración (ms)']

                        if pd.notna(camara_lateral):
                            matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                            if not matching_segment.empty:
                                matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                                # Generar estímulo para obtener 'start_frame' y 'current_frame'
                                compensar = False if duracion_ms == 1000 else True
                                amplitude_list, duration_list = generar_estimulo_desde_parametros(
                                    row['Forma del Pulso'],
                                    row['Amplitud (microA)'] * 1000,  # Convertir a μA
                                    duracion_ms * 1000 if duracion_ms else 1000000,  # Convertir a µs
                                    row['Frecuencia (Hz)'],
                                    200,  # Duración del pulso en µs
                                    compensar=compensar
                                )

                                # Definir start y current frame basado en la duración del estímulo
                                start_frame = 100
                                current_frame = int(start_frame + sum(duration_list))

                                for _, segment_row in matching_segment_sorted.iterrows():
                                    nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                                    csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                                    if csv_path:
                                        velocidades, posiciones = calcular_velocidades(csv_path)

                                        if part in velocidades:
                                            vel = velocidades[part]
                                            pos = posiciones[part]
                                            if len(vel) > 0:
                                                group_velocities.append(vel)
                                                group_positions['x'].append(pos['x'])
                                                group_positions['y'].append(pos['y'])
                                                group_trial_indices.append(trial_counter)
                                                trial_counter += 1

                                                # Determinar múltiples segmentos de movimiento
                                                frames_vel = np.arange(len(vel))
                                                within_stimulus = (frames_vel >= start_frame) & (frames_vel <= current_frame)
                                                above_threshold = (vel > threshold) & within_stimulus

                                                indices_above = frames_vel[above_threshold]

                                                # Encontrar segmentos continuos donde la velocidad excede el umbral
                                                if len(indices_above) > 0:
                                                    segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)
                                                    for segment in segments:
                                                        movement_start = segment[0]
                                                        movement_end = segment[-1]
                                                        movement_ranges.append({
                                                            'Ensayo': trial_counter,
                                                            'Inicio Movimiento (Frame)': movement_start,
                                                            'Fin Movimiento (Frame)': movement_end
                                                        })

                    if len(group_velocities) == 0:
                        continue  # No hay datos para graficar

                    # Crear una clave única para este estímulo
                    if duracion_ms is not None:
                        stimulus_key = f"{forma_pulso.capitalize()}_{duracion_ms}ms"
                    else:
                        stimulus_key = f"{forma_pulso.capitalize()}"

                    group_velocities_dict[stimulus_key] = group_velocities
                    group_positions_dict[stimulus_key] = group_positions
                    threshold_dict[stimulus_key] = threshold
                    amplitude_list_dict[stimulus_key] = amplitude_list
                    duration_list_dict[stimulus_key] = duration_list
                    start_frame_dict[stimulus_key] = start_frame
                    current_frame_dict[stimulus_key] = current_frame
                    mean_vel_pre_dict[stimulus_key] = mean_vel_pre
                    std_vel_pre_dict[stimulus_key] = std_vel_pre
                    amplitud_real_dict[stimulus_key] = selected_amplitudes  # Lista de amplitudes seleccionadas
                    y_max_velocity_dict[stimulus_key] = y_max_velocity

                    # Almacenar los índices de ensayo por estímulo
                    trial_indices_per_stimulus[stimulus_key] = group_trial_indices

                    # Almacenar los rangos de movimiento
                    movement_ranges_dict[stimulus_key] = movement_ranges

                    # Almacenar los parámetros del estímulo
                    form_dict[stimulus_key] = forma_pulso.capitalize()
                    duration_ms_dict[stimulus_key] = duracion_ms
                    frequency_dict[stimulus_key] = frequency

                    # Almacenar datos de umbrales
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

                if len(group_velocities_dict) == 0:
                    continue  # No hay datos para graficar para este grupo

                # Generar el gráfico para este grupo y articulación
                stimuli_in_group_names = '_'.join([f"{s['Forma del Pulso'].capitalize()}{s['Duración (ms)'] if s['Duración (ms)'] else ''}" for s in stimuli_in_group])
                output_pdf_path = os.path.join(
                    output_comparisons_dir,
                    f'{sanitize_filename(group_name)}_{sanitize_filename(stimuli_in_group_names)}_{sanitize_filename(part)}_dia_{sanitize_filename(str(dia_experimental))}.pdf'
                )

                plot_group_graphs(
                    group_velocities_dict, group_positions_dict,
                    threshold_dict, amplitude_list_dict, duration_list_dict,
                    group_name, part, dia_experimental,
                    start_frame_dict, current_frame_dict,
                    mean_vel_pre_dict, std_vel_pre_dict, amplitud_real_dict,
                    y_max_velocity_dict, output_pdf_path,
                    trial_indices_per_stimulus,
                    all_movement_data,
                    movement_ranges_dict,
                    form_dict,
                    duration_ms_dict,
                    frequency_dict  # Añadimos los nuevos diccionarios aquí
                )

    # Crear un DataFrame a partir de los datos de movimiento recolectados
    counts_df = pd.DataFrame(all_movement_data)
    counts_df.to_csv(os.path.join(output_comparisons_dir, 'movement_counts_summary.csv'), index=False)

    # Crear y guardar el DataFrame de umbrales
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_df.to_csv(os.path.join(output_comparisons_dir, 'thresholds_summary.csv'), index=False)
    print(f"Datos de umbrales guardados en {os.path.join(output_comparisons_dir, 'thresholds_summary.csv')}")

    # Verificar las combinaciones procesadas
    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        print(f"Día: {combo[0]}, Grupo: {combo[1]}, Articulación: {combo[2]}")

    return counts_df

# Funciones analyze_best_bodyparts_and_stimuli y plot_heatmap (sin cambios)
def analyze_best_bodyparts_and_stimuli(counts_df):
    # Crear una columna para identificar el estímulo
    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

    # Ordenar por proporción de movimiento para identificar las mejores articulaciones y estímulos
    sorted_df = counts_df.sort_values(by='proportion_movement', ascending=False)

    # Mostrar top 5 articulaciones con mayor proporción de movimiento
    top_bodyparts = sorted_df.groupby('body_part')['proportion_movement'].mean().sort_values(ascending=False)
    print("Top articulaciones con mayor proporción de movimiento:")
    print(top_bodyparts.head(5))

    # Mostrar top 5 estímulos con mayor proporción de movimiento
    top_stimuli = sorted_df.groupby('Estímulo')['proportion_movement'].mean().sort_values(ascending=False)
    print("\nTop estímulos con mayor proporción de movimiento:")
    print(top_stimuli.head(5))

    # Guardar resultados en archivos CSV
    top_bodyparts.to_csv(os.path.join(output_comparisons_dir, 'top_bodyparts.csv'))
    top_stimuli.to_csv(os.path.join(output_comparisons_dir, 'top_stimuli.csv'))

def plot_heatmap(counts_df):
    # Crear una columna para identificar el estímulo
    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

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

    plt.figure(figsize=(20, 15))
    sns.heatmap(pivot_prop, annot=annot_matrix, fmt='', cmap='viridis')
    plt.title('Proporción de Movimiento Detectado por Articulación, Día y Estímulo')
    plt.xlabel('Día Experimental y Estímulo')
    plt.ylabel('Articulación')
    plt.tight_layout()
    plt.savefig(os.path.join(output_comparisons_dir, 'heatmap_bodypart_day_stimulus.png'))
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
