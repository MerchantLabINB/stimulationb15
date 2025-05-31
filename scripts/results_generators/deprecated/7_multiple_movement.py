# Importaciones y configuración inicial
import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from math import sqrt
import matplotlib
matplotlib.use('Agg')  # Para entornos sin interfaz gráfica

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import seaborn as sns

from scipy.signal import savgol_filter, find_peaks
import re
import shutil
import glob  # Importar el módulo glob

# Configuración del logging
refactored_log_file_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\filtered_processing_log.txt'
logging.basicConfig(
    filename=refactored_log_file_path,
    level=logging.DEBUG,  # Cambiado a DEBUG para más detalles
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # 'w' para sobrescribir el archivo cada vez
)

# Añadir un StreamHandler para ver los logs en la consola
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)  # Cambiado a DEBUG para más detalles en consola
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Confirmar inicio del script
print("Inicio del script: 5_velocity_thresholds_comparaciones_refined.py")
logging.info("Inicio del script: 5_velocity_thresholds_comparaciones_refined.py")

# Añadir la ruta a Stimulation.py
stimulation_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\scripts\GUI_pattern_generator'
if not os.path.exists(stimulation_path):
    logging.error(f"La ruta a Stimulation.py no existe: {stimulation_path}")
    print(f"La ruta a Stimulation.py no existe: {stimulation_path}")
    sys.exit(f"La ruta a Stimulation.py no existe: {stimulation_path}")
sys.path.append(stimulation_path)
logging.info(f"Ruta añadida al PATH: {stimulation_path}")
print(f"Ruta añadida al PATH: {stimulation_path}")

# Importar la función estimulo de Stimulation.py
try:
    from Stimulation import estimulo
    logging.info("Función 'estimulo' importada correctamente.")
    print("Función 'estimulo' importada correctamente.")
except ImportError as e:
    logging.error(f'Error al importar la función estimulo: {e}')
    print(f'Error al importar la función estimulo: {e}')
    sys.exit(f'Error al importar la función estimulo: {e}')

# Directorios
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
segmented_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\informacion_archivos_segmentados.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\xv_lat-Br-2024-10-02\videos'
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\multiple_filtered_variability_plots'

# Asegurarse de que el directorio de salida existe
if not os.path.exists(output_comparisons_dir):
    os.makedirs(output_comparisons_dir)
    logging.info(f"Directorio de salida creado: {output_comparisons_dir}")
    print(f"Directorio de salida creado: {output_comparisons_dir}")
else:
    logging.info(f"Directorio de salida ya existe: {output_comparisons_dir}")
    print(f"Directorio de salida ya existe: {output_comparisons_dir}")

# Función para verificar la existencia de archivos
def verificar_archivo(path, nombre_archivo):
    if not os.path.exists(path):
        logging.error(f"Archivo no encontrado: {path}")
        print(f"Archivo no encontrado: {path}")
        sys.exit(f"Archivo no encontrado: {path}")
    else:
        logging.info(f"Archivo encontrado: {path}")
        print(f"Archivo encontrado: {path}")

# Verificar la existencia de los archivos CSV
verificar_archivo(stimuli_info_path, 'Stimuli_information.csv')
verificar_archivo(segmented_info_path, 'informacion_archivos_segmentados.csv')

# Cargar archivos CSV
print(f"Cargando Stimuli_information desde: {stimuli_info_path}")
logging.info(f"Cargando Stimuli_information desde: {stimuli_info_path}")
try:
    stimuli_info = pd.read_csv(stimuli_info_path)
    logging.info(f"'Stimuli_information.csv' cargado con {len(stimuli_info)} filas.")
    print(f"'Stimuli_information.csv' cargado con {len(stimuli_info)} filas.")
except Exception as e:
    logging.error(f'Error al cargar Stimuli_information.csv: {e}')
    print(f'Error al cargar Stimuli_information.csv: {e}')
    sys.exit(f'Error al cargar Stimuli_information.csv: {e}')

print(f"Cargando informacion_archivos_segmentados desde: {segmented_info_path}")
logging.info(f"Cargando informacion_archivos_segmentados desde: {segmented_info_path}")
try:
    segmented_info = pd.read_csv(segmented_info_path)
    logging.info(f"'informacion_archivos_segmentados.csv' cargado con {len(segmented_info)} filas.")
    print(f"'informacion_archivos_segmentados.csv' cargado con {len(segmented_info)} filas.")
except Exception as e:
    logging.error(f'Error al cargar informacion_archivos_segmentados.csv: {e}')
    print(f'Error al cargar informacion_archivos_segmentados.csv: {e}')
    sys.exit(f'Error al cargar informacion_archivos_segmentados.csv: {e}')

# Filtrar entradas donde 'Descartar' es 'No'
print("Filtrando entradas donde 'Descartar' es 'No'")
logging.info("Filtrando entradas donde 'Descartar' es 'No'")
if 'Descartar' not in stimuli_info.columns:
    logging.error("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
    print("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
    sys.exit("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']
logging.info(f"'Stimuli_information.csv' después del filtrado tiene {len(stimuli_info)} filas.")
print(f"'Stimuli_information.csv' después del filtrado tiene {len(stimuli_info)} filas.")

# Normalizar 'Forma del Pulso' a minúsculas para evitar problemas de coincidencia
print("Normalizando 'Forma del Pulso' a minúsculas")
logging.info("Normalizando 'Forma del Pulso' a minúsculas")
if 'Forma del Pulso' not in stimuli_info.columns:
    logging.error("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
    print("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
    sys.exit("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
stimuli_info['Forma del Pulso'] = stimuli_info['Forma del Pulso'].str.lower()
logging.info("'Forma del Pulso' normalizada a minúsculas.")
print("'Forma del Pulso' normalizada a minúsculas.")

# Verificar si stimuli_info no está vacío
if stimuli_info.empty:
    logging.error("El DataFrame stimuli_info está vacío después de filtrar por 'Descartar' == 'No'. Verifica el archivo CSV.")
    print("El DataFrame stimuli_info está vacío después de filtrar por 'Descartar' == 'No'. Verifica el archivo CSV.")
    sys.exit("El DataFrame stimuli_info está vacío. No hay datos para procesar.")
logging.info("El DataFrame stimuli_info no está vacío después del filtrado.")
print("El DataFrame stimuli_info no está vacío después del filtrado.")

# Diccionario de colores específicos para cada articulación (actualizado)
body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muneca': 'red',  # Reemplazar 'ñ' por 'n'
    'Nudillo Central': 'purple',
    'DedoMedio': 'pink',
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}

body_parts = list(body_parts_specific_colors.keys())
def minimum_jerk_velocity(t, *params):
    n_submovements = len(params) // 3
    v_total = np.zeros_like(t)
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        # Ensure T > 0
        if T <= 0:
            continue
        # Compute the time within the submovement
        tau = (t - t0) / T
        # Only consider times within [0, T]
        valid_idx = (tau >= 0) & (tau <= 1)
        v = np.zeros_like(t)
        v[valid_idx] = A * 30 * (tau[valid_idx]**2) * (1 - tau[valid_idx])**2
        v_total += v
    return v_total

# Define the sum of submovements model
def sum_of_minimum_jerk(t, *params):
    n_submovements = len(params) // 3
    v_total = np.zeros_like(t)
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        v_total += minimum_jerk_velocity(t, A, t0, T)
    return v_total

def fit_velocity_profile(t, observed_velocity, n_submovements):
    # Check if t is valid
    if len(t) <= 1:
        logging.warning("Time array t is too short to fit the model.")
        return None

    # Ensure t is sorted in ascending order
    if not np.all(np.diff(t) >= 0):
        logging.warning("Time array t is not sorted in ascending order. Sorting t.")
        sorted_indices = np.argsort(t)
        t = t[sorted_indices]
        observed_velocity = observed_velocity[sorted_indices]

    # Calculate total time
    total_time = t[-1] - t[0]
    logging.debug(f"Trial: t[0]={t[0]}, t[-1]={t[-1]}, total_time={total_time}")

    if total_time <= 0:
        logging.warning("Total time duration is not positive. Skipping fitting for this trial.")
        return None

    # Initial guess for parameters
    params_init = []
    for i in range(n_submovements):
        A_init = np.max(observed_velocity) / n_submovements
        t0_init = t[0] + i * total_time / n_submovements
        T_init = total_time / n_submovements
        params_init.extend([A_init, t0_init, T_init])

    # Bounds for parameters
    lower_bounds = []
    upper_bounds = []
    for i in range(n_submovements):
        min_T = 0.01  # Minimum duration of 10ms
        lower_bounds.extend([0, t[0], min_T])
        upper_bounds.extend([np.inf, t[-1], total_time])

    # Ensure initial guesses are within bounds
    params_init = np.maximum(params_init, lower_bounds)
    params_init = np.minimum(params_init, upper_bounds)

    # Fit the model
    try:
        result = least_squares(
            lambda params: sum_of_minimum_jerk(t, *params) - observed_velocity,
            x0=params_init,
            bounds=(lower_bounds, upper_bounds)
        )
    except ValueError as e:
        logging.error(f"Least squares fitting failed for trial: {e}")
        return None
    return result


def sanitize_filename(filename):
    """
    Reemplaza los caracteres inválidos por guiones bajos.
    """
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # logging.debug(f"Sanitizing filename: Original='{filename}', Sanitized='{sanitized}'")
    return sanitized

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
            logging.debug(f'Searching for CSV files with pattern: {search_pattern}')
            matching_files = glob.glob(search_pattern)
            if matching_files:
                csv_path = matching_files[0]
                logging.debug(f'Archivo CSV filtrado encontrado: {csv_path}')
                return csv_path
            else:
                logging.warning(f'Archivo CSV filtrado no encontrado para la cámara: {camara_lateral}, segmento: {nombre_segmento}')
                return None
        else:
            logging.warning(f'No se pudieron extraer los dígitos del nombre del segmento: {nombre_segmento}')
            return None
    except Exception as e:
        logging.error(f'Error al acceder a los archivos CSV: {e}')
        return None

def suavizar_datos(data, window_length=10):
    if len(data) < window_length:
        logging.warning(f"Datos demasiado cortos para suavizar. Longitud={len(data)}, window_length={window_length}")
        return data

    # Creamos una ventana de unos y luego la dividimos para hacer el promedio.
    window = np.ones(window_length) / window_length

    # Convolve calcula la suma ponderada en cada posición.
    # mode='valid' devuelve solo las posiciones donde la ventana entra completa.
    filtered = np.convolve(data, window, mode='valid')

    # Si queremos mantener el mismo largo que la señal original, podemos rellenar los bordes.
    # Aquí, por ejemplo, rellenamos el inicio y el final con el valor más cercano:
    pad_size = (window_length - 1) // 2
    # Relleno inicial y final para mantener longitud
    filtered = np.concatenate((
        np.full(pad_size, filtered[0]), 
        filtered, 
        np.full(pad_size, filtered[-1])
    ))

    return filtered


# Función para calcular velocidades y posiciones para cada articulación con suavizado
def calcular_velocidades(csv_path):
    logging.debug(f"Calculando velocidades para CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        logging.debug(f"Archivo CSV cargado: {csv_path}")
        
        # Flatten the MultiIndex columns into single-level column names
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        logging.debug("Columnas aplanadas a single-level.")
        
        # Remove the 'scorer_bodyparts_coords' column if it exists
        if 'scorer_bodyparts_coords' in df.columns:
            df = df.drop(columns=['scorer_bodyparts_coords'])
            logging.debug("Columna 'scorer_bodyparts_coords' eliminada.")
        
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
                logging.warning(f"Columns for {part_original} are incompletas en {csv_path}.")
                print(f"Columns for {part_original} are incomplete.")
                continue

            logging.debug(f"Using columns for {part_original}: x_col={x_col}, y_col={y_col}, likelihood_col={likelihood_col}")
            print(f"Using columns for {part_original}: x_col={x_col}, y_col={y_col}, likelihood_col={likelihood_col}")

            # Filter rows based on likelihood
            df_filtered = df[df[likelihood_col] > 0.1]
            valid_frames = len(df_filtered)
            logging.info(f'{part_original} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')
            print(f'{part_original} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')

            if df_filtered.empty:
                logging.warning(f'No hay datos suficientes para {part_original} en {csv_path} después de filtrar por likelihood.')
                print(f'No hay datos suficientes para {part_original} en {csv_path} después de filtrar por likelihood.')
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
            velocidad_part = suavizar_datos(velocidad_part, window_length=10)
            logging.debug(f"Velocidades suavizadas para {part_original}.")
            print(f"Velocidades suavizadas para {part_original}.")

            # Store velocities and positions
            velocidades[part_original] = velocidad_part
            posiciones[part_original] = {'x': x, 'y': y}

        logging.info(f"Finalizado cálculo de velocidades para {csv_path}.")
        print(f"Finalizado cálculo de velocidades para {csv_path}.")
        return velocidades, posiciones
    except Exception as e:
        logging.error(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        print(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        return {}, {}

# Convertir tiempo de microsegundos a frames
def us_to_frames(duracion_us):
    return duracion_us / 10000  # 1 frame = 10,000 µs

# Función para generar el estímulo desde parámetros usando la lógica de Stimulation.py
def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    logging.debug(f"Generando estímulo con parámetros: forma={forma}, amplitud={amplitud}, duracion={duracion}, frecuencia={frecuencia}, duracion_pulso={duracion_pulso}, compensar={compensar}")
    try:
        forma = forma.strip().lower()  # Asegurar minúsculas

        # Verificar parámetros válidos
        if duracion <= 0 or frecuencia <= 0 or duracion_pulso <= 0:
            logging.error(f"Parámetros inválidos: duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}")
            print(f"Parámetros inválidos: duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}")
            return [], []

        # Generar estímulo usando la función estimulo
        lista_amplitud, lista_tiempo = estimulo(
            forma=forma, amplitud=amplitud, duracion=duracion,
            frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
        )
        logging.debug("Estímulo generado con éxito.")
        print("Estímulo generado con éxito.")

        # Asegurar generación correcta del estímulo
        if not lista_amplitud or not lista_tiempo:
            logging.error(f"Estímulo inválido con parámetros: forma={forma}, amplitud={amplitud}, duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}, compensar={compensar}")
            print(f"Estímulo inválido con parámetros: forma={forma}, amplitud={amplitud}, duración={duracion}, frecuencia={frecuencia}, duración_pulso={duracion_pulso}, compensar={compensar}")
            return [], []

        # Convertir todos los tiempos del estímulo (en µs) a frames
        lista_tiempo = [us_to_frames(tiempo) for tiempo in lista_tiempo]

        return lista_amplitud, lista_tiempo
    except Exception as e:
        logging.error(f'Error al generar estímulo: {e}')
        print(f'Error al generar estímulo: {e}')
        return [], []
def create_color_palette(df):
    # Obtener las formas de pulso únicas
    formas_unicas = df['Forma del Pulso'].unique()
    
    # Asignar un color base a cada forma de pulso
    base_colors = sns.color_palette('tab10', n_colors=len(formas_unicas))
    forma_color_dict = dict(zip(formas_unicas, base_colors))
    
    # Crear un diccionario para almacenar los colores finales
    stim_color_dict = {}
    
    for forma in formas_unicas:
        # Filtrar las duraciones para esta forma
        duraciones = sorted(df[df['Forma del Pulso'] == forma]['Duración (ms)'].unique())
        num_duraciones = len(duraciones)
        
        # Generar variaciones de color utilizando la paleta "light" de seaborn
        shades = sns.light_palette(forma_color_dict[forma], n_colors=num_duraciones, reverse=True, input='rgb')
        
        for i, duracion in enumerate(duraciones):
            stim_key = f"{forma.capitalize()}_{duracion}ms"
            stim_color_dict[stim_key] = shades[i]
    
    return stim_color_dict

def plot_all_stimuli_graphs(all_stimuli_data, group_name, body_part, dia_experimental, output_png_path):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    logging.info(f"Generando gráfico consolidado para: Articulación={body_part}, Día={dia_experimental}")
    print(f"Generando gráfico consolidado para: Articulación={body_part}, Día={dia_experimental}")
    
    num_stimuli = len(all_stimuli_data)
    
    if num_stimuli == 0:
        logging.warning(f"No hay estímulos para graficar para {body_part} en el día {dia_experimental}.")
        print(f"No hay estímulos para graficar para {body_part} en el día {dia_experimental}.")
        return
    
    # Calcular los límites dinámicos para mantenerlos iguales en todos los subgráficos
    all_displacements = []
    all_velocities = []
    max_ensayo = 0
    max_time = 0  # Para ajustar el eje X del estímulo
    
    for data in all_stimuli_data.values():
        # Desplazamiento
        for x, y in zip(data['positions']['x'], data['positions']['y']):
            displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)
            all_displacements.extend(displacement)
        
        # Velocidades
        for vel in data['velocities']:
            all_velocities.extend(vel)
        
        # Ensayos
        if data.get('movement_ranges'):
            for mr in data['movement_ranges']:
                if mr['Ensayo'] > max_ensayo:
                    max_ensayo = mr['Ensayo']
        
        # Actualizar el tiempo máximo para el estímulo
        stim_end_time = data['current_frame'] / 100
        if stim_end_time > max_time:
            max_time = stim_end_time

    # Asegurar que max_time es al menos un valor mínimo para evitar escalamiento excesivo
    min_time_limit = 4  # Puedes ajustar este valor según tus necesidades
    max_time = max(max_time, min_time_limit)
    
    # Definir límites para los gráficos
    # Aplicar detección de outliers para el desplazamiento
    if all_displacements:
        disp_array = np.array(all_displacements)
        mean_disp = np.mean(disp_array)
        std_disp = np.std(disp_array)
        # Filtrar outliers que están más allá de 3 desviaciones estándar
        disp_filtered = disp_array[(disp_array >= mean_disp - 3 * std_disp) & (disp_array <= mean_disp + 3 * std_disp)]
        disp_min = np.min(disp_filtered) - 10
        disp_max = np.max(disp_filtered) + 10
        # Establecer un valor mínimo para disp_max
        disp_min_limit = 100  # Valor mínimo para disp_max
        disp_max = max(disp_max, disp_min_limit)
        # Asegurar que disp_max es mayor que disp_min
        if disp_max <= disp_min:
            disp_max = disp_min + 10
    else:
        disp_min, disp_max = (0, 500)
    
    # Aplicar detección de outliers para las velocidades
    if all_velocities:
        vel_array = np.array(all_velocities)
        mean_vel = np.mean(vel_array)
        std_vel = np.std(vel_array)
        # Filtrar outliers que están más allá de 3 desviaciones estándar
        vel_filtered = vel_array[(vel_array >= mean_vel - 3 * std_vel) & (vel_array <= mean_vel + 3 * std_vel)]
        vel_min = np.min(vel_filtered) - 5
        vel_max = np.max(vel_filtered) + 5
        # Establecer un valor mínimo para vel_max
        vel_min_limit = 10  # Valor mínimo para vel_max
        vel_max = max(vel_max, vel_min_limit)
        vel_max = max(vel_max, vel_min + 1)  # Asegurar que vel_max es mayor que vel_min
    else:
        vel_min, vel_max = (0, 50)
    
    mov_min, mov_max = (0, max_ensayo + 2)
    amp_min, amp_max = -160, 160  # Límites fijos para el gráfico de estimulación
    
    # Calcular disposición de subplots
    cols = min(num_stimuli, 3)  # Máximo 3 columnas
    rows = int(np.ceil(num_stimuli / cols))
    
    min_fig_height = 10  # Ajusta este valor según sea necesario
    fig_height = max(rows * 5, min_fig_height)  # Ajustar la altura según el número de filas
    fig_width = cols * 15  # Ajustar el ancho según el número de columnas

    fig, axes = plt.subplots(5, num_stimuli, figsize=(fig_width, fig_height),
                             gridspec_kw={'height_ratios': [4, 4, 4, 4, 2]})
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    # Asegurar que axes es un arreglo 2D
    if num_stimuli == 1:
        axes = np.array([axes]).reshape(5, 1)
    
    # Colores para los estímulos
    cmap = plt.get_cmap('tab10')
    stimulus_colors = {stimulus_key: body_parts_specific_colors.get(body_part, cmap(idx % 10)) for idx, stimulus_key in enumerate(all_stimuli_data.keys())}
    
    for idx, (stimulus_key, data) in enumerate(all_stimuli_data.items()):
        logging.debug(f"Procesando estímulo: {stimulus_key}")
        print(f"Procesando estímulo: {stimulus_key}")
        
        # Convertir frames a segundos
        start_time = data['start_frame'] / 100
        current_time = data['current_frame'] / 100

        # -----------------------
        # 1. Graficar Desplazamiento
        # -----------------------
        ax_disp = axes[0, idx] if num_stimuli > 1 else axes[0]
        for x, y in zip(data['positions']['x'], data['positions']['y']):
            frames = np.arange(len(x))
            seconds = frames / 100
            displacement = np.sqrt((x - x[0])**2 + (y - y[0])**2)
            ax_disp.plot(seconds, displacement, color=stimulus_colors[stimulus_key], linestyle='-', alpha=0.5)
        
        ax_disp.set_title(f'Desplazamiento - {stimulus_key}', fontsize=12)
        ax_disp.set_xlabel('Tiempo (s)')
        ax_disp.set_ylabel('Desplazamiento (px)')
        ax_disp.set_xlim(0, max_time)
        ax_disp.set_ylim(disp_min, disp_max)
        ax_disp.axvspan(start_time, current_time, color='blue', alpha=0.1)
        
        # Configurar ticks mayores y menores para el eje X
        ax_disp.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_disp.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_disp.tick_params(axis='x', which='minor', length=4, color='grey')

        # -----------------------
        # 2. Graficar Velocidades
        # -----------------------
        ax_vel = axes[1, idx] if num_stimuli > 1 else axes[1]

        movement_trials_passed = 0  # Contador de ensayos que superaron el umbral durante el estímulo
        total_trials = len(data['velocities'])

        for trial_idx, vel in enumerate(data['velocities']):
            if len(vel) <= 1 or np.isnan(vel).all():
                logging.warning(f"Trial {trial_idx} has insufficient or invalid velocity data.")
                continue
            frames_vel = np.arange(len(vel))
            t = frames_vel / 100  # Convertir frames a tiempo en segundos
            observed_velocity = vel
            ax_vel.plot(t, vel, color=stimulus_colors[stimulus_key], alpha=0.5)

            # Obtener movimientos correspondientes a este ensayo
            movements_in_trial = [mr for mr in data['movement_ranges'] if mr['Ensayo'] == trial_idx + 1]

            # Obtener movimientos que inician durante el estímulo en este ensayo
            movements_in_trial_durante_estimulo = [mr for mr in movements_in_trial if mr['Periodo'] == 'Durante Estímulo']

            if movements_in_trial_durante_estimulo:
                movement_trials_passed += 1  # Contar el ensayo si hay al menos un movimiento que inicia durante el estímulo

            for movement in movements_in_trial:
                movement_start = movement['Inicio Movimiento (Frame)']
                movement_end = movement['Fin Movimiento (Frame)']
                periodo = movement['Periodo']

                # Obtener los índices de frames del movimiento
                segment_indices = np.arange(movement_start, movement_end + 1)
                # Asegurarse de que los índices estén dentro del rango de vel
                segment_indices = segment_indices[(segment_indices >= 0) & (segment_indices < len(vel))]

                # Color según el periodo
                if periodo == 'Pre-Estímulo':
                    color_mov = 'orange'
                elif periodo == 'Durante Estímulo':
                    color_mov = 'red'
                elif periodo == 'Post-Estímulo':
                    color_mov = 'gray'
                else:
                    color_mov = 'blue'  # Por defecto

                ax_vel.plot(segment_indices / 100, vel[segment_indices], color=color_mov, linewidth=1, alpha=0.7)

        # Añadir líneas horizontales para el umbral y la media pre-estímulo
        ax_vel.axhline(data['threshold'], color='k', linestyle='--', label=f'Umbral ({data["threshold"]:.2f})')
        ax_vel.axhline(data['mean_vel_pre'], color='lightcoral', linestyle='-', linewidth=1,
                       label=f'Media pre-estímulo ({data["mean_vel_pre"]:.2f})')
        
        # Añadir líneas de cuadrícula verticales cada 1 segundo
        grid_interval = 1
        grid_seconds = np.arange(0, max_time + 1, grid_interval)
        for gs in grid_seconds:
            ax_vel.axvline(gs, color='lightgray', linestyle='--', linewidth=0.5)
        
        # Establecer límites dinámicos
        ax_vel.set_ylim(vel_min, vel_max)
        
        # Añadir título y leyenda con información de ensayos que pasaron el umbral
        ax_vel.set_title(f'Velocidades - {stimulus_key}', fontsize=12)
        ax_vel.set_xlabel('Tiempo (s)')
        ax_vel.set_ylabel('Velocidad (pixeles/s)')
        ax_vel.set_xlim(0, max_time)
        ax_vel.axvspan(start_time, current_time, color='blue', alpha=0.1)
        
        # Configurar ticks mayores y menores para el eje X
        ax_vel.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_vel.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_vel.tick_params(axis='x', which='minor', length=4, color='grey')
        
        # Crear elementos de leyenda
        legend_elements = [
            Line2D([0], [0], color='lightcoral', lw=2.5, label=f'Media pre-estímulo ({data["mean_vel_pre"]:.2f})'),
            Line2D([0], [0], color='k', linestyle='--', label=f'Umbral ({data["threshold"]:.2f})'),
            mpatches.Patch(facecolor='blue', alpha=0.1, label='Duración del estímulo')
        ]
        legend_text = f'Superaron: {movement_trials_passed}/{total_trials}'
        ax_vel.legend(handles=legend_elements, title=legend_text, loc='upper right', fontsize=9)

        # -----------------------
        # 3. Graficar Duraciones de Movimiento
        # -----------------------
        ax_mov = axes[2, idx] if num_stimuli > 1 else axes[2]
        if data.get('movement_ranges'):
            for mr in data['movement_ranges']:
                ensayo = mr['Ensayo']
                inicio = mr['Inicio Movimiento (Frame)'] / 100
                fin = mr['Fin Movimiento (Frame)'] / 100

                # Obtener el periodo del movimiento
                periodo = mr['Periodo']

                # Determinar el color según el periodo
                if periodo == 'Pre-Estímulo':
                    color_mov = 'orange'
                elif periodo == 'Durante Estímulo':
                    color_mov = 'red'
                elif periodo == 'Post-Estímulo':
                    color_mov = 'gray'
                else:
                    color_mov = 'blue'  # Por defecto

                ax_mov.hlines(y=ensayo, xmin=inicio, xmax=fin, color=color_mov, linewidth=2)
                
                # Obtener las velocidades para este ensayo
                if ensayo - 1 < len(data['velocities']):
                    vel = data['velocities'][ensayo - 1]
                    segment_indices = (np.arange(len(vel)) / 100 >= inicio) & (np.arange(len(vel)) / 100 <= fin)
                    segment_velocities = vel[segment_indices]
                    if len(segment_velocities) > 0:
                        max_vel_index = np.argmax(segment_velocities)
                        max_vel_time = inicio + (max_vel_index / 100)
                        ax_mov.plot(max_vel_time, ensayo, marker='o', markersize=3, color='black')
                        # ax_mov.axvline(max_vel_time, color='black', linestyle=':', linewidth=0.5)
            
            
            # Calcular duración media
            total_durations = [mr['Fin Movimiento (Frame)'] - mr['Inicio Movimiento (Frame)'] for mr in data['movement_ranges']]
            mean_duration_frames = np.mean(total_durations) if total_durations else 0
            mean_duration = mean_duration_frames / 100  # Convertir a segundos
            
            # Establecer límites
            ax_mov.set_ylim(mov_min, mov_max)
            ax_mov.set_xlim(0, max_time)  # Usar max_time fijo para evitar escalamiento
            
            ax_mov.set_title(f'Duraciones de Movimiento por Ensayo - {stimulus_key}', fontsize=12)
            ax_mov.set_xlabel('Tiempo (s)')
            ax_mov.set_ylabel('Ensayo')
            
            # Dibujar banda vertical para la duración del estímulo
            ax_mov.axvspan(start_time, current_time, color='blue', alpha=0.1, label=f'Duración del estímulo: {current_time - start_time:.2f}s')
            
            # Añadir líneas de cuadrícula verticales
            for gs in grid_seconds:
                ax_mov.axvline(gs, color='lightgray', linestyle='--', linewidth=0.5)
            
            # Configurar ticks mayores y menores para el eje X
            ax_mov.xaxis.set_major_locator(MultipleLocator(0.5))
            ax_mov.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax_mov.tick_params(axis='x', which='minor', length=4, color='grey')
            
            # Añadir leyenda
            legend_elements_mov = [
                Line2D([0], [0], color='orange', lw=2, label='Mov. pre-estímulo'),
                Line2D([0], [0], color='red', lw=2, label='Mov. durante estímulo'),
                Line2D([0], [0], color='gray', lw=2, label='Mov. post-estímulo'),
                mpatches.Patch(facecolor='blue', alpha=0.1, label='Duración del estímulo')
            ]
            ax_mov.legend(handles=legend_elements_mov, loc='lower right', fontsize=9)
        else:
            ax_mov.axis('off')
            ax_mov.text(0.5, 0.5, 'No se detectaron movimientos que excedan el umbral.',
                       horizontalalignment='center', verticalalignment='center', fontsize=10)

        # -----------------------
        # 4. Graficar Submovimientos
        # -----------------------
        ax_submov = axes[3, idx] if num_stimuli > 1 else axes[3]
        bell_shaped_functions = []  # Para almacenar las funciones bell-shaped de cada ensayo

        for trial_idx, vel in enumerate(data['velocities']):
            if len(vel) <= 1 or np.isnan(vel).all():
                logging.warning(f"Trial {trial_idx} tiene datos de velocidad insuficientes o inválidos.")
                continue
            frames_vel = np.arange(len(vel))
            t = frames_vel / 100  # Convertir frames a tiempo en segundos

            # Obtener movimientos que inician durante el estímulo para este ensayo
            movements_in_trial = [mr for mr in data['movement_ranges'] if mr['Ensayo'] == trial_idx + 1 and mr['Periodo'] == 'Durante Estímulo']

            if not movements_in_trial:
                logging.info(f"No hay movimientos que inician durante el estímulo para el ensayo {trial_idx + 1}")
                continue

            for movement in movements_in_trial:
                movement_start = movement['Inicio Movimiento (Frame)']
                movement_end = movement['Fin Movimiento (Frame)']

                # Extraer el segmento de velocidad correspondiente al movimiento
                segment_indices = np.arange(movement_start, movement_end + 1)
                # Asegurarse de que los índices estén dentro del rango de vel
                segment_indices = segment_indices[(segment_indices >= 0) & (segment_indices < len(vel))]

                t_segment = segment_indices / 100  # Convertir a tiempo en segundos
                vel_segment = vel[segment_indices]

                if len(vel_segment) <= 1 or np.isnan(vel_segment).all():
                    logging.warning(f"Movimiento en el ensayo {trial_idx + 1} tiene datos de velocidad insuficientes o inválidos.")
                    continue

                # Ajustar el perfil de velocidad al modelo de mínimo jerk
                n_submovements = 1  # Analizando movimientos individuales que inician durante el estímulo

                result = fit_velocity_profile(t_segment, vel_segment, n_submovements)
                if result is None:
                    logging.warning(f"Omitiendo movimiento en el ensayo {trial_idx + 1} debido a datos insuficientes para el ajuste.")
                    continue

                fitted_params = result.x
                # Extraer parámetros del submovimiento
                A = fitted_params[0]
                t0 = fitted_params[1]
                T = fitted_params[2]
                submovement = {'A': A, 't0': t0, 'T': T}

                # Generar la función bell-shaped usando la velocidad de mínimo jerk
                v_sm = minimum_jerk_velocity(t_segment, A, t0, T)

                # Almacenar la función bell-shaped
                bell_shaped_functions.append({'t': t_segment, 'v': v_sm})

                # Graficar la función bell-shaped
                ax_submov.plot(t_segment, v_sm, linestyle='-', linewidth=1, color='red', alpha=0.5)

        # Después de procesar todos los ensayos, calcular la función bell-shaped promedio
        if bell_shaped_functions:
            # Re-samplear todas las funciones bell-shaped en un vector de tiempo común
            # Primero, encontrar el rango de tiempo común
            t_min = min(bf['t'][0] for bf in bell_shaped_functions)
            t_max = max(bf['t'][-1] for bf in bell_shaped_functions)
            t_common = np.linspace(t_min, t_max, 500)  # 500 puntos

            # Interpolar cada función bell-shaped en t_common
            v_interp_list = []
            for bf in bell_shaped_functions:
                v_interp = np.interp(t_common, bf['t'], bf['v'])
                v_interp_list.append(v_interp)

            # Calcular el promedio
            v_mean = np.mean(v_interp_list, axis=0)

            # Graficar la función bell-shaped promedio
            ax_submov.plot(t_common, v_mean, linestyle='-', linewidth=2, color='blue', alpha=1, label='Promedio')

        # Resaltar el periodo de estimulación
        ax_submov.axvspan(start_time, current_time, color='blue', alpha=0.1)

        ax_submov.set_title(f'Submovimientos - {stimulus_key}', fontsize=12)
        ax_submov.set_xlabel('Tiempo (s)')
        ax_submov.set_ylabel('Velocidad (pixeles/s)')
        ax_submov.set_xlim(0, max_time)
        ax_submov.set_ylim(vel_min, vel_max)

        # Configurar ticks para el eje X
        ax_submov.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_submov.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_submov.tick_params(axis='x', which='minor', length=4, color='grey')
        ax_submov.legend(loc='upper right')

        # -----------------------
        # 5. Graficar Estímulo
        # -----------------------
        ax_stim = axes[4, idx] if num_stimuli > 1 else axes[4]
        x_vals = [start_time]
        y_vals = [0]
        current_time_stimulus = start_time
        for amp, dur in zip(data['amplitude_list'], data['duration_list']):
            next_time = current_time_stimulus + dur / 100  # Convertir duración a segundos
            x_vals.extend([current_time_stimulus, next_time])
            y_vals.extend([amp / 1000, amp / 1000])  # Convertir a μA
            current_time_stimulus = next_time

        # Graficar el estímulo sin cortar
        ax_stim.step(x_vals, y_vals, color='purple', where='pre', linewidth=1,
                     label=f'Amplitud(es): {data["amplitud_real"]} μA')
        
        # Establecer límites
        ax_stim.set_ylim(amp_min - 20, amp_max + 20)
        ax_stim.set_xlim(0, max_time)  # Usar max_time fijo para evitar escalamiento
        
        # Añadir líneas punteadas en los valores máximos y mínimos de amplitud
        ax_stim.axhline(amp_min, color='darkblue', linestyle=':', linewidth=1)
        ax_stim.axhline(amp_max, color='darkblue', linestyle=':', linewidth=1)
        
        # Establecer ticks fijos en el eje Y
        ax_stim.set_yticks([amp_min, 0, amp_max])
        
        # Cambiar el color de las etiquetas de los ticks en max_amp y min_amp
        yticks = ax_stim.get_yticks()
        yticklabels = ax_stim.get_yticklabels()
        for tick, label in zip(yticks, yticklabels):
            if tick == amp_min or tick == amp_max:
                label.set_color('darkblue')
        
        ax_stim.set_xlabel('Tiempo (s)')
        ax_stim.set_ylabel('Amplitud (μA)')
        
        ax_stim.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax_stim.axvspan(start_time, current_time, color='blue', alpha=0.3)
        
        # Añadir líneas de cuadrícula verticales
        for gs in grid_seconds:
            ax_stim.axvline(gs, color='lightgray', linestyle='--', linewidth=0.5)
        
        # Configurar ticks mayores y menores para el eje X
        ax_stim.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_stim.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_stim.tick_params(axis='x', which='minor', length=4, color='grey')
        
        # Añadir texto de parámetros del estímulo
        estimulo_params_text = f"Forma: {data['form']}\nDuración: {data['duration_ms']} ms\nFrecuencia: {data['frequency']} Hz"
        ax_stim.text(0.95, 0.95, estimulo_params_text, transform=ax_stim.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Mostrar leyenda
        ax_stim.legend(loc='lower right', fontsize=9)
    
    # Título principal
    main_title = f'{group_name} - {body_part} - Día {dia_experimental}'
    fig.suptitle(main_title, fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar para el título superior
    
    # Guardar y cerrar la figura
    try:
        plt.savefig(output_png_path)
        logging.info(f"Gráfico guardado en: {output_png_path}")
        print(f"Gráfico guardado en: {output_png_path}")
    except Exception as e:
        logging.error(f"Error al guardar el gráfico en {output_png_path}: {e}")
        print(f"Error al guardar el gráfico en {output_png_path}: {e}")
    finally:
        plt.close()

        
# Función modificada para recolectar datos de velocidad y umbrales
def collect_velocity_threshold_data():
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")
    
    total_trials = 0
    all_movement_data = []  # Para recolectar datos de movimiento a través de los ensayos
    thresholds_data = []  # Para almacenar los umbrales para cada articulación
    processed_combinations = set()  # Para verificar las combinaciones procesadas
    movement_ranges_all = []  # Para almacenar todos los movement_ranges

    grouped_by_day = stimuli_info.groupby('Dia experimental')

    for dia_experimental, day_df in grouped_by_day:
        print(f'Procesando Día Experimental: {dia_experimental}')
        logging.info(f'Procesando Día Experimental: {dia_experimental}')
        print(f'Número de ensayos en day_df: {len(day_df)}')
        logging.info(f'Número de ensayos en day_df: {len(day_df)}')

        # Normalizar 'Forma del Pulso' en day_df
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()

        # Restablecer el índice para obtener el orden de los estímulos
        day_df = day_df.reset_index(drop=True)
        day_df['Order'] = day_df.index + 1  # El primer estímulo tendrá Order = 1

        for part in body_parts:
            # Añadir registro de depuración para verificar las combinaciones
            logging.info(f'Procesando: Día {dia_experimental}, Articulación {part}')
            print(f'Procesando: Día {dia_experimental}, Articulación {part}')
            processed_combinations.add((dia_experimental, 'All_Stimuli', part))  # 'All_Stimuli' indica que no hay grupos específicos

            # Inicializar diccionarios para acumular datos de todos los estímulos
            all_stimuli_data = {}  # Clave: stimulus_key, Valor: datos para graficar

            # Velocidades pre-estímulo para la articulación específica en todos los ensayos del día
            pre_stim_velocities = []

            # Iterar sobre todos los ensayos del día para el cálculo del umbral
            for index, row in day_df.iterrows():
                camara_lateral = row['Camara Lateral']

                if pd.notna(camara_lateral):
                    matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
                    if not matching_segment.empty:
                        matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                        # Extraer datos de velocidad pre-estímulo para la articulación
                        for _, segment_row in matching_segment_sorted.iterrows():
                            nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                            csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                            if csv_path:
                                velocidades, posiciones = calcular_velocidades(csv_path)

                                vel = velocidades.get(part, [])
                                if len(vel) > 0:
                                    vel_pre_stim = vel[:100]  # Primeros 100 frames para pre-estímulo
                                    vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]  # Eliminar NaN
                                    pre_stim_velocities.extend(vel_pre_stim)

            # Calcular el umbral basado en las velocidades pre-estímulo de todos los estímulos
            vel_list = pre_stim_velocities

            if len(vel_list) < 10:
                logging.warning(f'Datos insuficientes para calcular el umbral para {part} en el día {dia_experimental}')
                print(f'Datos insuficientes para calcular el umbral para {part} en el día {dia_experimental}')
                continue

            # Manejo de outliers: eliminar valores que excedan 3 desviaciones estándar
            mean_vel_pre = np.nanmean(vel_list)
            std_vel_pre = np.nanstd(vel_list)
            vel_list_filtered = [v for v in vel_list if (mean_vel_pre - 3 * std_vel_pre) <= v <= (mean_vel_pre + 3 * std_vel_pre)]

            if len(vel_list_filtered) < 10:
                logging.warning(f'Datos insuficientes después de eliminar outliers para {part} en el día {dia_experimental}')
                print(f'Datos insuficientes después de eliminar outliers para {part} en el día {dia_experimental}')
                continue

            # Recalcular la media y desviación estándar sin outliers
            mean_vel_pre = np.nanmean(vel_list_filtered)
            std_vel_pre = np.nanstd(vel_list_filtered)
            threshold = mean_vel_pre + 2 * std_vel_pre  # Umbral es media + 2*desviación estándar

            # Registrar los cálculos de umbral para verificación
            logging.info(f'Umbral calculado para {part} en el día {dia_experimental}: Media={mean_vel_pre:.4f}, Desviación Estándar={std_vel_pre:.4f}, Umbral={threshold:.4f}')
            print(f'Umbral calculado para {part} en el día {dia_experimental}: Media={mean_vel_pre:.4f}, Desviación Estándar={std_vel_pre:.4f}, Umbral={threshold:.4f}')

            # Almacenar datos de umbrales
            thresholds_data.append({
                'body_part': part,
                'Dia experimental': dia_experimental,
                'threshold': threshold,
                'mean_pre_stim': mean_vel_pre,
                'std_pre_stim': std_vel_pre,
                'num_pre_stim_values': len(vel_list_filtered)
            })

            # Ahora, procesar cada estímulo individualmente utilizando el umbral calculado
            unique_stimuli = day_df.drop_duplicates(
                subset=['Forma del Pulso', 'Duración (ms)'],
                keep='first'
            )[['Forma del Pulso', 'Duración (ms)', 'Order']]

            for _, stim in unique_stimuli.iterrows():
                forma_pulso = stim['Forma del Pulso'].lower()
                duracion_ms = stim.get('Duración (ms)', None)
                order = stim['Order']  # Obtener el número de orden del estímulo

                if duracion_ms is not None:
                    stim_df = day_df[
                        (day_df['Forma del Pulso'].str.lower() == forma_pulso) &
                        (day_df['Duración (ms)'] == duracion_ms)
                    ]
                else:
                    stim_df = day_df[
                        (day_df['Forma del Pulso'].str.lower() == forma_pulso)
                    ]

                if stim_df.empty:
                    logging.debug(f"No hay datos para el estímulo: {forma_pulso}, Duración: {duracion_ms}ms")
                    continue  # No hay datos para este estímulo

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
                                                vel_pre_stim = vel[:start_frame]
                                                vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                                if len(vel_pre_stim) == 0:
                                                    logging.warning(f"Trial has no valid pre-stimulus velocity data.")
                                                    continue

                                                mean_vel_pre_trial = np.nanmean(vel_pre_stim)
                                                if mean_vel_pre_trial > mean_vel_pre + 3 * std_vel_pre:
                                                    logging.info(f"Excluyendo ensayo debido a movimiento pre-estímulo alto.")
                                                    continue  # Excluir este ensayo

                                                total_trials_part += 1

                                                # Determinar múltiples segmentos de movimiento
                                                frames_vel = np.arange(len(vel))
                                                above_threshold = (vel > threshold)
                                                indices_above = frames_vel[above_threshold]

                                                if len(indices_above) > 0:
                                                    segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)
                                                    for segment in segments:
                                                        movement_start = segment[0]
                                                        movement_end = segment[-1]

                                                        # Verificar si el movimiento inicia durante el estímulo
                                                        if start_frame <= movement_start <= current_frame:
                                                            movement_trials += 1  # Incrementar el conteo
                                                            break  # Solo necesitamos contar una vez por trial

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
                        'Order': order,  # Añadir el número de orden
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'no_movement_trials': total_trials_part - movement_trials,
                        'proportion_movement': proportion_movement
                    })

                # Seleccionar todas las amplitudes con mayor 'proportion_movement'
                selected_amplitudes = []
                if amplitude_movement_counts:
                    # Encontrar el valor máximo de 'proportion_movement'
                    max_proportion = max([data['proportion_movement'] for data in amplitude_movement_counts.values()])

                    # Obtener todas las amplitudes que tienen la proporción máxima
                    selected_amplitudes = [amp for amp, data in amplitude_movement_counts.items() if data['proportion_movement'] == max_proportion]

                    # Para fines de graficado, seleccionamos los ensayos con las amplitudes seleccionadas
                    selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amplitudes)]

                    # Imprimir las amplitudes seleccionadas para verificar
                    print(f"Amplitudes seleccionadas para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms} ms: {selected_amplitudes} μA con proporción de movimiento: {max_proportion:.2f}")
                    logging.info(f"Amplitudes seleccionadas para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms} ms: {selected_amplitudes} μA con proporción de movimiento: {max_proportion:.2f}")
                else:
                    logging.debug(f"No hay amplitudes con movimientos para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms}ms.")
                    continue  # No hay datos para este estímulo y articulación

                # Calcular y_max_velocity como la media más la desviación estándar de los máximos
                if selected_amplitudes:
                    max_velocities = []
                    for amplitude in selected_amplitudes:
                        data_amp = amplitude_movement_counts.get(amplitude, {})
                        max_velocities.extend(data_amp.get('max_velocities', []))
                    if max_velocities:
                        y_max_velocity = np.mean(max_velocities) + np.std(max_velocities)
                    else:
                        y_max_velocity = 50  # Valor fijo para y_max_velocity si no hay datos
                else:
                    y_max_velocity = 50  # Valor fijo para y_max_velocity si no hay datos

                # Obtener la frecuencia para este estímulo
                frequencies = selected_trials['Frecuencia (Hz)'].unique()
                if len(frequencies) == 1:
                    frequency = frequencies[0]
                elif len(frequencies) > 1:
                    logging.warning(f'Múltiples frecuencias encontradas para el estímulo {forma_pulso} {duracion_ms} ms. Usando la primera.')
                    print(f'Múltiples frecuencias encontradas para el estímulo {forma_pulso} {duracion_ms} ms. Usando la primera.')
                    frequency = frequencies[0]
                else:
                    frequency = None  # No hay frecuencia disponible

                # Inicializar variables para almacenar ensayos que pasaron el umbral
                movement_trials_in_selected = 0
                trials_passed = []

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
                                row['Frecuencia (Hz)'] if frequency is None else frequency,
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
                                            # Verificar si el ensayo debe ser excluido
                                            vel_pre_stim = vel[:start_frame]
                                            vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                            if len(vel_pre_stim) == 0:
                                                logging.warning(f"Trial {trial_counter} has no valid pre-stimulus velocity data.")
                                                continue

                                            mean_vel_pre_trial = np.nanmean(vel_pre_stim)
                                            if mean_vel_pre_trial > mean_vel_pre + 3 * std_vel_pre:
                                                logging.info(f"Excluyendo ensayo {trial_counter} debido a movimiento pre-estímulo alto.")
                                                continue  # Excluir este ensayo

                                            group_velocities.append(vel)
                                            group_positions['x'].append(pos['x'])
                                            group_positions['y'].append(pos['y'])
                                            group_trial_indices.append(trial_counter)

                                            # Determinar si el ensayo pasó el umbral durante el estímulo
                                            vel_stimulus = vel[start_frame:current_frame]
                                            trial_passed = np.any(vel_stimulus > threshold)
                                            trials_passed.append(trial_passed)
                                            if trial_passed:
                                                movement_trials_in_selected += 1

                                            # Determinar múltiples segmentos de movimiento
                                            frames_vel = np.arange(len(vel))
                                            above_threshold = (vel > threshold)

                                            indices_above = frames_vel[above_threshold]

                                            # Encontrar segmentos continuos donde la velocidad excede el umbral
                                            if len(indices_above) > 0:
                                                segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)
                                                for segment in segments:
                                                    movement_start = segment[0]
                                                    movement_end = segment[-1]

                                                    # Calcular latencia al pico del movimiento (en segundos)
                                                    segment_velocities = vel[segment]
                                                    max_vel_index = np.argmax(segment_velocities)
                                                    peak_frame = segment[max_vel_index]
                                                    latency_to_peak = (peak_frame - start_frame) / 100  # Asumiendo 100 fps
                                                    peak_velocity = segment_velocities[max_vel_index]  # Valor del pico de velocidad

                                                    total_duration = (movement_end - movement_start) / 100

                                                    movement_onset_latency = (movement_start - start_frame) / 100  # Asumiendo 100 fps

                                                    # Determinar el periodo del movimiento basado en el inicio
                                                    if movement_start < start_frame:
                                                        periodo = 'Pre-Estímulo'
                                                    elif start_frame <= movement_start <= current_frame:
                                                        periodo = 'Durante Estímulo'
                                                    else:
                                                        periodo = 'Post-Estímulo'

                                                    # Almacenar todos los movimientos relevantes
                                                    if periodo in ['Pre-Estímulo', 'Durante Estímulo', 'Post-Estímulo']:
                                                        movement_data = {
                                                            'Ensayo': trial_counter + 1,  # Sumar 1 para que los ensayos empiecen en 1
                                                            'Inicio Movimiento (Frame)': movement_start,
                                                            'Fin Movimiento (Frame)': movement_end,
                                                            'Latencia al Inicio (s)': movement_onset_latency,
                                                            'Latencia al Pico (s)': latency_to_peak,
                                                            'Valor Pico (velocidad)': peak_velocity,
                                                            'Duración Total (s)': total_duration,
                                                            'Duración durante Estímulo (s)': max(0, min(movement_end, current_frame) - max(movement_start, start_frame)) / 100,
                                                            'body_part': part,
                                                            'Dia experimental': dia_experimental,
                                                            'Order': order,  # Añadir el número de orden
                                                            'Estímulo': f"{forma_pulso.capitalize()}_{duracion_ms}ms",
                                                            'Periodo': periodo,
                                                            'Forma del Pulso': forma_pulso.capitalize(),  # Añadido
                                                            'Duración (ms)': duracion_ms  # Añadido
                                                        }
                                                        movement_ranges.append(movement_data)
                                                        movement_ranges_all.append(movement_data)

                                            trial_counter += 1  # Incrementar el contador de ensayos

                if len(group_velocities) == 0:
                    logging.debug(f"No hay datos de velocidades para graficar para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms} ms.")
                    print(f"No hay datos de velocidades para graficar para {part} en el día {dia_experimental}, estímulo {forma_pulso} {duracion_ms} ms.")
                    continue  # No hay datos para graficar

                # Actualizar total_trials con el número de ensayos después de excluir
                total_trials_filtered = len(group_velocities)

                # Crear una clave única para este estímulo con el número de orden
                if duracion_ms is not None:
                    stimulus_key = f"{order}. {forma_pulso.capitalize()}_{duracion_ms}ms"
                else:
                    stimulus_key = f"{order}. {forma_pulso.capitalize()}"

                # Almacenar los datos para graficar
                all_stimuli_data[stimulus_key] = {
                    'velocities': group_velocities,
                    'positions': group_positions,
                    'threshold': threshold,
                    'amplitude_list': amplitude_list,
                    'duration_list': duration_list,
                    'start_frame': start_frame,
                    'current_frame': current_frame,
                    'mean_vel_pre': mean_vel_pre,
                    'std_vel_pre': std_vel_pre,
                    'amplitud_real': selected_amplitudes,
                    'y_max_velocity': y_max_velocity,
                    'trial_indices': group_trial_indices,
                    'form': forma_pulso.capitalize(),
                    'duration_ms': duracion_ms,
                    'frequency': frequency,
                    'movement_ranges': movement_ranges,
                    'movement_trials': movement_trials_in_selected,
                    'total_trials': total_trials_filtered,  # Usar total de ensayos después de excluir
                    'trials_passed': trials_passed,
                    'Order': order  # Añadir el número de orden
                }

            if len(all_stimuli_data) == 0:
                logging.debug(f"No hay estímulos para graficar para {part} en el día {dia_experimental}.")
                print(f"No hay estímulos para graficar para {part} en el día {dia_experimental}.")
                continue  # No hay datos para graficar para este grupo

            # Generar el gráfico consolidado para esta articulación y día
            stimuli_in_group_names = '_'.join([
                f"{s}" for s in all_stimuli_data.keys()
            ])
            output_png_path = os.path.join(
                output_comparisons_dir,
                f'All_Stimuli_{sanitize_filename(part)}_dia_{sanitize_filename(str(dia_experimental))}.png'
            )
            logging.info(f"Guardando gráfico consolidado en: {output_png_path}")
            print(f"Guardando gráfico consolidado en: {output_png_path}")

            plot_all_stimuli_graphs(
                all_stimuli_data,
                group_name='All_Stimuli',
                body_part=part,
                dia_experimental=dia_experimental,
                output_png_path=output_png_path
            )

    # Crear un DataFrame a partir de los datos de movimiento recolectados
    counts_df = pd.DataFrame(all_movement_data)
    counts_counts_path = os.path.join(output_comparisons_dir, 'movement_counts_summary.csv')
    counts_df.to_csv(counts_counts_path, index=False)
    logging.info(f"Datos de movimiento guardados en {counts_counts_path}")
    print(f"Datos de movimiento guardados en {counts_counts_path}")

    # Crear y guardar el DataFrame de umbrales
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_counts_path = os.path.join(output_comparisons_dir, 'thresholds_summary.csv')
    thresholds_df.to_csv(thresholds_counts_path, index=False)
    logging.info(f"Datos de umbrales guardados en {thresholds_counts_path}")
    print(f"Datos de umbrales guardados en {thresholds_counts_path}")

    # Crear y guardar el DataFrame de movement_ranges
    movement_ranges_df = pd.DataFrame(movement_ranges_all)
    movement_ranges_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    movement_ranges_df.to_csv(movement_ranges_path, index=False)
    logging.info(f"Datos de movement_ranges guardados en {movement_ranges_path}")
    print(f"Datos de movement_ranges guardados en {movement_ranges_path}")

    # Llamar a la función para graficar los datos de movement_ranges
    plot_summary_movement_data(movement_ranges_df)

    # Verificar las combinaciones procesadas
    logging.info("Combinaciones procesadas:")
    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        logging.info(f"Día: {combo[0]}, Grupo: {combo[1]}, Articulación: {combo[2]}")
        print(f"Día: {combo[0]}, Grupo: {combo[1]}, Articulación: {combo[2]}")

    logging.info("Finalizada la recopilación de datos de umbral de velocidad.")
    print("Finalizada la recopilación de datos de umbral de velocidad.")
    return counts_df

# Función modificada para generar gráficos comparativos por día
# Función para simplificar los gráficos de resumen
def plot_summary_movement_data(movement_ranges_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    import numpy as np
    import textwrap

    logging.info("Generando gráficos comparativos simplificados de movimientos durante el estímulo por articulación y día, incluyendo Latencia al Pico y Valor Pico.")
    print("Generando gráficos comparativos simplificados de movimientos durante el estímulo por articulación y día, incluyendo Latencia al Pico y Valor Pico.")

    # Filtrar movimientos durante el estímulo y hacer una copia
    df_durante_estimulo = movement_ranges_df[movement_ranges_df['Periodo'] == 'Durante Estímulo'].copy()

    if df_durante_estimulo.empty:
        logging.info("No hay movimientos durante el estímulo para graficar.")
        print("No hay movimientos durante el estímulo para graficar.")
        return

    # Convertir tiempos a milisegundos
    df_durante_estimulo['Latencia al Inicio (ms)'] = df_durante_estimulo['Latencia al Inicio (s)'] * 1000
    df_durante_estimulo['Duración Total (ms)'] = df_durante_estimulo['Duración Total (s)'] * 1000
    df_durante_estimulo['Latencia al Pico (ms)'] = df_durante_estimulo['Latencia al Pico (s)'] * 1000
    # 'Valor Pico (velocidad)' ya está en unidades de velocidad

    # Asegurarse de que 'Forma del Pulso' y 'Duración (ms)' están bien definidas
    df_durante_estimulo['Forma del Pulso'] = df_durante_estimulo['Forma del Pulso'].str.capitalize()
    df_durante_estimulo['Duración (ms)'] = df_durante_estimulo['Duración (ms)'].astype(int)

    # Definir las duraciones disponibles para cada forma de pulso
    pulse_duration_dict = {
        'Rectangular': [500, 750, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa ascendente': [1000],
        'Rampa descendente': [1000],
        'Triple rombo': [700]
    }

    # Crear una paleta de colores para las formas de pulso
    pulse_shapes = list(pulse_duration_dict.keys())
    colors = sns.color_palette('tab10', n_colors=len(pulse_shapes))
    pulse_shape_colors = dict(zip(pulse_shapes, colors))

    # Añadir mediciones a graficar
    measurements = ['Latencia al Inicio (ms)', 'Latencia al Pico (ms)', 'Duración Total (ms)', 'Valor Pico (velocidad)']

    # Generar gráficos que integran todas las articulaciones
    for dia_experimental in df_durante_estimulo['Dia experimental'].unique():
        df_day = df_durante_estimulo[df_durante_estimulo['Dia experimental'] == dia_experimental]

        if df_day.empty:
            continue

        # Crear los plots
        num_measurements = len(measurements)
        fig, axs = plt.subplots(1, num_measurements, figsize=(6 * num_measurements, 7), sharey=False)

        for idx, measurement in enumerate(measurements):
            ax = axs[idx] if num_measurements > 1 else axs
            boxplot_data = []
            x_positions = []
            x_labels = []
            x_label_positions = []
            box_colors = []
            current_pos = 0
            width = 0.6  # Ancho de cada boxplot
            gap_between_durations = 0.4
            gap_between_pulses = 1.5

            # Para almacenar posiciones centrales para las formas de pulso
            pulse_shape_positions = []

            for pulse_shape in pulse_shapes:
                durations = pulse_duration_dict[pulse_shape]
                data_pulse = df_day[df_day['Forma del Pulso'] == pulse_shape].copy()

                # Verificar si hay datos para esta forma de pulso
                if data_pulse.empty:
                    continue

                num_durations = len(durations)
                positions = np.arange(
                    current_pos,
                    current_pos + num_durations * (width + gap_between_durations),
                    width + gap_between_durations
                )
                for i, dur in enumerate(durations):
                    data_dur = data_pulse[data_pulse['Duración (ms)'] == dur].copy()
                    measurement_data = data_dur[measurement].dropna()
                    print(f"Forma: {pulse_shape}, Duración: {dur} ms, Medición: {measurement}")
                    print(measurement_data.describe())

                    boxplot_data.append(measurement_data)
                    x_positions.append(positions[i])
                    x_labels.append(str(dur) + ' ms')
                    x_label_positions.append(positions[i])
                    box_colors.append(pulse_shape_colors[pulse_shape])
                    # Añadir número de datos encima de cada boxplot
                    n_data = len(measurement_data)
                    if not measurement_data.empty:
                        y_position = measurement_data.max() + (measurement_data.max() * 0.05)
                    else:
                        y_position = 0  # Si no hay datos, colocar en 0
                    ax.text(positions[i], y_position, f'n={n_data}', ha='center', fontsize=9)
                # Añadir posición central de la forma de pulso
                if len(positions) > 0:
                    middle_pos = positions.mean()
                    pulse_shape_positions.append((middle_pos, pulse_shape))
                current_pos = positions[-1] + gap_between_pulses if len(positions) > 0 else current_pos

            if not boxplot_data:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No hay datos disponibles para este día.',
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                continue

            # Hacer el boxplot con matplotlib
            bp = ax.boxplot(boxplot_data, positions=x_positions, widths=width, patch_artist=True)

            # Colorear las cajas
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for median in bp['medians']:
                median.set_color('black')

            # Ajustar los ejes
            ax.set_xticks(x_label_positions)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlabel('Duración (ms)')
            ax.set_ylabel(measurement)
            ax.set_title(measurement)

            # Añadir los nombres de las formas de pulso encima de las duraciones
            ylim = ax.get_ylim()
            for pos, pulse_shape in pulse_shape_positions:
                # Envolver el texto si es demasiado largo
                wrapped_text = '\n'.join(textwrap.wrap(pulse_shape, width=10))
                ax.text(pos, ylim[1] + (ylim[1] - ylim[0]) * 0.05, wrapped_text,
                        ha='center', va='bottom', fontsize=10)

            # Ajustar límites de x e y
            ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
            ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.15)

        # Añadir leyenda de colores para las formas de pulso
        legend_elements = [Patch(facecolor=pulse_shape_colors[ps], label=ps) for ps in pulse_shapes]
        axs[-1].legend(handles=legend_elements, title='Forma del Pulso', loc='upper right')

        # Añadir título general
        fig.suptitle(f'Resumen General - Día Experimental: {dia_experimental}', fontsize=16)

        # Ajustar diseño
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename = f'summary_general_dia_{sanitize_filename(str(dia_experimental))}.png'
        output_path = os.path.join(output_comparisons_dir, output_filename)
        plt.savefig(output_path)
        logging.info(f"Gráfico simplificado general guardado en {output_path}")
        print(f"Gráfico simplificado general guardado en {output_path}")
        plt.close()


    # Generar gráficos por articulación y día (como antes)
    # Obtener combinaciones únicas de 'body_part' y 'Dia experimental'
    body_day_combinations = df_durante_estimulo[['body_part', 'Dia experimental']].drop_duplicates()

    for _, (body_part, dia_experimental) in body_day_combinations.iterrows():
        df_subset = df_durante_estimulo[
            (df_durante_estimulo['body_part'] == body_part) &
            (df_durante_estimulo['Dia experimental'] == dia_experimental)
        ]

        if df_subset.empty:
            continue

        # Crear los plots
        num_measurements = len(measurements)
        fig, axs = plt.subplots(1, num_measurements, figsize=(6 * num_measurements, 7), sharey=False)

        for idx, measurement in enumerate(measurements):
            ax = axs[idx] if num_measurements > 1 else axs
            boxplot_data = []
            x_positions = []
            x_labels = []
            x_label_positions = []
            box_colors = []
            current_pos = 0
            width = 0.6  # Ancho de cada boxplot
            gap_between_durations = 0.4
            gap_between_pulses = 1.5

            # Para almacenar posiciones centrales para las formas de pulso
            pulse_shape_positions = []

            for pulse_shape in pulse_shapes:
                durations = pulse_duration_dict[pulse_shape]
                data_pulse = df_subset[df_subset['Forma del Pulso'] == pulse_shape].copy()

                # Verificar si hay datos para esta forma de pulso y articulación
                if data_pulse.empty:
                    continue

                num_durations = len(durations)
                positions = np.arange(
                    current_pos,
                    current_pos + num_durations * (width + gap_between_durations),
                    width + gap_between_durations
                )
                for i, dur in enumerate(durations):
                    data_dur = data_pulse[data_pulse['Duración (ms)'] == dur].copy()
                    measurement_data = data_dur[measurement].dropna()
                    boxplot_data.append(measurement_data)
                    x_positions.append(positions[i])
                    x_labels.append(str(dur) + ' ms')
                    x_label_positions.append(positions[i])
                    box_colors.append(pulse_shape_colors[pulse_shape])
                    # Añadir número de datos encima de cada boxplot
                    n_data = len(measurement_data)
                    if not measurement_data.empty:
                        y_position = measurement_data.max() + (measurement_data.max() * 0.05)
                    else:
                        y_position = 0  # Si no hay datos, colocar en 0
                    ax.text(positions[i], y_position, f'n={n_data}', ha='center', fontsize=9)
                # Añadir posición central de la forma de pulso
                if len(positions) > 0:
                    middle_pos = positions.mean()
                    pulse_shape_positions.append((middle_pos, pulse_shape))
                current_pos = positions[-1] + gap_between_pulses if len(positions) > 0 else current_pos

            if not boxplot_data:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No hay datos disponibles para esta articulación y día.',
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                continue

            # Hacer el boxplot con matplotlib
            bp = ax.boxplot(boxplot_data, positions=x_positions, widths=width, patch_artist=True)

            # Colorear las cajas
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for median in bp['medians']:
                median.set_color('black')

            # Ajustar los ejes
            ax.set_xticks(x_label_positions)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlabel('Duración (ms)')
            ax.set_ylabel(measurement)
            ax.set_title(measurement)

            # Añadir los nombres de las formas de pulso encima de las duraciones
            ylim = ax.get_ylim()
            for pos, pulse_shape in pulse_shape_positions:
                # Envolver el texto si es demasiado largo
                wrapped_text = '\n'.join(textwrap.wrap(pulse_shape, width=10))
                ax.text(pos, ylim[1] + (ylim[1] - ylim[0]) * 0.05, wrapped_text,
                        ha='center', va='bottom', fontsize=10)

            # Ajustar límites de x e y
            ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
            ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.15)

        # Añadir leyenda de colores para las formas de pulso
        legend_elements = [Patch(facecolor=pulse_shape_colors[ps], label=ps) for ps in pulse_shapes]
        axs[-1].legend(handles=legend_elements, title='Forma del Pulso', loc='upper right')

        # Añadir título general
        fig.suptitle(f'Articulación: {body_part}, Día Experimental: {dia_experimental}', fontsize=16)

        # Ajustar diseño
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_filename = f'summary_{sanitize_filename(body_part)}_dia_{sanitize_filename(str(dia_experimental))}.png'
        output_path = os.path.join(output_comparisons_dir, output_filename)
        plt.savefig(output_path)
        logging.info(f"Gráfico simplificado guardado en {output_path}")
        print(f"Gráfico simplificado guardado en {output_path}")
        plt.close()



# Función para analizar los mejores bodyparts y estímulos
def analyze_best_bodyparts_and_stimuli(counts_df):
    logging.info("Iniciando análisis de las mejores articulaciones y estímulos.")
    print("Iniciando análisis de las mejores articulaciones y estímulos.")

    # Crear una columna para identificar el estímulo
    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'
    counts_df['Estímulo_label'] = counts_df['Forma del Pulso'].str.capitalize() + '\n' + counts_df['Duración (ms)'].astype(str) + ' ms'

    # Ordenar por proporción de movimiento para identificar las mejores articulaciones y estímulos
    sorted_df = counts_df.sort_values(by='proportion_movement', ascending=False)

    # Mostrar top 5 articulaciones con mayor proporción de movimiento
    top_bodyparts = sorted_df.groupby('body_part')['proportion_movement'].mean().sort_values(ascending=False)
    logging.info("Top articulaciones con mayor proporción de movimiento:")
    logging.info(top_bodyparts.head(5))
    print("Top articulaciones con mayor proporción de movimiento:")
    print(top_bodyparts.head(5))

    # Mostrar top 5 estímulos con mayor proporción de movimiento
    top_stimuli = sorted_df.groupby('Estímulo')['proportion_movement'].mean().sort_values(ascending=False)
    logging.info("\nTop estímulos con mayor proporción de movimiento:")
    logging.info(top_stimuli.head(5))
    print("\nTop estímulos con mayor proporción de movimiento:")
    print(top_stimuli.head(5))

    # Guardar resultados en archivos CSV
    top_bodyparts_path = os.path.join(output_comparisons_dir, 'top_bodyparts.csv')
    top_stimuli_path = os.path.join(output_comparisons_dir, 'top_stimuli.csv')
    top_bodyparts.to_csv(top_bodyparts_path)
    top_stimuli.to_csv(top_stimuli_path)
    logging.info(f"Top articulaciones guardadas en {top_bodyparts_path}")
    logging.info(f"Top estímulos guardados en {top_stimuli_path}")
    print(f"Top articulaciones guardadas en {top_bodyparts_path}")
    print(f"Top estímulos guardados en {top_stimuli_path}")

# Función para generar el heatmap
def plot_heatmap(counts_df):
    logging.info("Iniciando generación del heatmap.")
    print("Iniciando generación del heatmap.")

    # Crear una columna para identificar el estímulo
    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

    # Pivotear los datos para el heatmap de proporción
    try:
        pivot_prop = counts_df.pivot_table(
            index='body_part',
            columns=['Dia experimental', 'Estímulo'],
            values='proportion_movement',
            aggfunc='mean'
        )
        logging.debug("Pivot table para proporción de movimiento creada.")
    except Exception as e:
        logging.error(f'Error al pivotear datos para proporción de movimiento: {e}')
        print(f'Error al pivotear datos para proporción de movimiento: {e}')
        return

    # Pivotear los datos para los counts
    try:
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
        logging.debug("Pivot tables para movement_trials y total_trials creadas.")
    except Exception as e:
        logging.error(f'Error al pivotear datos para movement_trials y total_trials: {e}')
        print(f'Error al pivotear datos para movement_trials y total_trials: {e}')
        return

    # Asegurar que los pivotes tengan los mismos índices y columnas
    common_index = pivot_prop.index.union(pivot_movement.index).union(pivot_total.index)
    common_columns = pivot_prop.columns.union(pivot_movement.columns).union(pivot_total.columns)

    pivot_prop = pivot_prop.reindex(index=common_index, columns=common_columns)
    pivot_movement = pivot_movement.reindex(index=common_index, columns=common_columns)
    pivot_total = pivot_total.reindex(index=common_index, columns=common_columns)
    logging.debug("Reindexación de pivot tables completada.")

    # Crear una matriz de anotaciones con 'movement_trials/total_trials'
    annot_matrix = pivot_movement.fillna(0).astype(int).astype(str) + '/' + pivot_total.fillna(0).astype(int).astype(str)

    # Generar el heatmap
    plt.figure(figsize=(20, 15))
    try:
        sns.heatmap(pivot_prop, annot=annot_matrix, fmt='', cmap='viridis')
        logging.debug("Heatmap generado con éxito.")
    except Exception as e:
        logging.error(f'Error al generar el heatmap: {e}')
        print(f'Error al generar el heatmap: {e}')
        return

    plt.title('Proporción de Movimiento Detectado por Articulación, Día y Estímulo')
    plt.xlabel('Día Experimental y Estímulo')
    plt.ylabel('Articulación')
    plt.tight_layout()

    heatmap_path = os.path.join(output_comparisons_dir, 'heatmap_bodypart_day_stimulus.png')
    try:
        plt.savefig(heatmap_path)
        logging.info(f'Heatmap guardado en {heatmap_path}')
        print(f'Gráfico heatmap_bodypart_day_stimulus.png guardado en {heatmap_path}.')
    except Exception as e:
        logging.error(f'Error al guardar el heatmap: {e}')
        print(f'Error al guardar el heatmap: {e}')
    plt.close()

def plot_effectiveness_over_time(counts_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    logging.info("Generando gráficos cronológicos de efectividad de la estimulación por Día experimental.")
    print("Generando gráficos cronológicos de efectividad de la estimulación por Día experimental.")

    # Asegurarse de que 'Order' es numérico
    counts_df['Order'] = counts_df['Order'].astype(int)

    # Ordenar el DataFrame por 'Dia experimental' y 'Order'
    counts_df = counts_df.sort_values(by=['Dia experimental', 'Order'])

    # Obtener los días experimentales únicos
    dias_experimentales = counts_df['Dia experimental'].unique()

    for dia in dias_experimentales:
        df_dia = counts_df[counts_df['Dia experimental'] == dia]
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            x='Order',
            y='proportion_movement',
            hue='body_part',
            data=df_dia,
            marker='o'
        )
        plt.title(f'Efectividad de la Estimulación a lo largo del Día {dia}')
        plt.xlabel('Orden del Estímulo')
        plt.ylabel('Proporción de Movimiento')
        plt.legend(title='Articulación', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Usar 'Estímulo_label' para las etiquetas del eje X
        stimulus_labels = df_dia.groupby('Order')['Estímulo_label'].first()
        plt.xticks(df_dia['Order'].unique(), stimulus_labels, rotation=45, ha='right')

        output_path = os.path.join(output_comparisons_dir, f'efectividad_dia_{sanitize_filename(str(dia))}.png')
        plt.savefig(output_path)
        logging.info(f"Gráfico de efectividad guardado en {output_path}")
        print(f"Gráfico de efectividad guardado en {output_path}")
        plt.close()


if __name__ == "__main__":
    logging.info("Ejecutando el bloque principal del script.")
    print("Ejecutando el bloque principal del script.")

    # Llamar a collect_velocity_threshold_data
    counts_df = collect_velocity_threshold_data()
    logging.info(f"Counts DataFrame after collection: {counts_df.shape}")
    print("Counts DataFrame after collection:", counts_df.shape)
    print(counts_df.head())

    # Analizar los mejores bodyparts y estímulos
    analyze_best_bodyparts_and_stimuli(counts_df)

    # Generar heatmap
    plot_heatmap(counts_df)

    # Generar gráficos cronológicos de efectividad
    plot_effectiveness_over_time(counts_df)

    # Generar gráficos comparativos de movement_ranges
    movement_ranges_df_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    movement_ranges_df = pd.read_csv(movement_ranges_df_path)
    plot_summary_movement_data(movement_ranges_df)