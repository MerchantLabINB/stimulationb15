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
from matplotlib.patches import Patch

from scipy.signal import savgol_filter, find_peaks
import re
import shutil
import glob  # Importar el módulo glob

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import butter, filtfilt

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
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\decomposed_gaussian_velocity_plots24'

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
    #'Nudillo Central': 'purple',
    #'DedoMedio': 'pink',
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}


def butter_lowpass_filter(data, cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def five_point_central_diff(y, dt):
    """
    Calcula la derivada usando la fórmula de diferencias centrales de cinco puntos.
    """
    v = np.zeros_like(y)
    # Aplicar la fórmula para índices centrales
    for i in range(2, len(y) - 2):
        v[i] = (-y[i + 2] + 8 * y[i + 1] - 8 * y[i - 1] + y[i - 2]) / (12 * dt)
    # Para los bordes, se puede usar un método de orden inferior o dejar en cero.
    return v

# Suavizado biomecánico usando SavGol
def suavizar_velocidad_savgol(vel, window_length=31, polyorder=3):
    if len(vel) < window_length:
        return vel
    return savgol_filter(vel, window_length=window_length, polyorder=polyorder)

def suavizar_velocidad_loess(vel, frac=0.1):
    if len(vel)<5:
        return vel
    lo = lowess(vel, np.arange(len(vel)), frac=frac, return_sorted=False)
    return lo


def detectar_submovimientos_en_segmento(vel_segment, threshold):
    """
    Detecta picos locales (submovimientos) dentro de un segmento de velocidad 
    usando find_peaks, asegurando duración mínima.
    """
    # Suavizar el segmento con SavGol
    vel_suav_seg = suavizar_velocidad_savgol(vel_segment, window_length=31, polyorder=3)
    peak_indices, _ = find_peaks(vel_suav_seg, height=threshold*0.2)

    valid_peaks = []
    for pk in peak_indices:
        # Extender hacia atrás y adelante para medir duración sobre el umbral principal
        start_pk = pk
        while start_pk > 0 and vel_suav_seg[start_pk] > threshold:
            start_pk -= 1
        end_pk = pk
        while end_pk < len(vel_suav_seg)-1 and vel_suav_seg[end_pk] > threshold:
            end_pk += 1
        if (end_pk - start_pk) >= 3:  # al menos 3 frames (30 ms)
            valid_peaks.append(pk)
    return valid_peaks


body_parts = list(body_parts_specific_colors.keys())
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def fit_gaussian_submovement(t_segment, v_segment, threshold):
    """
    Ajusta la Gaussiana:
      mu_gauss  = tiempo del pico real de v_segment
      A_gauss   = valor de ese pico (sin escalas extras)
      sigma_gauss = se estima a partir del ancho sobre 'threshold', 
                    o como gustes.
    """
    if len(t_segment) < 3:
        return {'A_gauss': np.nan, 'mu_gauss': np.nan, 'sigma_gauss': np.nan}

    # 1) Localizar el pico real
    peak_idx = np.argmax(v_segment)
    A_gauss  = v_segment[peak_idx]         # Amplitud = velocidad real en el pico
    mu_gauss = t_segment[peak_idx]         # Tiempo = instante del pico
    
    # 2) Calcular sigma
    indices_above = np.where(v_segment > threshold)[0]
    if len(indices_above) < 2:
        sigma_gauss = 0.01
    else:
        width_frames = indices_above[-1] - indices_above[0]
        width_time   = width_frames / 100.0  # asumiendo 100fps => 1 frame = 0.01 s
        sigma_gauss  = max(width_time/4.0, 0.001)

    return {
        'A_gauss': A_gauss,
        'mu_gauss': mu_gauss,
        'sigma_gauss': sigma_gauss
    }






def minimum_jerk_velocity(t, *params):
    n_submovements = len(params) // 3
    v_total = np.zeros_like(t)
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        if T <= 0:
            continue
        tau = (t - t0) / T
        valid_idx = (tau >= 0) & (tau <= 1)
        v = np.zeros_like(t)
        v[valid_idx] = A * 30 * (tau[valid_idx]**2) * (1 - tau[valid_idx])**2
        v_total += v
    return v_total

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
    # Detectamos picos en la velocidad observada
    peak_indices, _ = find_peaks(observed_velocity, height=np.mean(observed_velocity), distance=10)
    peak_times = t[peak_indices]
    peak_amplitudes = observed_velocity[peak_indices]
    
    # Seleccionamos los n_submovements picos más prominentes
    if len(peak_amplitudes) >= n_submovements:
        top_indices = np.argsort(peak_amplitudes)[-n_submovements:]
        peak_times = peak_times[top_indices]
        peak_amplitudes = peak_amplitudes[top_indices]
    else:
        # En caso de no encontrar suficientes picos, completar con distribuciones uniformes
        extras = n_submovements - len(peak_amplitudes)
        peak_times = np.concatenate([peak_times, np.linspace(t[0], t[-1], extras)])
        peak_amplitudes = np.concatenate([peak_amplitudes, 
                                          np.full(extras, np.max(observed_velocity)/n_submovements)])
    
    total_time = t[-1] - t[0]
    params_init = []
    for i in range(n_submovements):
        A_init = peak_amplitudes[i]
        t0_init = peak_times[i]
        # Estimar T como un valor fijo o basado en análisis adicional del pico
        T_init = total_time / n_submovements  
        params_init.extend([A_init, t0_init, T_init])

    # Definir límites como antes
    lower_bounds = []
    upper_bounds = []
    for i in range(n_submovements):
        min_T = 0.01
        lower_bounds.extend([0, t[0], min_T])
        upper_bounds.extend([np.inf, t[-1], total_time])
    
    params_init = np.maximum(params_init, lower_bounds)
    params_init = np.minimum(params_init, upper_bounds)

    try:
        result = least_squares(
            lambda p: sum_of_minimum_jerk(t, *p) - observed_velocity,
            x0=params_init,
            bounds=(lower_bounds, upper_bounds)
        )
    except ValueError as e:
        logging.error(f"Fallo en el ajuste: {e}")
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

def aplicar_moving_average(data, window_size=10):
    """
    Aplica un Moving Average (promedio móvil) a la señal.
    - data: señal de entrada.
    - window_size: número de frames en la ventana del promedio móvil.
    """
    if len(data) < window_size or window_size < 1:
        return data  # Devuelve la señal sin cambios si la ventana es muy grande
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


# Función para calcular velocidades y posiciones para cada articulación con suavizado
def calcular_velocidades(csv_path):
    logging.debug(f"Calculando velocidades para CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        logging.debug(f"Archivo CSV cargado: {csv_path}")
        
        # Ajustar nombres de columnas
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        if 'scorer_bodyparts_coords' in df.columns:
            df = df.drop(columns=['scorer_bodyparts_coords'])
        
        body_parts_adjusted = [part.replace('ñ', 'n').replace(' ', '_') for part in body_parts]
        velocidades = {}
        posiciones = {}
        
        for part_original, part in zip(body_parts, body_parts_adjusted):
            x_col = y_col = likelihood_col = None

            # Identificar las columnas relevantes para cada parte del cuerpo
            for col in df.columns:
                if col.endswith('_x') and part in col:
                    x_col = col
                elif col.endswith('_y') and part in col:
                    y_col = col
                elif col.endswith('_likelihood') and part in col:
                    likelihood_col = col

            if not x_col or not y_col or not likelihood_col:
                logging.warning(f"Columns for {part_original} incompletas en {csv_path}.")
                continue

            df_filtered = df[df[likelihood_col] > 0.1]
            logging.info(f'{part_original} en {csv_path}: {len(df_filtered)}/{len(df)} frames válidos.')

            if df_filtered.empty:
                velocidades[part_original] = np.array([])
                posiciones[part_original] = {'x': np.array([]), 'y': np.array([])}
                continue

            x = df_filtered[x_col].values
            y = df_filtered[y_col].values

            # ---- 1) Filtro butterworth a la posición
            fs = 100.0
            cutoff = 10.0
            x_filt = butter_lowpass_filter(x, cutoff, fs)
            y_filt = butter_lowpass_filter(y, cutoff, fs)
            
            # ---- 2) Derivada con 5-point central diff
            dt = 1.0 / fs
            vx = five_point_central_diff(x_filt, dt)
            vy = five_point_central_diff(y_filt, dt)
            
            # ---- 3) Magnitud de la velocidad
            v_butter = np.hypot(vx, vy)
            
            # ---- 4) Aplicar moving average 10
            v_butter_sm = aplicar_moving_average(v_butter, window_size=10)
            
            # Guardar esa velocidad final
            velocidades[part_original] = v_butter_sm
            # Y posiciones filtradas (opcional) o crudas según se prefiera
            posiciones[part_original] = {'x': x_filt, 'y': y_filt}

        logging.info(f"Finalizado cálculo de velocidades para {csv_path}.")
        return velocidades, posiciones

    except Exception as e:
        logging.error(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        return {}, {}


# Ahora, en la detección de picos de submovimiento utilizamos find_peaks:
def detectar_picos_submovimiento(vel_suav_segment, threshold):
    """
    Detecta picos locales (submovimientos) dentro de un segmento de velocidad 
    usando find_peaks, asegurando duración mínima y distancia entre picos.
    """
    if len(vel_suav_segment) == 0:
        return []

    # Usar find_peaks con altura mínima de threshold*0.2 y distancia mínima de 10 frames
    peak_indices, peak_props = find_peaks(vel_suav_segment, height=threshold*0.2, distance=10)

    valid_peaks = []
    for pk in peak_indices:
        # Verificar duración sobre el umbral principal
        start_pk = pk
        while start_pk > 0 and vel_suav_segment[start_pk] > threshold:
            start_pk -= 1
        end_pk = pk
        while end_pk < len(vel_suav_segment)-1 and vel_suav_segment[end_pk] > threshold:
            end_pk += 1
        segment_length = end_pk - start_pk
        # Requerir mínimo 3 frames (30 ms)
        if segment_length >= 3:
            valid_peaks.append(pk)

    return valid_peaks



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

def plot_trials_side_by_side(stimulus_key, data, body_part, dia_experimental, output_dir,
                             coord_x=None, coord_y=None, dist_ic=None):
    """
    Genera gráficos de ensayos lado a lado (1 columna por trial),
    con 5 paneles (filas) para cada columna:
      1) Desplazamiento (ax_disp)
      2) Velocidad final + Umbral (ax_vel)
      3) Rango de Movimientos sobre Umbral (ax_mov)
      4) Vel. + Gaussianas ajustadas (ax_submov)
      5) Perfil del estímulo (ax_stim)

    En cada panel se dibuja la franja verde (axvspan) indicando el periodo del estímulo.

    Requisitos para que la amplitud de la Gaussiana (panel 4) coincida con la
    altura (pico) de la velocidad real:
      - 'A_gauss' = valor real de la velocidad pico (sin escalas extras).
      - Se grafica la Gaussiana con A_gauss * exp( ... ), sin multiplicaciones extra.

    'data' debe incluir 'trials_data', que es una lista de dicts:
      [
        {
          'velocity': array[ ... ],
          'positions': {'x': array[], 'y': array[]},
          'submovements': [...],
          'movement_ranges': [...],
          'threshold': float,
          'start_frame': int,
          'current_frame': int,
          ...
        },
        { ... }, ...
      ]

    Cada elemento submov en 'submovements':
      {
        'gaussian_params': {
           'A_gauss':   float,
           'mu_gauss':  float en seg,
           'sigma_gauss': float
        },
        't_segment': array,
        'v_sm': array (opcional, p.ej. min. jerk),
        ...
      }    
    """
    trials_data = data.get('trials_data', [])
    if not trials_data:
        logging.warning(
            f"No hay datos de ensayos (trials_data) para {stimulus_key} "
            f"en {body_part} día {dia_experimental}"
        )
        return

    max_per_figure = 15  # número de ensayos máximos por figura
    num_figures = (len(trials_data) // max_per_figure) + (1 if len(trials_data) % max_per_figure != 0 else 0)

    fs = 100.0  # asumiendo 100 fps

    for fig_index in range(num_figures):
        start_idx = fig_index * max_per_figure
        end_idx   = min(start_idx + max_per_figure, len(trials_data))
        subset    = trials_data[start_idx:end_idx]

        # Hallar tiempo máximo y velocidad máxima para escalas
        max_time = 0
        max_vel  = 0
        for td in subset:
            vel_ensayo = td.get('velocity', [])
            if len(vel_ensayo) > 0:
                max_time = max(max_time, len(vel_ensayo) / fs)
                max_vel  = max(max_vel, np.max(vel_ensayo))

        fig_height = 25
        fig_width  = len(subset) * 5

        # 5 filas
        height_ratios = [2, 2, 0.5, 2, 2]
        fig, axes = plt.subplots(
            5, len(subset),
            figsize=(fig_width, fig_height),
            sharey=False,
            gridspec_kw={'height_ratios': height_ratios}
        )
        if len(subset) == 1:
            axes = axes.reshape(5, 1)

        for idx_col, trial in enumerate(subset):
            vel = trial.get('velocity', [])
            pos = trial.get('positions', {'x': [], 'y': []})

            submovements = trial.get('submovements', [])
            mov_ranges   = trial.get('movement_ranges', [])
            threshold    = trial.get('threshold', 0.0)
            mean_vel_pre = trial.get('mean_vel_pre', 0.0)
            std_vel_pre  = trial.get('std_vel_pre', 0.0)

            start_frame   = trial.get('start_frame', 100)
            current_frame = trial.get('current_frame', 200)

            amplitude_list = trial.get('amplitude_list', [])
            duration_list  = trial.get('duration_list', [])

            frames_vel = np.arange(len(vel))
            t_vel      = frames_vel / fs

            ax_disp   = axes[0, idx_col]
            ax_vel    = axes[1, idx_col]
            ax_mov    = axes[2, idx_col]
            ax_submov = axes[3, idx_col]
            ax_stim   = axes[4, idx_col]

            stim_start_s = start_frame / fs
            stim_end_s   = current_frame / fs

            # -------------- Panel 1: Desplazamiento ---------------
            if len(pos['x']) == len(pos['y']) and len(pos['x']) > 0:
                displacement = np.sqrt((pos['x'] - pos['x'][0])**2 + (pos['y'] - pos['y'][0])**2)
                t_disp = np.arange(len(displacement)) / fs
                ax_disp.plot(t_disp, displacement, color='blue')
                ax_disp.set_ylabel('Desplaz. (px)')
                if len(displacement) > 0:
                    ax_disp.set_ylim(0, np.max(displacement) + 10)
            ax_disp.set_title(f"Ensayo {trial.get('trial_index', 0) + 1}")
            ax_disp.set_xlabel('Tiempo (s)')
            ax_disp.set_xlim(0, max_time)
            ax_disp.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # -------------- Panel 2: Vel. Final vs Umbral ----------
            ax_vel.plot(t_vel, vel, color='blue', alpha=0.8, label='Vel. Final')
            ax_vel.axhline(threshold, color='k', ls='--', label=f'Umbral={threshold:.2f}')
            ax_vel.axhline(mean_vel_pre, color='lightcoral', ls='-', label=f'MeanPre={mean_vel_pre:.2f}')
            ax_vel.fill_between(
                t_vel,
                mean_vel_pre - std_vel_pre,
                mean_vel_pre + std_vel_pre,
                color='lightcoral', alpha=0.1,
                label='±1STD Pre'
            )
            ax_vel.set_xlabel('Tiempo (s)')
            ax_vel.set_ylabel('Vel. (px/s)')
            ax_vel.set_xlim(0, max_time)
            ax_vel.set_ylim(0, max_vel + 5)
            if idx_col == 0:
                ax_vel.legend(fontsize=7)
            ax_vel.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # -------------- Panel 3: Rango Movimientos sobre Umbral ----
            ax_mov.set_xlabel('Tiempo (s)')
            ax_mov.set_ylabel('Mov.')
            ax_mov.set_xlim(0, max_time)
            ax_mov.set_ylim(0.95, 1.05)
            ax_mov.set_yticks([1.0])
            ax_mov.set_yticklabels(['Mov'])
            ax_mov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            period_colors = {
                'Pre-Estímulo': 'orange',
                'Durante Estímulo': 'red',
                'Post-Estímulo': 'gray'
            }
            for mov in mov_ranges:
                startF = mov['Inicio Movimiento (Frame)']
                endF   = mov['Fin Movimiento (Frame)']
                periodo = mov['Periodo']
                colorMov = period_colors.get(periodo, 'blue')
                ax_mov.hlines(
                    y=1.0,
                    xmin=startF / fs,
                    xmax=endF / fs,
                    color=colorMov,
                    linewidth=4
                )

            # -------------- Panel 4: Vel. + Gaussianas ---------------
            ax_submov.plot(t_vel, vel, color='darkorange', label='Vel. (px/s)')
            ax_submov.axhline(threshold, color='k', ls='--')
            ax_submov.set_xlabel('Tiempo (s)')
            ax_submov.set_ylabel('Vel. (px/s)')
            ax_submov.set_xlim(0, max_time)
            ax_submov.set_ylim(0, max_vel + 5)
            ax_submov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            valid_gauss_count = 0  # <-- Contador de Gaussianas que sí se muestran

            for i, subm in enumerate(submovements):
                # (Opcional) Graficar "v_sm" si existe
                if 't_segment' in subm and 'v_sm' in subm:
                    t_seg = subm['t_segment']
                    v_seg = subm['v_sm']
                    c = cm.tab10(i % 10)
                    ax_submov.plot(t_seg, v_seg, '--', color=c, alpha=0.9)

                gauss_params = subm.get('gaussian_params', {})
                A_g   = gauss_params.get('A_gauss', np.nan)
                mu_g  = gauss_params.get('mu_gauss', np.nan)
                sig_g = gauss_params.get('sigma_gauss', 0.001)

                if np.isnan(A_g) or np.isnan(mu_g) or sig_g <= 0:
                    continue  # Gauss inválida, la saltamos

                # Queremos graficar en mu ± 3*sigma
                left_g  = mu_g - 3*sig_g
                right_g = mu_g + 3*sig_g
                # Checar duración total < 3 frames => skip
                # 3 frames => 0.03 s a 100 fps.
                total_gauss_width = right_g - left_g  # en segundos
                if total_gauss_width < 0.03:
                    continue  # menor de 3 frames, no se muestra

                # Recortamos a [0, max_time] en caso de salirse
                if left_g < 0:
                    left_g = 0
                if right_g > max_time:
                    right_g = max_time

                # Generar la curva de la Gauss
                t_gauss = np.linspace(left_g, right_g, 300)
                gauss_curve = A_g * np.exp(-((t_gauss - mu_g)**2) / (2 * (sig_g**2)))

                c = cm.tab10(i % 10)
                ax_submov.plot(t_gauss, gauss_curve, color=c, linestyle=':', alpha=0.7)

                # En panel 3 (ax_mov) dibujamos la "banda" de la Gauss
                ax_mov.hlines(
                    y=0.96,
                    xmin=left_g,
                    xmax=right_g,
                    color=c,
                    linewidth=2,
                    alpha=0.8
                )
                ax_mov.plot(mu_g, 0.96, 'o', color=c, markersize=4)

                valid_gauss_count += 1

            # Al terminar de dibujar submovements
            if valid_gauss_count > 0:
                # Agregar leyenda con el número total
                ax_submov.legend([f"Submov. Gauss: {valid_gauss_count}"], loc='upper right', fontsize=7)

            # -------------- Panel 5: Perfil del Estímulo -------------
            ax_stim.set_xlabel('Tiempo (s)')
            ax_stim.set_ylabel('Amplitud (µA)')
            ax_stim.set_xlim(0, max_time)
            ax_stim.axhline(0, color='black', lw=0.5)
            ax_stim.set_title('Perfil del estímulo')
            ax_stim.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # Escalonado del estímulo (si hay amplitude_list/duration_list)
            if amplitude_list and duration_list and len(amplitude_list) == len(duration_list):
                x_vals = [stim_start_s]
                y_vals = [0]
                t_stim = stim_start_s
                for amp, dur in zip(amplitude_list, duration_list):
                    nxt_time = t_stim + (dur / fs)
                    x_vals.extend([t_stim, nxt_time])
                    y_vals.extend([amp, amp])  # amp en µA
                    t_stim = nxt_time
                ax_stim.step(x_vals, y_vals, color='purple', where='pre', linewidth=1)

        figTitle = (f"Día {dia_experimental}, {body_part}, {stimulus_key} "
                    f"- Grupo {fig_index+1}/{num_figures}")
        fig.suptitle(figTitle, fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        day_str = str(dia_experimental).replace('/', '-')
        out_filename = f"Dia_{day_str}_{body_part}_{stimulus_key}_Group_{fig_index+1}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=150)
        logging.info(f"Gráfico guardado en {out_path}")
        print(f"Gráfico guardado en {out_path}")
        plt.close()






# Después de cargar stimuli_info
print("Convirtiendo 'Distancia Intracortical' a punto decimal...")
stimuli_info['Distancia Intracortical'] = stimuli_info['Distancia Intracortical'].astype(str).str.replace(',', '.')
stimuli_info['Distancia Intracortical'] = pd.to_numeric(stimuli_info['Distancia Intracortical'], errors='coerce')
print("Conversión de Distancia Intracortical completada.")

# Ahora agrupamos por múltiples columnas, incluyendo las coordenadas y la distancia intracortical
grouped_data = stimuli_info.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical'])

# Modificar la función collect_velocity_threshold_data para llamar a la nueva función
# Definimos esta lista antes de la función principal para ir agregando información sobre submovimientos.
submovement_summary_all = []

def collect_velocity_threshold_data():
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")

    total_trials = 0
    all_movement_data = []
    thresholds_data = []
    processed_combinations = set()
    movement_ranges_all = []

    # Iterar por día, coordenadas y distancia intracortical (agrupados en grouped_data)
    for (dia_experimental, coord_x, coord_y, dist_ic), day_df in grouped_data:
        print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        print(f'Número de ensayos en este grupo: {len(day_df)}')
        logging.info(f'Número de ensayos en este grupo: {len(day_df)}')

        # Normalizar 'Forma del Pulso' a minúsculas
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()

        # Restablecer el índice para obtener el orden
        day_df = day_df.reset_index(drop=True)
        day_df['Order'] = day_df.index + 1

        for part in body_parts:
            logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}, Articulación {part}')
            print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}, Articulación {part}')
            processed_combinations.add((dia_experimental, coord_x, coord_y, dist_ic, 'All_Stimuli', part))

            all_stimuli_data = {}
            pre_stim_velocities = []

            # ----- CÁLCULO DE UMBRAL (velocidad pre-estímulo) -----
            for index, row in day_df.iterrows():
                camara_lateral = row['Camara Lateral']
                if pd.notna(camara_lateral):
                    matching_segment = segmented_info[
                        segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)
                    ]
                    if not matching_segment.empty:
                        matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')
                        for _, segment_row in matching_segment_sorted.iterrows():
                            nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                            csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                            if csv_path:
                                velocidades, posiciones = calcular_velocidades(csv_path)
                                vel = velocidades.get(part, [])
                                if len(vel) > 0:
                                    # Tomamos frames pre-estímulo ~[0:100]
                                    vel_pre_stim = vel[:100]
                                    vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                    pre_stim_velocities.extend(vel_pre_stim)

            # Cálculo del umbral basado en pre-estímulo
            vel_list = pre_stim_velocities
            if len(vel_list) < 10:
                logging.warning(f'Datos insuficientes para calcular umbral para {part} en Día={dia_experimental}...')
                continue

            mean_vel_pre = np.nanmean(vel_list)
            std_vel_pre  = np.nanstd(vel_list)
            vel_list_filtered = [
                v for v in vel_list
                if (mean_vel_pre - 3*std_vel_pre) <= v <= (mean_vel_pre + 3*std_vel_pre)
            ]
            if len(vel_list_filtered) < 10:
                logging.warning(f'Datos insuficientes tras filtrar outliers para {part}...')
                continue

            mean_vel_pre = np.nanmean(vel_list_filtered)
            std_vel_pre  = np.nanstd(vel_list_filtered)
            threshold    = mean_vel_pre + 2*std_vel_pre

            logging.info(f'Umbral {part} calculado: Media={mean_vel_pre:.4f}, '
                         f'STD={std_vel_pre:.4f}, Umbral={threshold:.4f}')
            thresholds_data.append({
                'body_part': part,
                'Dia experimental': dia_experimental,
                'Coordenada_x': coord_x,
                'Coordenada_y': coord_y,
                'Distancia Intracortical': dist_ic,
                'threshold': threshold,
                'mean_pre_stim': mean_vel_pre,
                'std_pre_stim': std_vel_pre,
                'num_pre_stim_values': len(vel_list_filtered)
            })

            # ----- PROCESAR ESTÍMULOS (forma_pulso, duracion) -----
            unique_stimuli = day_df.drop_duplicates(
                subset=['Forma del Pulso', 'Duración (ms)'],
                keep='first'
            )[['Forma del Pulso', 'Duración (ms)', 'Order']]

            for _, stim in unique_stimuli.iterrows():
                forma_pulso  = stim['Forma del Pulso'].lower()
                duracion_ms  = stim.get('Duración (ms)', None)
                order        = stim['Order']
                stimulus_key = f"{order}. {forma_pulso.capitalize()}"

                if duracion_ms is not None:
                    stim_df = day_df[(day_df['Forma del Pulso'] == forma_pulso) &
                                     (day_df['Duración (ms)'] == duracion_ms)]
                else:
                    stim_df = day_df[day_df['Forma del Pulso'] == forma_pulso]

                if stim_df.empty:
                    continue

                amplitudes = stim_df['Amplitud (microA)'].unique()
                amplitude_movement_counts = {}

                # Para cada amplitud, contar cuántos trials tienen mov
                for amplitude in amplitudes:
                    amplitude_trials = stim_df[stim_df['Amplitud (microA)'] == amplitude]
                    movement_trials = 0
                    total_trials_part = 0
                    max_velocities = []

                    for index, rowAmp in amplitude_trials.iterrows():
                        camara_lateral = rowAmp['Camara Lateral']
                        if pd.notna(camara_lateral):
                            matching_segment = segmented_info[
                                segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)
                            ]
                            if not matching_segment.empty:
                                matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                                # Generar estímulo
                                compensar = False if duracion_ms == 1000 else True
                                amplitude_list, duration_list = generar_estimulo_desde_parametros(
                                    rowAmp['Forma del Pulso'],
                                    amplitude * 1000,
                                    (duracion_ms*1000 if duracion_ms else 1000000),
                                    rowAmp['Frecuencia (Hz)'],
                                    200,
                                    compensar=compensar
                                )

                                start_frame   = 100
                                current_frame = int(start_frame + sum(duration_list))

                                # Buscar CSV de esa cámara
                                for _, segRow in matching_segment_sorted.iterrows():
                                    nombre_segmento = segRow['NombreArchivo'].replace('.mp4','').replace('lateral_','')
                                    csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                                    if csv_path:
                                        velocidades, posiciones = calcular_velocidades(csv_path)
                                        vel = velocidades.get(part, [])
                                        if len(vel) == 0:
                                            continue

                                        vel_pre_stim = vel[:start_frame]
                                        vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                        if len(vel_pre_stim) == 0:
                                            logging.warning("Trial sin datos pre-stim.")
                                            continue

                                        mean_vel_pre_trial = np.nanmean(vel_pre_stim)
                                        if mean_vel_pre_trial > mean_vel_pre + 3*std_vel_pre:
                                            logging.info("Descartando trial por exceso pre-stim.")
                                            continue

                                        total_trials_part += 1

                                        frames_vel     = np.arange(len(vel))
                                        above_threshold= (vel > threshold)
                                        indices_above  = frames_vel[above_threshold]
                                        if len(indices_above) > 0:
                                            # Dividir en sub-segmentos de frames consecutivos
                                            segments = np.split(indices_above, np.where(np.diff(indices_above)!=1)[0]+1)
                                            for seg_above in segments:
                                                movement_start = seg_above[0]
                                                movement_end   = seg_above[-1]
                                                # Checar si inició DURANTE el estímulo
                                                if start_frame <= movement_start <= current_frame:
                                                    movement_trials += 1

                                        maxVel = np.max(vel)
                                        max_velocities.append(maxVel)

                    prop_movement = (movement_trials / total_trials_part) if total_trials_part>0 else 0
                    amplitude_movement_counts[amplitude] = {
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'max_velocities': max_velocities,
                        'proportion_movement': prop_movement
                    }

                    all_movement_data.append({
                        'body_part': part,
                        'Dia experimental': dia_experimental,
                        'Forma del Pulso': forma_pulso,
                        'Duración (ms)': duracion_ms,
                        'Amplitud (microA)': amplitude,
                        'Order': order,
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'no_movement_trials': total_trials_part - movement_trials,
                        'proportion_movement': prop_movement
                    })

                # Elegir amplitudes con mayor prop. de movimiento
                if not amplitude_movement_counts:
                    logging.debug(f"No hay amplitudes con mov. para {part} día {dia_experimental} "
                                  f"{forma_pulso} {duracion_ms}ms.")
                    continue
                max_proportion = max([mdata['proportion_movement']
                                      for mdata in amplitude_movement_counts.values()])
                selected_amplitudes = [
                    amp for amp, mdata in amplitude_movement_counts.items()
                    if mdata['proportion_movement'] == max_proportion
                ]
                selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amplitudes)]
                print(f"Amplitudes selec. {selected_amplitudes} con prop mov={max_proportion:.2f} "
                      f"para {part}, día={dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                logging.info(f"Amplitudes selec. {selected_amplitudes} con prop mov={max_proportion:.2f}")

                # Preparar para graficar
                max_velocities = []
                for ampSel in selected_amplitudes:
                    data_amp = amplitude_movement_counts.get(ampSel, {})
                    max_velocities.extend(data_amp.get('max_velocities', []))
                if max_velocities:
                    y_max_velocity = np.mean(max_velocities) + np.std(max_velocities)
                else:
                    y_max_velocity = 50

                frequencies = selected_trials['Frecuencia (Hz)'].unique()
                if len(frequencies) == 1:
                    frequency = frequencies[0]
                elif len(frequencies) > 1:
                    frequency = frequencies[0]
                    logging.warning(f"Múltiples frecuencias para {forma_pulso} {duracion_ms}ms => usando la primera.")
                else:
                    frequency = None

                movement_trials_in_selected = 0
                trials_passed = []
                group_velocities = []
                group_positions = {'x': [], 'y': []}
                group_trial_indices = []
                trial_counter = 0
                movement_ranges = []
                trials_data = []  # <--- Lista de ensayos (cada uno con sus submovimientos)

                # ---------- PROCESAR TRIALS SELECCIONADOS ----------
                for index, rowStim in selected_trials.iterrows():
                    camara_lateral = rowStim['Camara Lateral']
                    if pd.notna(camara_lateral):
                        matching_segment = segmented_info[
                            segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)
                        ]
                        if not matching_segment.empty:
                            matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                            compensar = False if duracion_ms == 1000 else True
                            amplitude_list, duration_list = generar_estimulo_desde_parametros(
                                rowStim['Forma del Pulso'],
                                rowStim['Amplitud (microA)'] * 1000,
                                (duracion_ms*1000 if duracion_ms else 1000000),
                                (rowStim['Frecuencia (Hz)'] if frequency is None else frequency),
                                200,
                                compensar=compensar
                            )

                            start_frame   = 100
                            current_frame = int(start_frame + sum(duration_list))

                            # Recorrer cada "video" o "segmento" de la cámara
                            for _, segRow in matching_segment_sorted.iterrows():
                                nombre_segmento = segRow['NombreArchivo'].replace('.mp4','').replace('lateral_','')
                                csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                                if csv_path:
                                    velocidades, posiciones = calcular_velocidades(csv_path)
                                    if part in velocidades:
                                        vel = velocidades[part]
                                        pos = posiciones[part]
                                        if len(vel) == 0:
                                            continue

                                        vel_pre_stim = vel[:start_frame]
                                        vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                        if len(vel_pre_stim) == 0:
                                            logging.warning(f"Trial {trial_counter} sin pre-stim.")
                                            continue

                                        mean_vel_pre_trial = np.nanmean(vel_pre_stim)
                                        if mean_vel_pre_trial > mean_vel_pre + 3*std_vel_pre:
                                            logging.info(f"Descartado trial {trial_counter} por pre-stim alto.")
                                            continue

                                        # Revisar si pasa threshold DURANTE:
                                        vel_stimulus = vel[start_frame:current_frame]
                                        trial_passed = np.any(vel_stimulus > threshold)
                                        trials_passed.append(trial_passed)
                                        if trial_passed:
                                            movement_trials_in_selected += 1

                                        group_velocities.append(vel)
                                        group_positions['x'].append(pos['x'])
                                        group_positions['y'].append(pos['y'])
                                        group_trial_indices.append(trial_counter)

                                        # Detectar movimientos y submovimientos para ESTE trial
                                        frames_vel     = np.arange(len(vel))
                                        above_threshold= (vel > threshold)
                                        indices_above  = frames_vel[above_threshold]

                                        submovements_totales = []  # <--- almacenar submovs de este trial
                                        if len(indices_above) > 0:
                                            segments = np.split(indices_above, np.where(np.diff(indices_above)!=1)[0]+1)
                                            for seg_idx, seg_above in enumerate(segments):
                                                movement_start = seg_above[0]
                                                movement_end   = seg_above[-1]

                                                # Etiqueta de periodo (Pre, Durante, Post)
                                                if movement_start < start_frame:
                                                    periodo = 'Pre-Estímulo'
                                                elif start_frame <= movement_start <= current_frame:
                                                    periodo = 'Durante Estímulo'
                                                else:
                                                    periodo = 'Post-Estímulo'

                                                segment_velocities = vel[movement_start:movement_end+1]
                                                max_vel_idx   = np.argmax(segment_velocities)
                                                peak_frame    = movement_start + max_vel_idx
                                                latency2peak  = (peak_frame - start_frame)/100.0
                                                peak_velocity = segment_velocities[max_vel_idx]
                                                total_dur_sec = (movement_end - movement_start)/100.0
                                                onset_lat_sec = (movement_start - start_frame)/100.0

                                                movement_data = {
                                                    'Ensayo': trial_counter+1,
                                                    'Inicio Movimiento (Frame)': movement_start,
                                                    'Fin Movimiento (Frame)':    movement_end,
                                                    'Latencia al Inicio (s)':    onset_lat_sec,
                                                    'Latencia al Pico (s)':      latency2peak,
                                                    'Valor Pico (velocidad)':    peak_velocity,
                                                    'Duración Total (s)':        total_dur_sec,
                                                    'Duración durante Estímulo (s)': max(0, min(movement_end, current_frame) -
                                                                                          max(movement_start,start_frame))/100,
                                                    'body_part': part,
                                                    'Dia experimental': dia_experimental,
                                                    'Order': order,
                                                    'Estímulo': f"{forma_pulso.capitalize()}_{duracion_ms}ms",
                                                    'Periodo': periodo,
                                                    'Forma del Pulso': forma_pulso.capitalize(),
                                                    'Duración (ms)': duracion_ms
                                                }
                                                movement_ranges.append(movement_data)
                                                movement_ranges_all.append(movement_data)

                                                # ===== Ajuste de submovimientos con Minimum Jerk + Gaussian =====
                                                if periodo == 'Durante Estímulo':
                                                    t_segment   = np.arange(movement_start, movement_end+1)/100.0
                                                    vel_segment = vel[movement_start:movement_end+1]

                                                    vel_segment_filtrada = aplicar_moving_average(vel_segment, window_size=6)

                                                    # Submov. detectados
                                                    submov_peak_indices = detectar_submovimientos_en_segmento(
                                                        vel_segment_filtrada, threshold
                                                    )

                                                    # Ajustar con n_submovements = #picos detectados
                                                    n_submov = max(1, len(submov_peak_indices))
                                                    result = fit_velocity_profile(t_segment, vel_segment, n_submov)
                                                    if result is None:
                                                        logging.warning(f"Omitido mov en trial {trial_counter} por error en fitting.")
                                                    else:
                                                        fitted_params = result.x
                                                        # Construimos la info de cada submov
                                                        for i_sub in range(n_submov):
                                                            A_i  = fitted_params[3*i_sub]
                                                            t0_i = fitted_params[3*i_sub+1]
                                                            T_i  = fitted_params[3*i_sub+2]

                                                            subm_latency_onset = t0_i - (start_frame/100.0)
                                                            subm_peak_time     = t0_i + 0.4*T_i
                                                            subm_latency_peak  = subm_peak_time - (start_frame/100.0)
                                                            subm_peak_value    = A_i * 1.728  # factor de conversión?

                                                            # Generar "forma Gauss" (opcional) con la min. jerk
                                                            v_sm = minimum_jerk_velocity(t_segment, A_i, t0_i, T_i)
                                                            gauss_params = fit_gaussian_submovement(t_segment, v_sm, threshold)

                                                            subm_dict = {
                                                                'A': A_i,
                                                                't0': t0_i,
                                                                'T': T_i,
                                                                'latencia_inicio': subm_latency_onset,
                                                                'latencia_pico': subm_latency_peak,
                                                                'valor_pico': subm_peak_value,
                                                                'gaussian_params': gauss_params,
                                                                't_segment': t_segment,
                                                                'v_sm': v_sm,
                                                                'movement_info': movement_data
                                                            }
                                                            submovements_totales.append(subm_dict)
                                        else:
                                            # caso sin frames > threshold => sin submovs
                                            pass

                                        # Crear un "trial_data" para ESTE video
                                        trial_data = {
                                            'velocity': vel,
                                            'positions': pos,
                                            'trial_index': trial_counter,
                                            'submovements': submovements_totales,  # acumulamos aquí
                                            'movement_ranges': [
                                                md for md in movement_ranges
                                                if md['Ensayo'] == (trial_counter+1)
                                            ],
                                            'amplitude_list': amplitude_list,
                                            'duration_list': duration_list,
                                            'start_frame': start_frame,
                                            'current_frame': current_frame,
                                            'threshold': threshold,
                                            'mean_vel_pre': mean_vel_pre,
                                            'std_vel_pre': std_vel_pre
                                        }
                                        trials_data.append(trial_data)

                                        trial_counter += 1

                # Finalmente, guardamos todos los trials (ensayos) en all_stimuli_data
                # para graficar en plot_trials_side_by_side
                if len(trials_data) == 0:
                    logging.debug(f"No hay datos de veloc. para {part} en día {dia_experimental}, "
                                  f"{forma_pulso} {duracion_ms} ms.")
                    print(f"No hay datos de veloc. para {part} en día {dia_experimental}, "
                          f"{forma_pulso} {duracion_ms} ms.")
                    continue

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
                    'total_trials': len(trials_data),
                    'trials_passed': trials_passed,
                    'Order': order,
                    # Lo más importante: la lista de ensayos con submovimientos
                    'trials_data': trials_data
                }

            # (Opcional) graficar todos los estímulos que se guardaron
            for stimulus_key, dataSt in all_stimuli_data.items():
                plot_trials_side_by_side(
                    stimulus_key=stimulus_key,
                    data=dataSt,
                    body_part=part,
                    dia_experimental=dia_experimental,
                    output_dir=output_comparisons_dir,
                    coord_x=coord_x,
                    coord_y=coord_y,
                    dist_ic=dist_ic
                )

    # Al terminar la iteración de todos los días y articulaciones:

    # Guardar CSV: movement_counts_summary
    counts_df = pd.DataFrame(all_movement_data)
    counts_path = os.path.join(output_comparisons_dir, 'movement_counts_summary.csv')
    counts_df.to_csv(counts_path, index=False)
    print(f"Datos de movimiento guardados en: {counts_path}")

    # Guardar thresholds_summary
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_path = os.path.join(output_comparisons_dir, 'thresholds_summary.csv')
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"Datos de umbrales guardados en: {thresholds_path}")

    # Guardar movement_ranges_summary
    movement_ranges_df = pd.DataFrame(movement_ranges_all)
    movement_ranges_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    movement_ranges_df.to_csv(movement_ranges_path, index=False)
    print(f"Datos de movement_ranges guardados en: {movement_ranges_path}")

    # Llamar a la función que grafica el resumen final (si procede)
    if submovement_summary_all:
        submovement_df = pd.DataFrame(submovement_summary_all)
        submovement_path = os.path.join(output_comparisons_dir, 'submovement_summary.csv')
        submovement_df.to_csv(submovement_path, index=False)
        print(f"Datos de submovimientos guardados en: {submovement_path}")

    plot_summary_movement_data(movement_ranges_df)

    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        print(f"Día: {combo[0]}, X={combo[1]}, Y={combo[2]}, Dist={combo[3]}, {combo[4]}, {combo[5]}")

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



# Ejecutar el bloque principal del script
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