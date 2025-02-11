import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from math import sqrt
import matplotlib
matplotlib.use('Agg')  # Para entornos sin interfaz gráfica

import itertools

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

import textwrap

# --- Nuevo para ANOVA:
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\ht_24'

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

# Normalizando 'Forma del Pulso' a minúsculas
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

# Diccionario de colores específicos para cada articulación
body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muneca': 'red',  # Reemplazar 'ñ' por 'n'
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
    return v

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
    Ajusta una Gaussiana a v_segment vs t_segment de manera robusta 
    (usando curve_fit) para obtener A_gauss, mu_gauss y sigma_gauss reales.
    NO multiplica amplitud por factores empíricos.

    Si el ajuste falla, regresa una aproximación mínima.
    """
    from scipy.optimize import curve_fit

    if len(t_segment) < 3:
        return {'A_gauss': np.nan, 'mu_gauss': np.nan, 'sigma_gauss': np.nan}

    # 1) Estimaciones iniciales
    peak_idx = np.argmax(v_segment)
    A_init  = v_segment[peak_idx]   # amplitud inicial
    mu_init = t_segment[peak_idx]   # centro = frame del pico
    # Ej. estimar sigma como ~1/6 del ancho total o fallback si es muy corto
    total_width = t_segment[-1] - t_segment[0]
    sigma_init = max(total_width / 6.0, 0.01)

    p0 = [A_init, mu_init, sigma_init]

    # 2) Definimos bounds para A>=0, sigma>0, mu en [t0, tN]
    lb = [0, t_segment[0], 1e-4]
    ub = [max(v_segment)*5, t_segment[-1], total_width*10]

    try:
        popt, pcov = curve_fit(
            gaussian, 
            t_segment, 
            v_segment, 
            p0=p0, 
            bounds=(lb, ub)
        )
        A_gauss, mu_gauss, sigma_gauss = popt
        return {
            'A_gauss': A_gauss,
            'mu_gauss': mu_gauss,
            'sigma_gauss': sigma_gauss
        }
    except Exception as e:
        # Si falla curve_fit, fallback mínimo
        logging.warning(f"Fallo en curve_fit Gauss: {e}. Usando aproximación simple.")
        peak_idx = np.argmax(v_segment)
        A_gauss  = v_segment[peak_idx]
        mu_gauss = t_segment[peak_idx]
        indices_above = np.where(v_segment > threshold)[0]
        if len(indices_above) < 2:
            sigma_gauss = 0.01
        else:
            width_frames = indices_above[-1] - indices_above[0]
            width_time   = width_frames / 100.0
            sigma_gauss  = max(width_time/4.0, 0.001)

        return {
            'A_gauss': A_gauss,
            'mu_gauss': mu_gauss,
            'sigma_gauss': sigma_gauss
        }


# ------------------------------------------------------------------------------
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
    from scipy.signal import find_peaks
    peak_indices, _ = find_peaks(observed_velocity, height=np.mean(observed_velocity), distance=10)
    peak_times = t[peak_indices]
    peak_amplitudes = observed_velocity[peak_indices]
    
    if len(peak_amplitudes) >= n_submovements:
        top_indices = np.argsort(peak_amplitudes)[-n_submovements:]
        peak_times = peak_times[top_indices]
        peak_amplitudes = peak_amplitudes[top_indices]
    else:
        extras = n_submovements - len(peak_amplitudes)
        peak_times = np.concatenate([peak_times, np.linspace(t[0], t[-1], extras)])
        peak_amplitudes = np.concatenate([peak_amplitudes, 
                                          np.full(extras, np.max(observed_velocity)/n_submovements)])
    
    total_time = t[-1] - t[0]
    params_init = []
    for i in range(n_submovements):
        A_init = peak_amplitudes[i]
        t0_init = peak_times[i]
        T_init = total_time / n_submovements  
        params_init.extend([A_init, t0_init, T_init])

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
            bounds=(lower_bounds, upper_bounds),
            loss='soft_l1',  # Ajuste robusto
            f_scale=0.5      # Factor de escala del error
        )

    except ValueError as e:
        logging.error(f"Fallo en el ajuste: {e}")
        return None

    return result

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def encontrar_csv(camara_lateral, nombre_segmento):
    try:
        match = re.search(r'segment_(\d+_\d+)', nombre_segmento)
        if match:
            digits = match.group(1)
            pattern = f"{camara_lateral}_{digits}*filtered.csv"
            search_pattern = os.path.join(csv_folder, pattern)
            logging.debug(f'Searching for CSV with pattern: {search_pattern}')
            matching_files = glob.glob(search_pattern)
            if matching_files:
                csv_path = matching_files[0]
                logging.debug(f'Archivo CSV filtrado encontrado: {csv_path}')
                return csv_path
            else:
                logging.warning(f'Archivo CSV filtrado no encontrado para cámara={camara_lateral}, seg={nombre_segmento}')
                return None
        else:
            logging.warning(f'No se pudieron extraer dígitos de: {nombre_segmento}')
            return None
    except Exception as e:
        logging.error(f'Error al buscar CSV: {e}')
        return None

def aplicar_moving_average(data, window_size=10):
    if len(data) < window_size or window_size < 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def calcular_velocidades(csv_path):
    logging.debug(f"Calculando velocidades para CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        logging.debug(f"Archivo CSV cargado: {csv_path}")
        
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        if 'scorer_bodyparts_coords' in df.columns:
            df = df.drop(columns=['scorer_bodyparts_coords'])
        
        body_parts_adjusted = [part.replace('ñ', 'n').replace(' ', '_') for part in body_parts]
        velocidades = {}
        posiciones = {}
        
        for part_original, part in zip(body_parts, body_parts_adjusted):
            x_col = y_col = likelihood_col = None

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

            fs = 100.0
            cutoff = 10.0
            x_filt = butter_lowpass_filter(x, cutoff, fs)
            y_filt = butter_lowpass_filter(y, cutoff, fs)
            
            dt = 1.0 / fs
            vx = five_point_central_diff(x_filt, dt)
            vy = five_point_central_diff(y_filt, dt)
            
            v_butter = np.hypot(vx, vy)
            v_butter_sm = aplicar_moving_average(v_butter, window_size=10)
            
            velocidades[part_original] = v_butter_sm
            posiciones[part_original] = {'x': x_filt, 'y': y_filt}

        logging.info(f"Finalizado cálculo de velocidades para {csv_path}.")
        return velocidades, posiciones

    except Exception as e:
        logging.error(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        return {}, {}

def detectar_picos_submovimiento(vel_suav_segment, threshold):
    if len(vel_suav_segment) == 0:
        return []
    peak_indices, peak_props = find_peaks(vel_suav_segment, height=threshold*0.2, distance=10)

    valid_peaks = []
    for pk in peak_indices:
        start_pk = pk
        while start_pk > 0 and vel_suav_segment[start_pk] > threshold:
            start_pk -= 1
        end_pk = pk
        while end_pk < len(vel_suav_segment)-1 and vel_suav_segment[end_pk] > threshold:
            end_pk += 1
        segment_length = end_pk - start_pk
        if segment_length >= 3:
            valid_peaks.append(pk)

    return valid_peaks


def us_to_frames(duracion_us):
    return duracion_us / 10000  # 1 frame = 10,000 µs

def generar_estimulo_desde_parametros(forma, amplitud, duracion, frecuencia, duracion_pulso, compensar):
    logging.debug(f"Generando estímulo con: forma={forma}, amp={amplitud}, dur={duracion}, freq={frecuencia}, pulso={duracion_pulso}, comp={compensar}")
    try:
        forma = forma.strip().lower()

        if duracion <= 0 or frecuencia <= 0 or duracion_pulso <= 0:
            logging.error(f"Parámetros inválidos: duración={duracion}, frecuencia={frecuencia}, pulso={duracion_pulso}")
            print(f"Parámetros inválidos: duración={duracion}, frecuencia={frecuencia}, pulso={duracion_pulso}")
            return [], []

        lista_amplitud, lista_tiempo = estimulo(
            forma=forma, amplitud=amplitud, duracion=duracion,
            frecuencia=frecuencia, duracion_pulso=duracion_pulso, compensar=compensar
        )
        logging.debug("Estímulo generado con éxito.")
        print("Estímulo generado con éxito.")

        if not lista_amplitud or not lista_tiempo:
            logging.error(f"Estímulo inválido. Revisar parámetros.")
            print(f"Estímulo inválido. Revisar parámetros.")
            return [], []

        lista_tiempo = [us_to_frames(tiempo) for tiempo in lista_tiempo]
        return lista_amplitud, lista_tiempo
    except Exception as e:
        logging.error(f'Error al generar estímulo: {e}')
        print(f'Error al generar estímulo: {e}')
        return [], []

def create_color_palette(df):
    formas_unicas = df['Forma del Pulso'].unique()
    base_colors = sns.color_palette('tab10', n_colors=len(formas_unicas))
    forma_color_dict = dict(zip(formas_unicas, base_colors))
    
    stim_color_dict = {}
    for forma in formas_unicas:
        duraciones = sorted(df[df['Forma del Pulso'] == forma]['Duración (ms)'].unique())
        num_duraciones = len(duraciones)
        shades = sns.light_palette(forma_color_dict[forma], n_colors=num_duraciones, reverse=True, input='rgb')
        for i, duracion in enumerate(duraciones):
            stim_key = f"{forma.capitalize()}_{duracion}ms"
            stim_color_dict[stim_key] = shades[i]
    return stim_color_dict

def plot_trials_side_by_side(
    stimulus_key, 
    data, 
    body_part, 
    dia_experimental, 
    output_dir,
    coord_x=None, 
    coord_y=None, 
    dist_ic=None
):
    """
    Genera la figura de 5 paneles por cada ensayo:
    1) Desplazamiento (px)
    2) Velocidad cruda + Umbral
    3) Rango de Movimientos (threshold, submovs, etc.)
    4) Detalle de Velocidad + Submovimientos (filtrada, modelo min jerk, gauss)
    5) Perfil del Estímulo

    En el 3er panel se marcan:
    - Movimientos threshold (y=1.0) con in/out/pico
    - Submov filtrado (y=0.97)
    - Submov MinJerk (y=0.94)
    - Submov Gauss (y=0.91)

    Además, se marca el pico con '^' en cada caso.
    """
    trials_data = data.get('trials_data', [])
    if not trials_data:
        logging.warning(
            f"No hay datos de ensayos (trials_data) para {stimulus_key} "
            f"en {body_part} día {dia_experimental}"
        )
        return

    max_per_figure = 15
    num_figures = (len(trials_data) // max_per_figure) + (1 if len(trials_data) % max_per_figure != 0 else 0)
    fs = 100.0

    for fig_index in range(num_figures):
        start_idx = fig_index * max_per_figure
        end_idx   = min(start_idx + max_per_figure, len(trials_data))
        subset    = trials_data[start_idx:end_idx]

        max_time = 0
        max_vel  = 0
        for td in subset:
            vel_ensayo = td.get('velocity', [])
            if len(vel_ensayo) > 0:
                max_time = max(max_time, len(vel_ensayo) / fs)
                max_vel  = max(max_vel, np.max(vel_ensayo))

        fig_height = 25
        fig_width  = len(subset) * 5
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

            # ----------------------------------------------------------------
            # Panel 1: Desplazamiento
            # ----------------------------------------------------------------
            if len(pos['x']) == len(pos['y']) and len(pos['x']) > 0:
                displacement = np.sqrt((pos['x'] - pos['x'][0])**2 + (pos['y'] - pos['y'][0])**2)
                t_disp = np.arange(len(displacement)) / fs
                ax_disp.plot(t_disp, displacement, color='blue', label='Desplaz.(px)')
                if len(displacement) > 0:
                    ax_disp.set_ylim(0, np.max(displacement) + 10)
            ax_disp.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')
            ax_disp.set_ylabel('Desplaz. (px)')
            ax_disp.set_xlabel('Tiempo (s)')
            ax_disp.set_xlim(0, max_time)
            ax_disp.set_title(f"Ensayo {trial.get('trial_index', 0) + 1}")
            if idx_col == 0:
                ax_disp.legend(fontsize=8)

            # ----------------------------------------------------------------
            # Panel 2: Vel + Umbral
            # ----------------------------------------------------------------
            ax_vel.plot(t_vel, vel, color='blue', alpha=0.8, label='Velocidad')
            ax_vel.axhline(threshold, color='k', ls='--', label=f'Umbral={threshold:.1f}')
            ax_vel.axhline(mean_vel_pre, color='lightcoral', ls='-', label=f'MeanPre={mean_vel_pre:.1f}')
            ax_vel.fill_between(
                t_vel,
                mean_vel_pre - std_vel_pre,
                mean_vel_pre + std_vel_pre,
                color='lightcoral', alpha=0.2,
                label='±1STD Pre'
            )
            ax_vel.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')
            ax_vel.set_xlabel('Tiempo (s)')
            ax_vel.set_ylabel('Vel. (px/s)')
            ax_vel.set_xlim(0, max_time)
            ax_vel.set_ylim(0, max_vel + 5)
            if idx_col == 0:
                ax_vel.legend(fontsize=7)

            # ----------------------------------------------------------------
            # Panel 3: Rango Movimientos + Submovs
            # ----------------------------------------------------------------
            ax_mov.set_xlabel('Tiempo (s)')
            ax_mov.set_ylabel('Mov. (indicador)')
            ax_mov.set_xlim(0, max_time)
            ax_mov.set_ylim(0.85, 1.05)
            ax_mov.set_yticks([])
            ax_mov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # Movimientos threshold en y=1.00
            for mov in mov_ranges:
                startF = mov['Inicio Movimiento (Frame)']
                endF   = mov['Fin Movimiento (Frame)']
                periodo = mov['Periodo']
                if periodo == 'Pre-Estímulo':
                    colorMov = 'orange'
                elif periodo == 'Durante Estímulo':
                    colorMov = 'red'
                else:
                    colorMov = 'gray'
                ax_mov.hlines(
                    y=1.00,
                    xmin=startF/fs,
                    xmax=endF/fs,
                    color=colorMov,
                    linewidth=2
                )
                ax_mov.plot(startF/fs, 1.00, marker='o', color=colorMov)
                ax_mov.plot(endF/fs,   1.00, marker='o', color=colorMov)

                lat_pico = mov.get('Latencia al Pico (s)', np.nan)
                if not np.isnan(lat_pico):
                    pico_time = (start_frame/fs) + lat_pico
                    ax_mov.plot(pico_time, 1.00, marker='^', color=colorMov)

            # Niveles submov
            y_dict = {'filt': 0.97, 'model': 0.94, 'gauss': 0.91}

            for i_sub, subm in enumerate(submovements):
                c = cm.tab10(i_sub % 10)
                move_info = subm.get('movement_info', {})
                startF_sub = move_info.get('Inicio Movimiento (Frame)', None)
                endF_sub   = move_info.get('Fin Movimiento (Frame)', None)

                peakF_sub = None
                lat_pico_sub = move_info.get('Latencia al Pico (s)', np.nan)
                if not np.isnan(lat_pico_sub):
                    peakF_sub = start_frame + int(lat_pico_sub*100)

                # 1) Submov filtrada
                if 't_segment_data' in subm and 'v_segment_filt' in subm:
                    y_filt = y_dict['filt']
                    if (startF_sub is not None) and (endF_sub is not None):
                        ax_mov.hlines(y=y_filt, 
                                      xmin=startF_sub/fs, 
                                      xmax=endF_sub/fs, 
                                      color=c, linewidth=3)
                        ax_mov.plot(startF_sub/fs, y_filt, marker='o', color=c)
                        ax_mov.plot(endF_sub/fs,   y_filt, marker='o', color=c)
                        if peakF_sub is not None:
                            ax_mov.plot(peakF_sub/fs, y_filt, marker='^', color=c)

                # 2) Submov (MinJerk)
                if 't_segment_model' in subm and 'v_sm' in subm:
                    y_model = y_dict['model']
                    T_i = subm.get('T', 0.0)
                    t0_i= subm.get('t0', np.nan)
                    if not np.isnan(t0_i) and T_i>0:
                        left_model  = t0_i
                        right_model = t0_i + T_i
                        ax_mov.hlines(y=y_model, 
                                      xmin=left_model, 
                                      xmax=right_model, 
                                      color=c, linewidth=3)
                        ax_mov.plot(left_model,  y_model, marker='o', color=c)
                        ax_mov.plot(right_model, y_model, marker='o', color=c)
                        peak_model = t0_i + 0.4*T_i
                        ax_mov.plot(peak_model, y_model, marker='^', color=c)
                    else:
                        # fallback
                        if (startF_sub is not None) and (endF_sub is not None):
                            ax_mov.hlines(y=y_model,
                                          xmin=startF_sub/fs,
                                          xmax=endF_sub/fs,
                                          color=c, linewidth=3)
                            ax_mov.plot(startF_sub/fs, y_model, marker='o', color=c)
                            ax_mov.plot(endF_sub/fs,   y_model, marker='o', color=c)

                # 3) Submov (Gauss)
                '''
                gauss_params = subm.get('gaussian_params_segment', {})
                A_g   = gauss_params.get('A_gauss', np.nan)
                mu_g  = gauss_params.get('mu_gauss', np.nan)
                sig_g = gauss_params.get('sigma_gauss', np.nan)
                if not np.isnan(A_g) and not np.isnan(mu_g) and not np.isnan(sig_g) and sig_g>0:
                    y_gauss = y_dict['gauss']
                    left_g  = mu_g - 3*sig_g
                    right_g = mu_g + 3*sig_g
                    if left_g < 0: left_g = 0
                    if right_g > max_time: right_g = max_time
                    ax_mov.hlines(y=y_gauss, xmin=left_g, xmax=right_g, color=c, linewidth=3)
                    ax_mov.plot(left_g,  y_gauss, marker='o', color=c)
                    ax_mov.plot(right_g, y_gauss, marker='o', color=c)
                    # pico gauss
                    ax_mov.plot(mu_g, y_gauss, marker='^', color=c)

                '''
                
            # ----------------------------------------------------------------
            # Panel 4: Vel + submov detallado
            # ----------------------------------------------------------------
            ax_submov.plot(t_vel, vel, color='darkorange', label='Vel. (px/s)')
            ax_submov.axhline(threshold, color='k', ls='--', label='Umbral')
            ax_submov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')
            ax_submov.set_xlabel('Tiempo (s)')
            ax_submov.set_ylabel('Vel. (px/s)')
            ax_submov.set_xlim(0, max_time)
            ax_submov.set_ylim(0, max_vel + 5)

            # Leyenda para submovs
            lines_for_legend = []
            labels_for_legend= []

            for i, subm in enumerate(submovements):
                c = cm.tab10(i % 10)

                # submov_filt
                if 't_segment_data' in subm and 'v_segment_filt' in subm:
                    t_seg_data = subm['t_segment_data']
                    v_seg_filt = subm['v_segment_filt']
                    line_filt, = ax_submov.plot(
                        t_seg_data, v_seg_filt, 
                        color=c, marker='o', linestyle='none', 
                        markersize=3, alpha=0.8
                    )
                    if i == 0:  
                        lines_for_legend.append(line_filt)
                        labels_for_legend.append("Submov (filt)")

                # submov_model
                if 't_segment_model' in subm and 'v_sm' in subm:
                    t_seg_model = subm['t_segment_model']
                    v_sm        = subm['v_sm']
                    line_model, = ax_submov.plot(
                        t_seg_model, v_sm, 
                        '--', color=c, alpha=0.9
                    )
                    if i == 0:
                        lines_for_legend.append(line_model)
                        labels_for_legend.append("Submov (MinJerk)")

                # Gauss

                '''
                gauss_params = subm.get('gaussian_params_segment', {})
                A_g   = gauss_params.get('A_gauss', np.nan)
                mu_g  = gauss_params.get('mu_gauss', np.nan)
                sig_g = gauss_params.get('sigma_gauss', 0.001)
                if not np.isnan(A_g) and sig_g>0:
                    left_g  = mu_g - 3*sig_g
                    right_g = mu_g + 3*sig_g
                    if left_g < 0: left_g = 0
                    if right_g > max_time: right_g = max_time
                    t_gauss = np.linspace(left_g, right_g, 200)
                    gauss_curve = A_g * np.exp(-((t_gauss - mu_g)**2) / (2 * (sig_g**2)))
                    line_gauss, = ax_submov.plot(
                        t_gauss, gauss_curve, color=c, linestyle=':', alpha=0.7
                    )
                    if i == 0:
                        lines_for_legend.append(line_gauss)
                        labels_for_legend.append("Submov (Gauss)")

                '''
                
            ax_submov.legend(lines_for_legend, labels_for_legend, fontsize=8, loc='upper right')

            # ----------------------------------------------------------------
            # Panel 5: Perfil del Estímulo
            # ----------------------------------------------------------------
            ax_stim.set_xlabel('Tiempo (s)')
            ax_stim.set_ylabel('Amplitud (µA)')
            ax_stim.set_xlim(0, max_time)
            ax_stim.set_title('Perfil del estímulo')
            ax_stim.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)
            ax_stim.axhline(0, color='black', lw=0.5)

            if amplitude_list and duration_list and len(amplitude_list) == len(duration_list):
                x_vals = [stim_start_s]
                y_vals = [0]
                t_stim = stim_start_s
                for amp, dur in zip(amplitude_list, duration_list):
                    nxt_time = t_stim + (dur / fs)
                    x_vals.extend([t_stim, nxt_time])
                    y_vals.extend([amp, amp])
                    t_stim = nxt_time
                ax_stim.step(x_vals, y_vals, color='purple', where='pre', linewidth=1, label='Estímulo')

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




print("Convirtiendo 'Distancia Intracortical' a punto decimal...")
stimuli_info['Distancia Intracortical'] = stimuli_info['Distancia Intracortical'].astype(str).str.replace(',', '.')
stimuli_info['Distancia Intracortical'] = pd.to_numeric(stimuli_info['Distancia Intracortical'], errors='coerce')
print("Conversión de Distancia Intracortical completada.")

grouped_data = stimuli_info.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical'])

submovement_summary_all = []

def collect_velocity_threshold_data():
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")

    all_movement_data = []
    thresholds_data = []
    processed_combinations = set()
    movement_ranges_all = []

    for (dia_experimental, coord_x, coord_y, dist_ic), day_df in grouped_data:
        print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        print(f'Número de ensayos en este grupo: {len(day_df)}')
        logging.info(f'Número de ensayos en este grupo: {len(day_df)}')

        # Aseguramos que la forma de pulso sea minúscula
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()
        day_df = day_df.reset_index(drop=True)
        day_df['Order'] = day_df.index + 1

        for part in body_parts:
            logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, '
                         f'Dist={dist_ic}, Articulación {part}')
            print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, '
                  f'Dist={dist_ic}, Articulación {part}')
            processed_combinations.add((dia_experimental, coord_x, coord_y, dist_ic, 'All_Stimuli', part))

            all_stimuli_data = {}
            pre_stim_velocities = []

            # -----------------------------
            # 1) Cálculo de umbral
            # -----------------------------
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
                                    vel_pre_stim = vel[:100]
                                    vel_pre_stim = vel_pre_stim[~np.isnan(vel_pre_stim)]
                                    pre_stim_velocities.extend(vel_pre_stim)

            # Si hay pocos datos, no podemos calcular umbral
            if len(pre_stim_velocities) < 10:
                logging.warning(f'Datos insuficientes para calcular umbral para {part} '
                                f'en Día={dia_experimental}.')
                continue

            mean_vel_pre = np.nanmean(pre_stim_velocities)
            std_vel_pre  = np.nanstd(pre_stim_velocities)
            # Filtramos outliers a ±3std
            vel_list_filtered = [
                v for v in pre_stim_velocities
                if (mean_vel_pre - 3*std_vel_pre) <= v <= (mean_vel_pre + 3*std_vel_pre)
            ]
            if len(vel_list_filtered) < 10:
                logging.warning(f'Datos insuficientes tras filtrar outliers para {part}.')
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

            # -----------------------------
            # 2) Procesar Estímulos
            # -----------------------------
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
                    stim_df = day_df[
                        (day_df['Forma del Pulso'] == forma_pulso) &
                        (day_df['Duración (ms)'] == duracion_ms)
                    ]
                else:
                    stim_df = day_df[day_df['Forma del Pulso'] == forma_pulso]

                if stim_df.empty:
                    continue

                amplitudes = stim_df['Amplitud (microA)'].unique()
                amplitude_movement_counts = {}

                # Recorremos cada amplitud para ver cuántos trials
                # generan movimiento por encima del threshold
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

                                for _, segRow in matching_segment_sorted.iterrows():
                                    nombre_segmento = segRow['NombreArchivo'].replace(
                                        '.mp4',''
                                    ).replace('lateral_','')
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
                                            segments = np.split(
                                                indices_above,
                                                np.where(np.diff(indices_above)!=1)[0]+1
                                            )
                                            for seg_above in segments:
                                                movement_start = seg_above[0]
                                                movement_end   = seg_above[-1]
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

                    # Almacenamos info
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

                if not amplitude_movement_counts:
                    logging.debug(f"No hay amplitudes con mov. para {part}, "
                                  f"día {dia_experimental}, {forma_pulso} {duracion_ms}ms.")
                    continue

                max_proportion = max(
                    mdata['proportion_movement'] for mdata in amplitude_movement_counts.values()
                )
                selected_amplitudes = [
                    amp for amp, mdata in amplitude_movement_counts.items()
                    if mdata['proportion_movement'] == max_proportion
                ]
                selected_trials = stim_df[
                    stim_df['Amplitud (microA)'].isin(selected_amplitudes)
                ]
                print(f"Amplitudes selec. {selected_amplitudes} => prop mov={max_proportion:.2f} "
                      f"para {part}, día={dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                logging.info(f"Amplitudes selec. {selected_amplitudes} con prop mov={max_proportion:.2f}")

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
                    logging.warning("Múltiples frecuencias => usando la primera.")
                else:
                    frequency = None

                movement_trials_in_selected = 0
                trials_passed = []
                group_velocities = []
                group_positions = {'x': [], 'y': []}
                group_trial_indices = []
                trial_counter = 0
                movement_ranges = []
                trials_data = []

                # -------------------------------------------------------
                # Detección de submovimientos Threshold / Gauss / MinJerk
                # -------------------------------------------------------
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

                                        # Chequeamos si al menos hay un frame > threshold
                                        # en [start_frame, current_frame]
                                        trial_passed = np.any(vel[start_frame:current_frame] > threshold)
                                        if trial_passed:
                                            movement_trials_in_selected += 1

                                        group_velocities.append(vel)
                                        group_positions['x'].append(pos['x'])
                                        group_positions['y'].append(pos['y'])
                                        group_trial_indices.append(trial_counter)

                                        frames_vel     = np.arange(len(vel))
                                        above_threshold= (vel > threshold)
                                        indices_above  = frames_vel[above_threshold]

                                        submovements_totales = []
                                        if len(indices_above) > 0:
                                            # Dividimos en tramos contiguos:
                                            segments = np.split(
                                                indices_above,
                                                np.where(np.diff(indices_above)!=1)[0]+1
                                            )
                                            for seg_idx, seg_above in enumerate(segments):
                                                movement_start = seg_above[0]
                                                movement_end   = seg_above[-1]
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

                                                # Datos para movement_ranges
                                                movement_data = {
                                                    'Ensayo': trial_counter+1,
                                                    'Inicio Movimiento (Frame)': movement_start,
                                                    'Fin Movimiento (Frame)':    movement_end,
                                                    'Latencia al Inicio (s)':    onset_lat_sec,
                                                    'Latencia al Pico (s)':      latency2peak,
                                                    'Valor Pico (velocidad)':    peak_velocity,
                                                    'Duración Total (s)':        total_dur_sec,
                                                    'Duración durante Estímulo (s)': max(
                                                        0,
                                                        min(movement_end, current_frame) - 
                                                        max(movement_start,start_frame)
                                                    )/100.0,
                                                    'body_part': part,
                                                    'Dia experimental': dia_experimental,
                                                    'Order': order,
                                                    'Estímulo': f"{forma_pulso.capitalize()}_{duracion_ms}ms",
                                                    'Periodo': periodo,
                                                    'Forma del Pulso': forma_pulso.capitalize(),
                                                    'Duración (ms)': duracion_ms,
                                                    'MovementType': 'Threshold-based'
                                                }
                                                movement_ranges.append(movement_data)
                                                movement_ranges_all.append(movement_data)

                                                # ------------------------------------------------------
                                                # 3) Ajuste MinJerk/Gauss SOLO si periodo == "Durante Estímulo"
                                                # ------------------------------------------------------
                                                if periodo == 'Durante Estímulo':
                                                    t_segment   = np.arange(movement_start, movement_end+1)/100.0
                                                    vel_segment = vel[movement_start:movement_end+1]
                                                    vel_segment_filtrada = aplicar_moving_average(
                                                        vel_segment, window_size=6
                                                    )

                                                    # submovimientos dentro de esa ventana:
                                                    submov_peak_indices = detectar_submovimientos_en_segmento(
                                                        vel_segment_filtrada, threshold
                                                    )
                                                    n_submov = max(1, len(submov_peak_indices))

                                                    # Ajuste robusto:
                                                    result = fit_velocity_profile(
                                                        t_segment, vel_segment, n_submov
                                                    )

                                                    gauss_params_segment = fit_gaussian_submovement(
                                                        t_segment, vel_segment_filtrada, threshold
                                                    )

                                                    if result is not None:
                                                        fitted_params = result.x
                                                        for i_sub in range(n_submov):
                                                            A_i  = fitted_params[3*i_sub]
                                                            t0_i = fitted_params[3*i_sub+1]
                                                            T_i  = fitted_params[3*i_sub+2]

                                                            # Generamos la curva MinJerk
                                                            v_sm = minimum_jerk_velocity(
                                                                t_segment, A_i, t0_i, T_i
                                                            )

                                                            # -----------------------------
                                                            # NUEVO: Corrección de amplitud
                                                            # para alinear el pico teórico
                                                            # con la velocidad real
                                                            # -----------------------------
                                                            peak_time_model = t0_i + 0.4*T_i
                                                            # Buscamos índice + cercano
                                                            idx_peak_model = np.argmin(
                                                                np.abs(t_segment - peak_time_model)
                                                            )
                                                            if 0 <= idx_peak_model < len(v_sm):
                                                                v_model_peak = v_sm[idx_peak_model]
                                                                v_obs_peak   = vel_segment[idx_peak_model]
                                                                if abs(v_model_peak) > 1e-6:
                                                                    scale_factor = v_obs_peak / v_model_peak
                                                                else:
                                                                    scale_factor = 1.0

                                                                # Recalculamos amplitud
                                                                A_i_corrected = A_i * scale_factor
                                                                # Nueva curva con la amplitud corregida
                                                                v_sm_corrected = minimum_jerk_velocity(
                                                                    t_segment, A_i_corrected, t0_i, T_i
                                                                )
                                                                subm_peak_value = v_obs_peak
                                                            else:
                                                                # Si no pudimos alinear
                                                                A_i_corrected = A_i
                                                                v_sm_corrected = v_sm
                                                                subm_peak_value= np.max(v_sm)

                                                            # Ajuste Gauss final sobre la curva corregida
                                                            gauss_params = fit_gaussian_submovement(
                                                                t_segment, v_sm_corrected, threshold
                                                            )

                                                            if gauss_params:
                                                                A_gauss  = gauss_params.get('A_gauss', np.nan)
                                                                mu_gauss = gauss_params.get('mu_gauss', np.nan)
                                                                sig_gauss= gauss_params.get('sigma_gauss', 0.0)

                                                                # Submov 'Gaussian-based'
                                                                gauss_submov_data = {
                                                                    'Ensayo':           trial_counter+1,
                                                                    'body_part':        part,
                                                                    'Dia experimental': dia_experimental,
                                                                    'Order':            order,
                                                                    'Estímulo':         f"{forma_pulso.capitalize()}_{duracion_ms}ms",
                                                                    'Forma del Pulso':  forma_pulso.capitalize(),
                                                                    'Duración (ms)':    duracion_ms,
                                                                    'Periodo':          'Durante Estímulo',
                                                                    'MovementType':     'Gaussian-based',
                                                                    'Latencia al Inicio (s)': (t0_i - start_frame/100.0),
                                                                    'Latencia al Pico (s)':   (mu_gauss - (start_frame/100.0)),
                                                                    'Valor Pico (velocidad)': A_gauss,
                                                                    'Duración Total (s)':     (6*sig_gauss),
                                                                    'Duración durante Estímulo (s)': (6*sig_gauss)
                                                                }
                                                                movement_ranges_all.append(gauss_submov_data)

                                                            # Guardamos el submov MinJerk con amplitud corregida
                                                            subm_dict = {
                                                                'A_original':    A_i,
                                                                'A_corrected':   A_i_corrected,
                                                                't0':            t0_i,
                                                                'T':             T_i,
                                                                'latencia_inicio': (t0_i - start_frame/100.0),
                                                                'latencia_pico':   (peak_time_model - (start_frame/100.0)),
                                                                'valor_pico':      subm_peak_value,

                                                                't_segment_data':   t_segment,
                                                                'v_segment_data':   vel_segment,
                                                                'v_segment_filt':   vel_segment_filtrada,
                                                                't_segment_model':  t_segment,
                                                                'v_sm':             v_sm_corrected,  # la curva final
                                                                'gaussian_params_segment': gauss_params_segment,
                                                                'gaussian_params':         gauss_params,

                                                                'movement_info': {
                                                                    'Inicio Movimiento (Frame)': movement_start,
                                                                    'Fin Movimiento (Frame)':    movement_end,
                                                                    'Latencia al Pico (s)':      (peak_time_model - (start_frame/100.0))
                                                                }
                                                            }
                                                            submovements_totales.append(subm_dict)

                                        # Guardamos la info del trial para gráficas
                                        trial_data = {
                                            'velocity': vel,
                                            'positions': pos,
                                            'trial_index': trial_counter,
                                            'submovements': submovements_totales,
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

                # Si no hubo trials con datos válidos
                if len(trials_data) == 0:
                    logging.debug(f"No hay datos de veloc. para {part} en día {dia_experimental}, "
                                  f"{forma_pulso} {duracion_ms} ms.")
                    print(f"No hay datos de veloc. para {part} en día {dia_experimental}, "
                          f"{forma_pulso} {duracion_ms} ms.")
                    continue

                # Guardamos la info para cada estímulo
                all_stimuli_data[stimulus_key] = {
                    'velocities':      group_velocities,
                    'positions':       group_positions,
                    'threshold':       threshold,
                    'amplitude_list':  amplitude_list,
                    'duration_list':   duration_list,
                    'start_frame':     start_frame,
                    'current_frame':   current_frame,
                    'mean_vel_pre':    mean_vel_pre,
                    'std_vel_pre':     std_vel_pre,
                    'amplitud_real':   selected_amplitudes,
                    'y_max_velocity':  y_max_velocity,
                    'trial_indices':   group_trial_indices,
                    'form':            forma_pulso.capitalize(),
                    'duration_ms':     duracion_ms,
                    'frequency':       frequency,
                    'movement_ranges': movement_ranges,
                    'movement_trials': movement_trials_in_selected,
                    'total_trials':    len(trials_data),
                    'trials_passed':   trials_passed,
                    'Order':           order,
                    'trials_data':     trials_data
                }

            # Graficamos cada estímulo
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

    # Al concluir todos los días/articulaciones, guardamos los CSV
    counts_df = pd.DataFrame(all_movement_data)
    counts_path = os.path.join(output_comparisons_dir, 'movement_counts_summary.csv')
    counts_df.to_csv(counts_path, index=False)
    print(f"Datos de movimiento guardados en: {counts_path}")

    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_path = os.path.join(output_comparisons_dir, 'thresholds_summary.csv')
    thresholds_df.to_csv(thresholds_path, index=False)
    print(f"Datos de umbrales guardados en: {thresholds_path}")

    movement_ranges_df = pd.DataFrame(movement_ranges_all)
    movement_ranges_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    movement_ranges_df.to_csv(movement_ranges_path, index=False)
    print(f"Datos de movement_ranges guardados en: {movement_ranges_path}")

    # (Opcional) si quieres submovement_summary_all
    if submovement_summary_all:
        submovement_df = pd.DataFrame(submovement_summary_all)
        submovement_path = os.path.join(output_comparisons_dir, 'submovement_summary.csv')
        submovement_df.to_csv(submovement_path, index=False)
        print(f"Datos de submovimientos guardados en: {submovement_path}")

    # Finalmente, hacemos los boxplots de resumen
    plot_summary_movement_data(movement_ranges_df)

    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        print(f"Día: {combo[0]}, X={combo[1]}, Y={combo[2]}, Dist={combo[3]}, {combo[4]}, {combo[5]}")

    print("Finalizada la recopilación de datos de umbral de velocidad.")
    return counts_df




def plot_summary_movement_data(movement_ranges_df):
    """
    Genera boxplots (resumen general) de métricas como:
      - Latencia al Inicio
      - Latencia al Pico
      - Duración Total
      - Velocidad Pico
      - Número de Submovimientos (N_Mov)
    Agrupando por:
      - Movimiento Durante Estímulo
      - Forma del Pulso
      - Duración
      - MovementType (Threshold-based, Gaussian-based, TercerModelo, etc.)
    """

    logging.info("Generando gráficos comparativos de movimientos (Threshold/Gaussian/TercerModelo) durante el estímulo.")
    print("Generando gráficos comparativos de movimientos (Threshold/Gaussian/TercerModelo) durante el estímulo.")

    # Filtramos sólo DURANTE estímulo
    df_durante = movement_ranges_df[movement_ranges_df['Periodo'] == 'Durante Estímulo'].copy()
    if df_durante.empty:
        logging.info("No hay movimientos DURANTE estímulo para graficar.")
        print("No hay movimientos DURANTE estímulo para graficar.")
        return

    # Convertimos tiempos a ms
    df_durante['Latencia al Inicio (ms)'] = df_durante['Latencia al Inicio (s)'] * 1000
    df_durante['Latencia al Pico (ms)']   = df_durante['Latencia al Pico (s)']   * 1000
    df_durante['Duración Total (ms)']     = df_durante['Duración Total (s)']     * 1000

    # Normalizar la forma a capitalizar
    df_durante['Forma del Pulso'] = df_durante['Forma del Pulso'].str.capitalize()

    # Asegurar que la Duración (ms) sea int, si aplica
    df_durante['Duración (ms)']   = df_durante['Duración (ms)'].astype(int, errors='ignore')

    # Mapeo de Formas y duraciones
    pulse_duration_dict = {
        'Rectangular': [500, 750, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa ascendente': [1000],
        'Rampa descendente': [1000],
        'Triple rombo': [700]
    }
    pulse_shapes = list(pulse_duration_dict.keys())
    base_colors  = sns.color_palette('tab10', n_colors=len(pulse_shapes))
    pulse_shape_colors = dict(zip(pulse_shapes, base_colors))

    # MovementTypes
    movement_types = ['Threshold-based', 'Gaussian-based', 'ThirdModel-based']

    def shade_color(base_rgb, movement_type):
        r,g,b = base_rgb
        if movement_type == 'Gaussian-based':
            factor = 0.7
        elif movement_type == 'ThirdModel-based':
            factor = 0.5
        else:
            factor = 1.0
        return (min(r*factor,1.0), min(g*factor,1.0), min(b*factor,1.0))

    grouping_cols = [
        'Dia experimental', 'body_part', 'Ensayo',
        'MovementType', 'Forma del Pulso', 'Duración (ms)'
    ]
    grouped = df_durante.groupby(grouping_cols, dropna=False).agg({
        'Latencia al Inicio (ms)': 'mean',
        'Latencia al Pico (ms)':   'mean',
        'Valor Pico (velocidad)':  'mean',
        'Duración Total (ms)':     'sum',
        'Ensayo': 'count'
    }).rename(columns={'Ensayo': 'N_Mov'}).reset_index()

    # N_Mov => número de submovimientos detectados
    grouped['Numero Mov/Sub'] = grouped['N_Mov'].astype(int)

    measurements = [
        'Latencia al Inicio (ms)',
        'Latencia al Pico (ms)',
        'Duración Total (ms)',
        'Valor Pico (velocidad)',
        'Numero Mov/Sub'
    ]

    unique_days = grouped['Dia experimental'].unique()

    for day in unique_days:
        df_day = grouped[grouped['Dia experimental'] == day].copy()
        if df_day.empty:
            continue

        fig, axs = plt.subplots(1, len(measurements), figsize=(7*len(measurements), 8), sharey=False)
        for idx, measurement in enumerate(measurements):
            ax = axs[idx] if len(measurements) > 1 else axs

            boxplot_data = []
            x_positions  = []
            x_labels     = []
            x_label_positions = []
            box_colors   = []
            current_pos  = 0
            width        = 0.6
            gap_dur      = 0.4
            gap_shape    = 1.5
            shape_positions = []

            for shape in pulse_shapes:
                df_shape = df_day[df_day['Forma del Pulso'] == shape]
                durations = pulse_duration_dict.get(shape, [])
                if df_shape.empty or not durations:
                    continue

                positions = np.arange(
                    current_pos,
                    current_pos + len(durations)*(width+gap_dur),
                    (width+gap_dur)
                )
                for i, dur in enumerate(durations):
                    base_x = positions[i]
                    for mtype in movement_types:
                        sub_df = df_shape[
                            (df_shape['Duración (ms)'] == dur) &
                            (df_shape['MovementType'] == mtype)
                        ]
                        data_measure = sub_df[measurement].dropna()
                        offset = (width/2.0 if mtype=='Gaussian-based' else 0.0)
                        if mtype=='ThirdModel-based':
                            offset = width*0.75

                        x_pos = base_x + offset

                        boxplot_data.append(data_measure)
                        x_positions.append(x_pos)
                        x_labels.append(f"{dur} ms")
                        x_label_positions.append(x_pos)

                        base_rgb = pulse_shape_colors[shape]
                        final_rgb= shade_color(base_rgb, mtype)
                        box_colors.append(final_rgb)

                if len(positions) > 0:
                    shape_positions.append((positions.mean(), shape))
                current_pos = positions[-1] + gap_shape

            if not boxplot_data:
                ax.axis('off')
                ax.text(0.5,0.5,'No data',ha='center',va='center',fontsize=12)
                continue

            bp = ax.boxplot(
                boxplot_data,
                positions=x_positions,
                widths=width/2.2,
                patch_artist=True,
                showfliers=False,
                whis=(0,100)
            )
            for part_name in ('whiskers', 'caps'):
                for line in bp[part_name]:
                    line.set_visible(False)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_edgecolor('black')
                patch.set_facecolor(color)

            ax.set_xticks(x_label_positions)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlabel('Duración (ms)')
            ax.set_ylabel(measurement)
            ax.set_title(measurement)

            y_lims = ax.get_ylim()
            for x_ctr, shape_name in shape_positions:
                wrapped_text = '\n'.join(textwrap.wrap(shape_name, width=10))
                ax.text(x_ctr, y_lims[1]+(y_lims[1]-y_lims[0])*0.05,
                        wrapped_text, ha='center',va='bottom',fontsize=10)
            ax.set_ylim(y_lims[0], y_lims[1]+(y_lims[1]-y_lims[0])*0.15)

        shape_legend = [Patch(facecolor=pulse_shape_colors[ps], label=ps) for ps in pulse_shapes]
        legend_expl  = Patch(facecolor='white', edgecolor='black', 
                             label='Threshold => color\nGaussian => +Oscuro\nThird => +MásOscuro')
        axs[-1].legend(
            handles=[legend_expl]+shape_legend,
            loc='upper right',
            title='Leyenda (Pulso vs. MovType)',
            fontsize=9
        )

        fig.suptitle(f'Resumen General (Día {day})', fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.90])
        fname = f'summary_general_dia_{sanitize_filename(str(day))}.png'
        out_path = os.path.join(output_comparisons_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Plot] Resumen General Día {day} guardado en {out_path}")



def plot_summary_movement_data_by_bodypart(movement_ranges_df, output_dir):
    """
    Genera boxplots (resumen por body_part) de métricas (Latencia al Inicio, 
    Latencia al Pico, Duración Total y Velocidad Pico) para cada body_part y 
    día experimental, desglosadas por Forma del Pulso y Duración, siguiendo el 
    mismo estilo que el resumen general.
    """
    logging.info("Generando gráficos DESGLOSADOS por body_part y día.")
    print("Generando gráficos DESGLOSADOS por body_part y día.")

    # Filtramos solo los datos durante el estímulo.
    df_durante = movement_ranges_df[movement_ranges_df['Periodo'] == 'Durante Estímulo'].copy()
    if df_durante.empty:
        logging.info("No hay movimientos DURANTE estímulo para graficar (por body_part).")
        return

    # Convertimos las métricas de segundos a milisegundos.
    df_durante['Latencia al Inicio (ms)'] = df_durante['Latencia al Inicio (s)'] * 1000
    df_durante['Latencia al Pico (ms)']   = df_durante['Latencia al Pico (s)']   * 1000
    df_durante['Duración Total (ms)']     = df_durante['Duración Total (s)']     * 1000

    # Normalizamos la forma del pulso (capitalizando)
    df_durante['Forma del Pulso'] = df_durante['Forma del Pulso'].str.capitalize()
    # Aseguramos que Duración (ms) sea un entero
    df_durante['Duración (ms)'] = df_durante['Duración (ms)'].astype(int, errors='ignore')

    # --- Parámetros para graficar (idénticos al resumen general) ---
    pulse_duration_dict = {
        'Rectangular': [500, 750, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa ascendente': [1000],
        'Rampa descendente': [1000],
        'Triple rombo': [700]
    }
    pulse_shapes = list(pulse_duration_dict.keys())
    base_colors = sns.color_palette('tab10', n_colors=len(pulse_shapes))
    pulse_shape_colors = dict(zip(pulse_shapes, base_colors))
    movement_types = ['Threshold-based', 'Gaussian-based', 'ThirdModel-based']

    def shade_color(base_rgb, movement_type):
        r, g, b = base_rgb
        if movement_type == 'Gaussian-based':
            factor = 0.7
        elif movement_type == 'ThirdModel-based':
            factor = 0.5
        else:
            factor = 1.0
        return (min(r*factor, 1.0), min(g*factor, 1.0), min(b*factor, 1.0))

    # Obtenemos los días y las articulaciones presentes
    dias = df_durante['Dia experimental'].unique()
    bodyparts = df_durante['body_part'].unique()

    # Por cada día y por cada body_part se crea un resumen al estilo general.
    for day in dias:
        for bp in bodyparts:
            df_bp = df_durante[(df_durante['Dia experimental'] == day) & (df_durante['body_part'] == bp)]
            if df_bp.empty:
                continue

            # Seleccionamos las métricas a graficar (puedes ajustarlas)
            measurements = ['Latencia al Inicio (ms)', 'Latencia al Pico (ms)', 'Duración Total (ms)', 'Valor Pico (velocidad)']

            fig, axs = plt.subplots(1, len(measurements), figsize=(7*len(measurements), 8), sharey=False)
            if len(measurements) == 1:
                axs = [axs]

            # Para cada métrica se agrupa por "Forma del Pulso" y "Duración (ms)" 
            # desglosando además por MovementType, de manera similar al resumen general.
            for idx, measurement in enumerate(measurements):
                ax = axs[idx]

                boxplot_data = []
                x_positions = []
                x_labels = []
                x_label_positions = []
                box_colors = []
                current_pos = 0
                width = 0.6
                gap_dur = 0.4
                gap_shape = 1.5
                shape_positions = []

                # Se recorre cada forma de pulso
                for shape in pulse_shapes:
                    df_shape = df_bp[df_bp['Forma del Pulso'] == shape]
                    durations = pulse_duration_dict.get(shape, [])
                    if df_shape.empty or not durations:
                        continue

                    positions = np.arange(current_pos, current_pos + len(durations)*(width+gap_dur), (width+gap_dur))
                    for i, dur in enumerate(durations):
                        base_x = positions[i]
                        # Se recorre cada MovementType
                        for mtype in movement_types:
                            sub_df = df_shape[(df_shape['Duración (ms)'] == dur) & (df_shape['MovementType'] == mtype)]
                            data_measure = sub_df[measurement].dropna()
                            offset = width/2.0 if mtype == 'Gaussian-based' else 0.0
                            if mtype == 'ThirdModel-based':
                                offset = width * 0.75
                            x_pos = base_x + offset
                            boxplot_data.append(data_measure)
                            x_positions.append(x_pos)
                            x_labels.append(f"{dur} ms")
                            x_label_positions.append(x_pos)
                            base_rgb = pulse_shape_colors[shape]
                            final_rgb = shade_color(base_rgb, mtype)
                            box_colors.append(final_rgb)
                    if len(positions) > 0:
                        shape_positions.append((positions.mean(), shape))
                    current_pos = positions[-1] + gap_shape

                if not boxplot_data:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                    continue

                bp_plot = ax.boxplot(
                    boxplot_data,
                    positions=x_positions,
                    widths=width/2.2,
                    patch_artist=True,
                    showfliers=False,
                    whis=(0,100)
                )
                for part_name in ('whiskers', 'caps'):
                    for line in bp_plot[part_name]:
                        line.set_visible(False)
                for median in bp_plot['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                for patch, color in zip(bp_plot['boxes'], box_colors):
                    patch.set_edgecolor('black')
                    patch.set_facecolor(color)
                ax.set_xticks(x_label_positions)
                ax.set_xticklabels(x_labels, rotation=45)
                ax.set_xlabel('Duración (ms)')
                ax.set_ylabel(measurement)
                ax.set_title(measurement)
                y_lims = ax.get_ylim()
                for x_ctr, shape_name in shape_positions:
                    wrapped_text = '\n'.join(textwrap.wrap(shape_name, width=10))
                    ax.text(x_ctr, y_lims[1] + (y_lims[1]-y_lims[0])*0.05,
                            wrapped_text, ha='center', va='bottom', fontsize=10)
                ax.set_ylim(y_lims[0], y_lims[1] + (y_lims[1]-y_lims[0])*0.15)

            # Agregamos la leyenda, igual que en el resumen general.
            shape_legend = [Patch(facecolor=pulse_shape_colors[ps], label=ps) for ps in pulse_shapes]
            legend_expl = Patch(facecolor='white', edgecolor='black', 
                                label='Threshold => color\nGaussian => +Oscuro\nThird => +MásOscuro')
            axs[-1].legend(handles=[legend_expl] + shape_legend, loc='upper right', 
                           title='Leyenda (Pulso vs. MovType)', fontsize=9)

            fig.suptitle(f"{bp} - Día {day} (desglose Forma vs. Duración)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            fname = f"summary_by_bp_{bp}_{sanitize_filename(str(day))}.png"
            out_path = os.path.join(output_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[Plot] Boxplot Interacción Forma*Duración para {bp} en día {day} guardado en {out_path}")






def analyze_best_bodyparts_and_stimuli(counts_df):
    logging.info("Iniciando análisis de las mejores articulaciones y estímulos.")
    print("Iniciando análisis de las mejores articulaciones y estímulos.")

    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'
    counts_df['Estímulo_label'] = counts_df['Forma del Pulso'].str.capitalize() + '\n' + counts_df['Duración (ms)'].astype(str) + ' ms'

    sorted_df = counts_df.sort_values(by='proportion_movement', ascending=False)

    top_bodyparts = sorted_df.groupby('body_part')['proportion_movement'].mean().sort_values(ascending=False)
    logging.info("Top articulaciones con mayor proporción de movimiento:")
    logging.info(top_bodyparts.head(5))
    print("Top articulaciones con mayor proporción de movimiento:")
    print(top_bodyparts.head(5))

    top_stimuli = sorted_df.groupby('Estímulo')['proportion_movement'].mean().sort_values(ascending=False)
    logging.info("\nTop estímulos con mayor proporción de movimiento:")
    logging.info(top_stimuli.head(5))
    print("\nTop estímulos con mayor proporción de movimiento:")
    print(top_stimuli.head(5))

    top_bodyparts_path = os.path.join(output_comparisons_dir, 'top_bodyparts.csv')
    top_stimuli_path = os.path.join(output_comparisons_dir, 'top_stimuli.csv')
    top_bodyparts.to_csv(top_bodyparts_path)
    top_stimuli.to_csv(top_stimuli_path)
    logging.info(f"Top articulaciones guardadas en {top_bodyparts_path}")
    logging.info(f"Top estímulos guardados en {top_stimuli_path}")
    print(f"Top articulaciones guardadas en {top_bodyparts_path}")
    print(f"Top estímulos guardadas en {top_stimuli_path}")

def plot_heatmap(counts_df):
    logging.info("Iniciando generación del heatmap.")
    print("Iniciando generación del heatmap.")

    counts_df['Estímulo'] = counts_df['Forma del Pulso'].str.capitalize() + ', ' + counts_df['Duración (ms)'].astype(str) + ' ms'

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

    common_index = pivot_prop.index.union(pivot_movement.index).union(pivot_total.index)
    common_columns = pivot_prop.columns.union(pivot_movement.columns).union(pivot_total.columns)

    pivot_prop = pivot_prop.reindex(index=common_index, columns=common_columns)
    pivot_movement = pivot_movement.reindex(index=common_index, columns=common_columns)
    pivot_total = pivot_total.reindex(index=common_index, columns=common_columns)
    logging.debug("Reindexación completada en pivot tables.")

    annot_matrix = pivot_movement.fillna(0).astype(int).astype(str) + '/' + pivot_total.fillna(0).astype(int).astype(str)

    plt.figure(figsize=(20, 15))
    try:
        sns.heatmap(pivot_prop, annot=annot_matrix, fmt='', cmap='viridis')
        logging.debug("Heatmap generado con éxito.")
    except Exception as e:
        logging.error(f'Error al generar el heatmap: {e}')
        print(f'Error al generar el heatmap: {e}')
        return

    plt.title('Proporción de Movimiento por Articulación, Día y Estímulo')
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

    counts_df['Order'] = counts_df['Order'].astype(int)
    counts_df = counts_df.sort_values(by=['Dia experimental', 'Order'])

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

        stimulus_labels = df_dia.groupby('Order')['Estímulo_label'].first()
        plt.xticks(df_dia['Order'].unique(), stimulus_labels, rotation=45, ha='right')

        output_path = os.path.join(output_comparisons_dir, f'efectividad_dia_{sanitize_filename(str(dia))}.png')
        plt.savefig(output_path)
        logging.info(f"Gráfico de efectividad guardado en {output_path}")
        print(f"Gráfico de efectividad guardado en {output_path}")
        plt.close()

# -------------------------------------------------------------------
# 1) FUNCIÓN NUEVA PARA PRUEBAS DE HIPÓTESIS:
#    Realiza ANOVA (dos factores: Forma del Pulso y Duración) y ejemplo de Friedman
# -------------------------------------------------------------------

# --- NUEVO: Función auxiliar para generar una "matriz de p-values" a partir de resultados Tukey
def build_significance_matrix(tukey_result, factor_levels, alpha=0.05):
    """
    Construye una matriz NxN con p-values de comparaciones Tukey 
    para un factor que tiene N niveles (factor_levels).
    
    tukey_result es un objeto pairwise_tukeyhsd ya calculado.
    factor_levels es la lista/orden de los niveles.
    alpha se puede usar para resaltar celdas < alpha, etc.
    """
    # Inicializar dataframe NxN con las comparaciones
    n = len(factor_levels)
    pval_matrix = pd.DataFrame(np.ones((n, n)), 
                               index=factor_levels, 
                               columns=factor_levels)
    
    # Recorremos las comparaciones en tukey_result.summary()
    # tukey_result._results_table.data es una forma, 
    # o podemos usar tukey_result.reject, etc.
    # Lo más sencillo es convertirlo a DataFrame:
    tk_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                         columns=tukey_result._results_table.data[0])
    # tk_df => "group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"
    # Llenamos la matriz con p-values
    for row in tk_df.itertuples():
        g1 = getattr(row, 'group1')
        g2 = getattr(row, 'group2')
        p_adj = getattr(row, 'p-adj')
        # Colocamos p_adj en [g1,g2] y [g2,g1] => simétrico
        if g1 in pval_matrix.index and g2 in pval_matrix.columns:
            pval_matrix.loc[g1, g2] = p_adj
            pval_matrix.loc[g2, g1] = p_adj
    
    return pval_matrix

def build_significance_matrix_from_arrays(factor_levels, tukey_df):
    """
    Similar a build_significance_matrix, pero usando la DataFrame 
    ya construida con columnas: 'group1','group2','p-adj', etc.
    """
    n = len(factor_levels)
    pval_matrix = pd.DataFrame(np.ones((n, n)), 
                               index=factor_levels, 
                               columns=factor_levels)

    for row in tukey_df.itertuples():
        g1 = getattr(row, 'group1')
        g2 = getattr(row, 'group2')
        p_adj = getattr(row, 'p_adj')
        if g1 in pval_matrix.index and g2 in pval_matrix.columns:
            pval_matrix.loc[g1, g2] = p_adj
            pval_matrix.loc[g2, g1] = p_adj

    return pval_matrix

# --- NUEVO: Función para graficar la matriz de significancia
# --- NUEVO: Función para graficar la matriz de significancia
def plot_significance_heatmap(pval_matrix, metric, factor, output_dir, 
                              alpha=0.05, cmap='Reds'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pval_matrix, annot=True, cmap=cmap,
                cbar_kws={'label': 'p-value'}, 
                vmin=0.0, vmax=1.0)
    plt.title(f'Matriz de Significancia (Tukey) - {factor}\nMétrica: {metric}')
    plt.tight_layout()
    fname = f"posthoc_heatmap_{sanitize_filename(factor)}_{sanitize_filename(metric)}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Heatmap post-hoc guardado en: {out_path}")



# --- NUEVO: Calcula partial eta-squared a partir de la tabla anova_lm (Type II)
def calc_partial_eta_sq(anova_table, factor_row='C(Q(\'Forma del Pulso\'))', resid_row='Residual'):
    """
    Ejemplo básico: partial eta^2 = sum_sq(factor) / ( sum_sq(factor) + sum_sq(residual) )
    Devuelve un float con esa métrica, o np.nan si falla algo.
    """
    try:
        ss_factor = anova_table.loc[factor_row, 'sum_sq']
        ss_resid  = anova_table.loc[resid_row, 'sum_sq']
        eta_sq = ss_factor / (ss_factor + ss_resid)
        return eta_sq
    except:
        return np.nan

# --- NUEVO: Función para dibujar brackets de significancia en un boxplot
# --- Función para agregar brackets de significancia en un boxplot
def add_significance_brackets(ax, pairs, p_values, box_positions,
                              y_offset=0.02, line_height=0.03, font_size=10):
    """
    Dibuja brackets de significancia en un boxplot a partir del diccionario
    de p-values (p_values) y las posiciones de cada nivel (box_positions).
    """
    y_lim = ax.get_ylim()
    h_base = y_lim[1] + (y_lim[1] - y_lim[0]) * 0.05  # posición base un poco por encima
    step = 0
    for (lv1, lv2) in pairs:
        key1 = (lv1, lv2)
        key2 = (lv2, lv1)
        if key1 in p_values:
            pval = p_values[key1]
        elif key2 in p_values:
            pval = p_values[key2]
        else:
            continue
        x1 = box_positions[lv1]
        x2 = box_positions[lv2]
        if x1 > x2:
            x1, x2 = x2, x1
        h = h_base + step * (y_lim[1] - y_lim[0]) * line_height
        step += 1
        ax.plot([x1, x1, x2, x2],
                [h, h+0.001, h+0.001, h],
                lw=1.5, c='k')
        p_text = f"p={pval:.3g}" if pval >= 0.001 else "p<0.001"
        ax.text((x1+x2)*0.5, h+0.001, p_text, ha='center', va='bottom', fontsize=font_size)





# --- NUEVO: Función para correr el post-hoc (Tukey) para un factor dado
# --- Función para correr el post-hoc (Tukey) para un factor dado
def do_posthoc_tests(df, metric, factor_name):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df_clean = df.dropna(subset=[metric, factor_name])
    if df_clean[factor_name].nunique() < 2:
        return None, None

    try:
        tukey_res = pairwise_tukeyhsd(
            endog=df_clean[metric].values,
            groups=df_clean[factor_name].values,
            alpha=0.05
        )
        # Se construye un DataFrame a partir de la tabla de resultados.
        tk_df = pd.DataFrame(data=tukey_res._results_table.data[1:],
                             columns=tukey_res._results_table.data[0])
        # Renombramos "p-adj" a "p_adj" para facilitar el acceso posterior.
        tk_df = tk_df.rename(columns={'p-adj': 'p_adj'})
        factor_levels = sorted(df_clean[factor_name].unique())
        pval_matrix = build_significance_matrix_from_arrays(factor_levels, tk_df)
        return tk_df, pval_matrix

    except Exception as e:
        logging.warning(f"Error en Tukey para {factor_name}, {metric}: {e}")
        return None, None




# --- NUEVO: Función que realiza las pruebas de significancia y guarda resultados y gráficos
def do_significance_tests(movement_ranges_df, output_dir=None):
    """
    Realiza ANOVA de dos factores (Forma del Pulso y Duración (ms)),
    Friedman, y post-hoc Tukey. Además, genera boxplots con líneas de
    significancia y heatmaps de p-values.
    """
    if output_dir is None:
        output_dir = output_comparisons_dir
    
    df = movement_ranges_df.copy()
    df = df[df['Periodo'] == 'Durante Estímulo'].dropna(subset=['Duración (ms)', 'Forma del Pulso'])
    if df.empty:
        print("No hay datos 'Durante Estímulo' para análisis estadístico.")
        return

    # Convertir a milisegundos
    df['Latencia al Inicio (ms)'] = df['Latencia al Inicio (s)'] * 1000
    df['Latencia al Pico (ms)'] = df['Latencia al Pico (s)'] * 1000
    df['Duración Total (ms)'] = df['Duración Total (s)'] * 1000

    # Aseguramos que las columnas clave sean de tipo cadena (para post-hoc)
    df['Forma del Pulso'] = df['Forma del Pulso'].astype(str)
    df['Duración (ms)'] = df['Duración (ms)'].astype(str)

    metrics = ['Latencia al Inicio (ms)', 'Latencia al Pico (ms)',
               'Duración Total (ms)', 'Valor Pico (velocidad)']

    grouping = ['Dia experimental', 'body_part', 'MovementType']

    results_anova = []
    results_friedman = []
    posthoc_results = []    # Para almacenar las tablas de post-hoc
    posthoc_signifs = []    # Para almacenar las matrices de p-values en formato melt

    grouped_stats = df.groupby(grouping)
    for (dia, bp, movtype), df_sub in grouped_stats:
        if len(df_sub) < 3:
            continue

        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue

            formula = f"Q('{metric}') ~ C(Q('Forma del Pulso')) * C(Q('Duración (ms)'))"
            try:
                model = ols(formula, data=data_metric).fit()
                anova_res = anova_lm(model, typ=2)
                for row_idx in anova_res.index:
                    if row_idx == 'Residual':
                        continue
                    eta_sq = calc_partial_eta_sq(anova_res, factor_row=row_idx, resid_row='Residual')
                    results_anova.append({
                        'Dia experimental': dia,
                        'body_part': bp,
                        'MovementType': movtype,
                        'Metric': metric,
                        'Factor': row_idx,
                        'sum_sq': anova_res.loc[row_idx, 'sum_sq'],
                        'df': anova_res.loc[row_idx, 'df'],
                        'F': anova_res.loc[row_idx, 'F'],
                        'PR(>F)': anova_res.loc[row_idx, 'PR(>F)'],
                        'Partial_Eta_Sq': eta_sq,
                        'Num_observations': len(data_metric)
                    })

                # Post-hoc for "Forma del Pulso"
                try:
                    p_forma = anova_res.loc["C(Q('Forma del Pulso'))", "PR(>F)"]
                    if p_forma < 0.05:
                        tk_df, pvals_mat = do_posthoc_tests(data_metric, metric, factor_name='Forma del Pulso')
                        if tk_df is not None:
                            tk_df['Dia experimental'] = dia
                            tk_df['body_part'] = bp
                            tk_df['MovementType'] = movtype
                            tk_df['Metric'] = metric
                            tk_df['Factor'] = 'Forma del Pulso'
                            posthoc_results.append(tk_df)
                            if pvals_mat is not None:
                                plot_significance_heatmap(
                                    pvals_mat,
                                    metric=metric,
                                    factor=f"FormaPulso_{dia}_{bp}_{movtype}",
                                    output_dir=output_dir,
                                    alpha=0.05
                                )
                                pvals_flat = pvals_mat.reset_index().melt(
                                    id_vars=pvals_mat.index.name, var_name='group2', value_name='p-value'
                                )
                                pvals_flat['Dia experimental'] = dia
                                pvals_flat['body_part'] = bp
                                pvals_flat['MovementType'] = movtype
                                pvals_flat['Metric'] = metric
                                pvals_flat['Factor'] = 'Forma del Pulso'
                                posthoc_signifs.append(pvals_flat)
                                plot_boxplot_metric_with_posthoc(
                                    data_metric, metric, 'Forma del Pulso',
                                    tukey_df=tk_df, pvals_matrix=pvals_mat,
                                    output_dir=output_dir
                                )
                except KeyError:
                    pass

                # Post-hoc for "Duración (ms)"
                try:
                    p_duracion = anova_res.loc["C(Q('Duración (ms)'))", "PR(>F)"]
                    if p_duracion < 0.05:
                        tk_df, pvals_mat = do_posthoc_tests(data_metric, metric, factor_name='Duración (ms)')
                        if tk_df is not None:
                            tk_df['Dia experimental'] = dia
                            tk_df['body_part'] = bp
                            tk_df['MovementType'] = movtype
                            tk_df['Metric'] = metric
                            tk_df['Factor'] = 'Duración (ms)'
                            posthoc_results.append(tk_df)
                            if pvals_mat is not None:
                                plot_significance_heatmap(
                                    pvals_mat,
                                    metric=metric,
                                    factor=f"Duracion_{dia}_{bp}_{movtype}",
                                    output_dir=output_dir,
                                    alpha=0.05
                                )
                                pvals_flat = pvals_mat.reset_index().melt(
                                    id_vars=pvals_mat.index.name, var_name='group2', value_name='p-value'
                                )
                                pvals_flat['Dia experimental'] = dia
                                pvals_flat['body_part'] = bp
                                pvals_flat['MovementType'] = movtype
                                pvals_flat['Metric'] = metric
                                pvals_flat['Factor'] = 'Duración (ms)'
                                posthoc_signifs.append(pvals_flat)
                                plot_boxplot_metric_with_posthoc(
                                    data_metric, metric, 'Duración (ms)',
                                    tukey_df=tk_df, pvals_matrix=pvals_mat,
                                    output_dir=output_dir
                                )
                except KeyError:
                    pass

                # Interaction post-hoc (separately by Forma del Pulso)
                try:
                    p_inter = anova_res.loc["C(Q('Forma del Pulso')):C(Q('Duración (ms)'))", "PR(>F)"]
                    if p_inter < 0.05:
                        logging.info(f"Interacción significativa en {dia}, {bp}, {movtype}, {metric} => post-hoc parcial.")
                        for forma_val in data_metric['Forma del Pulso'].unique():
                            subset_forma = data_metric[data_metric['Forma del Pulso'] == forma_val]
                            if subset_forma['Duración (ms)'].nunique() < 2:
                                continue
                            tk_df_sub, pvals_mat_sub = do_posthoc_tests(
                                subset_forma, metric, factor_name='Duración (ms)'
                            )
                            if tk_df_sub is not None:
                                tk_df_sub['Dia experimental'] = dia
                                tk_df_sub['body_part'] = bp
                                tk_df_sub['MovementType'] = movtype
                                tk_df_sub['Metric'] = metric
                                tk_df_sub['Factor'] = f'Inter_Dur(F:{forma_val})'
                                posthoc_results.append(tk_df_sub)
                                if pvals_mat_sub is not None:
                                    plot_significance_heatmap(
                                        pvals_mat_sub,
                                        metric=metric,
                                        factor=f"InterDur_{dia}_{bp}_{movtype}_F_{forma_val}",
                                        output_dir=output_dir,
                                        alpha=0.05
                                    )
                                    pvals_flat_sub = pvals_mat_sub.reset_index().melt(
                                        id_vars=pvals_mat_sub.index.name, var_name='group2', value_name='p-value'
                                    )
                                    pvals_flat_sub['Dia experimental'] = dia
                                    pvals_flat_sub['body_part'] = bp
                                    pvals_flat_sub['MovementType'] = movtype
                                    pvals_flat_sub['Metric'] = metric
                                    pvals_flat_sub['Factor'] = f'Inter_Dur(F:{forma_val})'
                                    posthoc_signifs.append(pvals_flat_sub)
                                    plot_boxplot_metric_with_posthoc(
                                        subset_forma, metric, 'Duración (ms)',
                                        tukey_df=tk_df_sub, pvals_matrix=pvals_mat_sub,
                                        output_dir=output_dir
                                    )
                except KeyError:
                    pass

            except Exception as e:
                logging.warning(f"Fallo ANOVA {dia}, {bp}, {movtype}, {metric}: {e}")
                continue

        # Friedman test (opcional)
        try:
            metric_friedman = 'Latencia al Inicio (ms)'
            df_sub['Condicion'] = df_sub['Forma del Pulso'] + "_" + df_sub['Duración (ms)']
            pivot_f = df_sub.pivot_table(index='Ensayo', columns='Condicion', values=metric_friedman)
            pivot_f = pivot_f.dropna(axis=1, how='all').dropna(axis=0, how='any')
            if pivot_f.shape[1] > 1 and pivot_f.shape[0] > 2:
                stats_result = friedmanchisquare(*[pivot_f[col] for col in pivot_f])
                results_friedman.append({
                    'Dia experimental': dia,
                    'body_part': bp,
                    'MovementType': movtype,
                    'Metric': metric_friedman,
                    'Friedman_statistic': stats_result.statistic,
                    'Friedman_pvalue': stats_result.pvalue,
                    'Num_conditions': pivot_f.shape[1],
                    'Num_subjects': pivot_f.shape[0]
                })
        except Exception as e:
            logging.warning(f"Fallo Friedman {dia}, {bp}, {movtype}: {e}")

    # Guardamos resultados ANOVA, Friedman, post-hoc, etc.
    anova_df = pd.DataFrame(results_anova)
    if not anova_df.empty:
        anova_path = os.path.join(output_dir, 'anova_twofactor_results.csv')
        anova_df.to_csv(anova_path, index=False)
        print(f"Resultados ANOVA guardados en {anova_path}")
    else:
        print("No se generaron resultados ANOVA (pocos datos).")

    friedman_df = pd.DataFrame(results_friedman)
    if not friedman_df.empty:
        friedman_path = os.path.join(output_dir, 'friedman_results.csv')
        friedman_df.to_csv(friedman_path, index=False)
        print(f"Resultados Friedman guardados en {friedman_path}")
    else:
        print("No se generaron resultados Friedman (no hay datos repetidos).")

    if len(posthoc_results) > 0:
        posthoc_all = pd.concat(posthoc_results, ignore_index=True)
        posthoc_path = os.path.join(output_dir, 'posthoc_tukey_results.csv')
        posthoc_all.to_csv(posthoc_path, index=False)
        print(f"Resultados Post-hoc (Tukey) guardados en {posthoc_path}")

    if len(posthoc_signifs) > 0:
        pvals_concat = pd.concat(posthoc_signifs, ignore_index=True)
        pvals_path = os.path.join(output_dir, 'posthoc_significance_matrices.csv')
        pvals_concat.to_csv(pvals_path, index=False)
        print(f"Matrices de p-values guardadas en {pvals_path}")


# --- NUEVO: Función para graficar un boxplot con líneas de significancia basadas en el post-hoc
# --- Función para graficar un boxplot (usando 100% de los datos para calcular la caja)
# pero que muestre únicamente la caja central (IQR) y la mediana,
# junto con los brackets de significancia obtenidos del post-hoc.
def plot_boxplot_metric_with_posthoc(data_metric, metric, factor_name,
                                     tukey_df, pvals_matrix, output_dir,
                                     grouping_cols=['Dia experimental', 'body_part', 'MovementType']):
    import os
    # Convertir el factor a tipo categórico y ordenar sus niveles
    factor_levels = sorted(data_metric[factor_name].unique())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Generamos el boxplot usando todos los datos para el cálculo,
    # pero configuramos para que se muestren únicamente la caja y la mediana.
    bp = ax.boxplot(
        data_metric.groupby(factor_name)[metric].apply(list),
        positions=range(len(factor_levels)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100)  # Calcula bigotes al mínimo y máximo de los datos
    )
    # Ocultamos los bigotes y "caps" para que solo se muestre la caja central y la mediana.
    for part in ('whiskers', 'caps'):
        for line in bp[part]:
            line.set_visible(False)
    # Ajustamos el color de la mediana
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    # Personalizamos el color de las cajas usando un palette (por ejemplo, 'Set3')
    colors = sns.color_palette('Set3', n_colors=len(factor_levels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Configuramos el eje x
    ax.set_xticks(range(len(factor_levels)))
    ax.set_xticklabels(factor_levels, rotation=45)
    ax.set_xlabel(factor_name)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {factor_name} (post-hoc)")
    
    # Creamos un diccionario con las posiciones de cada nivel para dibujar los brackets
    box_positions = {lvl: pos for pos, lvl in enumerate(factor_levels)}
    # Extraemos los p-values del DataFrame de Tukey (la columna se llama 'p_adj')
    p_values_dict = {}
    for row in tukey_df.itertuples():
        g1 = getattr(row, 'group1')
        g2 = getattr(row, 'group2')
        pval = getattr(row, 'p_adj')
        p_values_dict[(g1, g2)] = pval
    # Generamos todos los pares de niveles para los que queremos dibujar los brackets
    pairs = list(itertools.combinations(factor_levels, 2))
    add_significance_brackets(ax, pairs, p_values_dict, box_positions,
                              y_offset=0.02, line_height=0.05, font_size=9)
    
    fname = f"boxplot_posthoc_{sanitize_filename(factor_name)}_{sanitize_filename(metric)}.png"
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Boxplot con post-hoc guardado en: {out_path}")

# --- NUEVO: Función para graficar un resumen global con significancia superpuesta (opcional)
def plot_global_summary_with_significance(movement_ranges_df, output_dir):
    """
    Genera un resumen global final de las métricas (Latencia al Inicio, Latencia al Pico,
    Duración Total, Valor Pico y Número de Submovimientos) en el que se superponen los
    resultados estadísticos (líneas y p-values de las comparaciones significativas) sobre
    los boxplots. Se agrupa, por ejemplo, por 'Estímulo'.
    """
    df = movement_ranges_df[movement_ranges_df['Periodo'] == 'Durante Estímulo'].copy()
    df['Latencia al Inicio (ms)'] = df['Latencia al Inicio (s)'] * 1000
    df['Latencia al Pico (ms)'] = df['Latencia al Pico (s)'] * 1000
    df['Duración Total (ms)'] = df['Duración Total (s)'] * 1000
    df['Forma del Pulso'] = df['Forma del Pulso'].str.capitalize()
    df['Duración (ms)'] = df['Duración (ms)'].astype(int, errors='ignore')
    
    grouped = df.groupby('Estímulo')
    for est, df_est in grouped:
        for metric in ['Latencia al Inicio (ms)', 'Latencia al Pico (ms)', 'Duración Total (ms)', 'Valor Pico (velocidad)']:
            tk_df, pvals_mat = do_posthoc_tests(df_est, metric, factor_name='Forma del Pulso')
            if tk_df is not None and pvals_mat is not None:
                plot_boxplot_metric_with_posthoc(df_est, metric, 'Forma del Pulso', tk_df, pvals_mat, output_dir)
    print("Resumen global con significancia generado.")


# -------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Ejecutando el bloque principal del script.")
    print("Ejecutando el bloque principal del script.")

    counts_df = collect_velocity_threshold_data()
    logging.info(f"Counts DataFrame after collection: {counts_df.shape}")
    print("Counts DataFrame after collection:", counts_df.shape)
    print(counts_df.head())

    analyze_best_bodyparts_and_stimuli(counts_df)
    plot_heatmap(counts_df)
    plot_effectiveness_over_time(counts_df)

    # Pruebas de hipótesis ANOVA + Friedman
    movement_ranges_df_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    if not os.path.exists(movement_ranges_df_path):
        print(f"No se encontró {movement_ranges_df_path}. No se pueden hacer pruebas de hipótesis.")
        sys.exit()

    # Llamada adicional
    

    movement_ranges_df = pd.read_csv(movement_ranges_df_path)

    plot_summary_movement_data_by_bodypart(movement_ranges_df, output_comparisons_dir)

    do_significance_tests(movement_ranges_df, output_dir=output_comparisons_dir)
    # Y en el bloque principal, después de hacer todos los resúmenes parciales, llamas:
    plot_global_summary_with_significance(movement_ranges_df, output_comparisons_dir)

    print("Proceso completo finalizado.")