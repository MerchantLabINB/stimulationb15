import os 
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from math import sqrt
import matplotlib
matplotlib.use('Agg')  

import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import seaborn as sns
from matplotlib.patches import Patch

VELOCITY_COLOR = "#000080"  # azul marino
GAUSSIAN_PALETTE = sns.husl_palette(l=.4)

from scipy.signal import savgol_filter, find_peaks
import re
import shutil
import glob  # Importar el módulo glob

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import butter, filtfilt

import textwrap
import math

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
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

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
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information_expanded.csv'

csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\xv_lat-Br-2024-10-02\videos'
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\ht_24newthresholds'

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
# verificar_archivo(segmented_info_path, 'informacion_archivos_segmentados.csv')

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
# Definir la paleta de colores global para las formas del pulso
PULSE_SHAPE_COLORS = {
    'Rectangular': sns.color_palette('tab10')[0],
    'Rombo': sns.color_palette('tab10')[1],
    'Rampa ascendente': sns.color_palette('tab10')[2],
    'Rampa descendente': sns.color_palette('tab10')[3],
    'Triple rombo': sns.color_palette('tab10')[4]
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


def fit_gaussians_submovement(t_segment, v_segment, threshold, stim_start):
    """
    Ajusta gaussianas al segmento. Se descartan aquellas cuyo inicio (mu - 2*sigma)
    ocurra antes de stim_start + 0.03 (30 ms después del inicio del estímulo) o cuya
    duración (4*sigma) exceda 0.8 s.
    """
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(v_segment, height=threshold)
    if len(peaks) == 0:
        logging.warning("No se detectaron picos en el segmento para ajustar gaussianas.")
        return []

    gaussians = []
    max_sigma = 0.5 * (t_segment[-1] - t_segment[0])
    for peak in peaks:
        window_size = max(3, int(len(t_segment) * 0.1))
        start_idx = max(0, peak - window_size)
        end_idx = min(len(t_segment), peak + window_size)
        t_win = t_segment[start_idx:end_idx]
        v_win = v_segment[start_idx:end_idx]
        if len(t_win) < 3:
            continue

        A_init = v_segment[peak]
        mu_init = t_segment[peak]
        sigma_init = (t_win[-1] - t_win[0]) / 2.0 if len(t_win) > 1 else 0.01

        # Condición: el inicio de la gaussiana (mu - 3*sigma) debe ser al menos 100 ms después del inicio del estímulo
        if (mu_init - 2 * sigma_init) < (stim_start + 0.03):
            logging.info(f"Gaussiana descartada: inicio {mu_init - 2*sigma_init:.3f}s < {stim_start + 0.1:.3f}s")
            continue

        if 4 * sigma_init > 0.8:
            logging.info(f"Gaussiana descartada: duración {4*sigma_init:.3f}s > 0.8s")
            continue

        p0 = [A_init, mu_init, sigma_init]
        lb = [0, t_win[0], 1e-4]
        ub = [max(v_win) * 5, t_win[-1], max_sigma]
        try:
            popt, _ = curve_fit(gaussian, t_win, v_win, p0=p0, bounds=(lb, ub))
            modelo_pico = gaussian(mu_init, *popt)
            epsilon = 1e-6  # Add a small epsilon to avoid division by zero
            scale = v_segment[peak] / (modelo_pico + epsilon)
            popt[0] *= scale

            # Reaplicar las condiciones con el sigma ajustado
            if (popt[1] - 2 * popt[2]) < (stim_start + 0.03):
                logging.info(f"Gaussiana descartada tras ajuste: inicio {popt[1]-2*popt[2]:.3f}s < {stim_start+0.1:.3f}s")
                continue
            if 4 * popt[2] > 0.8:
                logging.info(f"Gaussiana descartada tras ajuste: duración {4*popt[2]:.3f}s > 0.8s")
                continue

            gaussians.append({'A_gauss': popt[0], 'mu_gauss': popt[1], 'sigma_gauss': popt[2]})
        except Exception as e:
            logging.warning(f"Fallo en curve_fit para pico en índice {peak}: {e}. Usando aproximación simple.")
            sigma_simple = sigma_init if sigma_init <= max_sigma else max_sigma
            if (mu_init - 2 * sigma_simple) < (stim_start + 0.03):
                logging.info(f"Gaussiana descartada en aproximación simple: inicio {mu_init-2*sigma_simple:.3f}s < {stim_start+0.03:.3f}s")
                continue
            if 2 * sigma_simple > 0.8:
                logging.info(f"Gaussiana descartada en aproximación simple: duración {4*sigma_simple:.3f}s > 0.8s")
                continue
            gaussians.append({'A_gauss': A_init, 'mu_gauss': mu_init, 'sigma_gauss': sigma_simple})

    # Se mantiene el filtrado de solapamientos
    gaussians = sorted(gaussians, key=lambda g: g['mu_gauss'])
    filtered_gaussians = []
    skip_next = False
    for i in range(len(gaussians)):
        if skip_next:
            skip_next = False
            continue
        current = gaussians[i]
        if i < len(gaussians) - 1:
            next_g = gaussians[i+1]
            if abs(next_g['mu_gauss'] - current['mu_gauss']) < 0.5 * (current['sigma_gauss'] + next_g['sigma_gauss']):
                if current['A_gauss'] >= next_g['A_gauss']:
                    filtered_gaussians.append(current)
                else:
                    filtered_gaussians.append(next_g)
                skip_next = True
            else:
                filtered_gaussians.append(current)
        else:
            filtered_gaussians.append(current)
    return filtered_gaussians







# ------------------------------------------------------------------------------
def minimum_jerk_velocity(t, A, t0, T):
    if T <= 0:
        return np.zeros_like(t)
    tau = (t - t0) / T
    valid_idx = (tau >= 0) & (tau <= 1)
    v = np.zeros_like(t)
    # Note the division by T
    v[valid_idx] = (30 * A / T) * (tau[valid_idx]**2) * (1 - tau[valid_idx])**2
    return v



def sum_of_minimum_jerk(t, *params):
    n_submovements = len(params) // 3
    v_total = np.zeros_like(t)
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        v_total += minimum_jerk_velocity(t, A, t0, T)
    return v_total


def regularized_residuals(p, t, observed_velocity, lambda_reg, A_target):
    residual = sum_of_minimum_jerk(t, *p) - observed_velocity
    amplitudes = p[0::3]
    penalty = np.sqrt(lambda_reg) * (amplitudes - A_target)
    return np.concatenate([residual, penalty])



def fit_velocity_profile(t, observed_velocity, n_submovements, 
                         lambda_reg=0.1, loss='soft_l1', f_scale=0.5):
    # Get initial guess from peaks
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
    epsilon = 1e-3
    for i in range(n_submovements):
        min_T = 0.01
        lower_bounds.extend([epsilon, t[0], min_T])
        upper_bounds.extend([np.inf, t[-1], total_time])
    
    params_init = np.maximum(params_init, lower_bounds)
    params_init = np.minimum(params_init, upper_bounds)

    # Set A_target to be the maximum observed velocity (or a fraction thereof)
    A_target = np.max(observed_velocity)
    if A_target < epsilon:
        A_target = epsilon

    try:
        result = least_squares(
            lambda p: regularized_residuals(p, t, observed_velocity, lambda_reg, A_target),
            x0=params_init,
            bounds=(lower_bounds, upper_bounds),
            loss=loss,
            f_scale=f_scale
        )
    except ValueError as e:
        logging.error(f"Fallo en el ajuste: {e}")
        return None

    return result


def robust_fit_velocity_profile(t, observed_velocity, n_submovements, 
                                n_restarts=5, 
                                lambda_reg=0.1, loss='soft_l1', f_scale=0.5):
    """
    Try multiple restarts with slightly perturbed initial guesses.
    Returns the result with the lowest cost.
    """
    # Define a small epsilon in this function scope
    epsilon = 1e-3
    
    best_result = None
    best_cost = np.inf

    # Get a base initial guess from our original routine.
    base_result = fit_velocity_profile(t, observed_velocity, n_submovements, lambda_reg, loss, f_scale)
    if base_result is None:
        logging.error("Initial fit failed.")
        return None
    best_result = base_result
    best_cost = base_result.cost

    # Now try additional restarts with random perturbations.
    for i in range(n_restarts):
        # perturb the initial guess by up to ±10%
        perturbation = np.random.uniform(0.9, 1.1, size=len(base_result.x))
        perturbed_init = base_result.x * perturbation
        # Avoid amplitudes falling to zero
        perturbed_init = np.maximum(perturbed_init, epsilon)
        try:
            result = least_squares(
                lambda p: regularized_residuals(p, t, observed_velocity, lambda_reg, A_target=np.max(observed_velocity)),
                x0=perturbed_init,
                bounds=([epsilon for _ in base_result.x], [np.inf for _ in base_result.x]),
                loss=loss,
                f_scale=f_scale
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception as e:
            logging.warning(f"Restart {i} failed: {e}")
            continue

    logging.info(f"Robust fit result cost: {best_cost}")
    logging.info(f"Fitted parameters: {best_result.x}")
    return best_result





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

            df_filtered = df[df[likelihood_col] > 0.6]
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

            # Panel 1: Desplazamiento
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
            csv_name = trial.get('csv_filename', 'Archivo desconocido')
            if csv_name != 'Archivo desconocido':
                # Extraemos la llave que contiene fecha, hora y el frame de inicio usando regex.
                m = re.search(r'(\d{8}_\d{9}_\d+)', csv_name)
                key = m.group(1) if m else csv_name
            else:
                key = csv_name
            ax_disp.set_title(f"Ensayo {trial.get('trial_index', 0) + 1}\n{key}")


            if idx_col == 0:
                ax_disp.legend(fontsize=8)

            # Panel 2: Velocidad + Umbral
            # Panel 2: Velocidad + Umbral
            ax_vel.plot(t_vel, vel, color=VELOCITY_COLOR, alpha=0.8, label='Velocidad')
            ax_vel.axhline(mean_vel_pre, color='lightcoral', ls='-', label=f'Median Pre={mean_vel_pre:.1f}')
            ax_vel.fill_between(t_vel, mean_vel_pre - std_vel_pre, mean_vel_pre + std_vel_pre,
                                color='lightcoral', alpha=0.1, label='±1MAD Pre')
            ax_vel.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')

            ax_vel.set_xlabel('Tiempo (s)')
            ax_vel.set_ylabel('Vel. (px/s)')
            ax_vel.set_xlim(0, max_time)
            ax_vel.set_ylim(0, max_vel + 5)
            if idx_col == 0:
                ax_vel.legend(fontsize=7)

            # Panel 3: Rango de Movimientos
            ax_mov.set_xlabel('Tiempo (s)')
            ax_mov.set_ylabel('Modelos')
            ax_mov.set_xlim(0, max_time)
            ax_mov.set_ylim(0.8, 1.2)
            ax_mov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # Calcular número total de segmentos (Threshold) durante el estímulo
            n_threshold_segments = len([mov for mov in mov_ranges if mov.get('Periodo','') == 'Durante Estímulo'])

            # 1. Modelo Threshold-based:
            threshold_label_added = False
            
            for mov in mov_ranges:
                periodo = mov.get('Periodo', 'Desconocido')
                if periodo != 'Durante Estímulo':
                    continue
                startF = mov['Inicio Movimiento (Frame)']
                endF = mov['Fin Movimiento (Frame)']
                color_threshold = 'red'
                if not threshold_label_added:
                    label_thresh = f"Threshold: {n_threshold_segments} movs"
                    ax_mov.hlines(y=1.0, xmin=startF/fs, xmax=endF/fs, color=color_threshold, linewidth=2,
                                  label=label_thresh)
                    threshold_label_added = True
                else:
                    ax_mov.hlines(y=1.0, xmin=startF/fs, xmax=endF/fs, color=color_threshold, linewidth=2)
                idx_peak = np.argmax(vel[startF:endF+1])
                peak_time = (startF + idx_peak) / fs
                ax_mov.plot(peak_time, 1.0, 'o', color=color_threshold, markersize=4)
                peak_value = vel[startF + idx_peak]
                ax_mov.text(peak_time, 1.0 + 0.02, f"{peak_value:.1f}", fontsize=4, ha='center', va='bottom')


            # 2. Modelo Gaussian:
            # 2. Gaussian model:
            gauss_label_added = False
          
            total_gaussians = sum(len(subm.get('gaussians', [])) for subm in submovements)

            for i_sub, subm in enumerate(submovements):
                gauss_list = subm.get('gaussians', [])
                for j, g in enumerate(gauss_list):
                    A_g = g['A_gauss']
                    mu_g = g['mu_gauss']
                    sigma_g = g['sigma_gauss']
                    # Define the horizontal range for the gaussian (using ±2σ)
                    left_g = mu_g - 2 * sigma_g
                    right_g = mu_g + 2 * sigma_g
                    color_gauss = GAUSSIAN_PALETTE[j % len(GAUSSIAN_PALETTE)]
                    if not gauss_label_added:
                        label_gauss = f"Gaussian: {total_gaussians} submovs"
                        ax_mov.hlines(y=0.94, xmin=left_g, xmax=right_g, color=color_gauss,
                                    linewidth=1.5, linestyle='-', label=label_gauss)
                        gauss_label_added = True
                    else:
                        ax_mov.hlines(y=0.94, xmin=left_g, xmax=right_g, color=color_gauss,
                                    linewidth=1.5, linestyle='-')
                    # Plot the gaussian peak marker:
                    ax_mov.plot(mu_g, 0.94, 'o', color=color_gauss, markersize=4)
                    # Add the peak value annotation:
                    ax_mov.text(mu_g, 0.94 + 0.02, f"{A_g:.1f}", fontsize=4, ha='center', va='bottom')



            # 3. Modelo Minimum Jerk:
            minjerk_label_added = False
            for subm in submovements:
                if subm.get('MovementType') == 'MinimumJerk':
                    t_model = subm['t_segment_model']  # vector de tiempos del ajuste Minimum Jerk
                    v_sm = subm['v_sm']               # velocidad del ajuste Minimum Jerk
                    t_start = t_model[0]
                    t_end = t_model[-1]
                    peak_idx = np.argmax(v_sm)
                    t_peak = t_model[peak_idx]
                    # Dibujar el marcador
                    ax_mov.plot(t_peak, 0.86, 'o', color='lightgreen', markersize=4)
                    # Agregar el texto con el valor del pico del modelo Minimum Jerk
                    peak_value_minjerk = v_sm[peak_idx]
                    ax_mov.text(t_peak, 0.86 + 0.02, f"{peak_value_minjerk:.1f}", fontsize=4, ha='center', va='bottom')
                    # También dibujar la línea horizontal representativa
                    if not minjerk_label_added:
                        minjerk_count = sum(1 for s in submovements if s.get('MovementType') == 'MinimumJerk')
                        label_minjerk = f"MinJerk: {minjerk_count} submovs"
                        ax_mov.hlines(y=0.86, xmin=t_start, xmax=t_end, color='lightgreen',
                                    linewidth=1.5, linestyle='-', label=label_minjerk)
                        minjerk_label_added = True
                    else:
                        ax_mov.hlines(y=0.86, xmin=t_start, xmax=t_end, color='lightgreen',
                                    linewidth=1.5, linestyle='-')


            handles, labels = ax_mov.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax_mov.legend(unique.values(), unique.keys(), fontsize=9, loc='upper right')

            # Panel 4: Velocidad observada + Modelo Minimum Jerk
            ax_submov.plot(t_vel, vel, color=VELOCITY_COLOR, label='Velocidad')
            ax_submov.axhline(threshold, color='k', ls='--', label='Umbral')
            ax_submov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')

            # Filtrar solo los submovimientos de MinimumJerk
            minjerk_submovs = [s for s in submovements if s.get('MovementType') == 'MinimumJerk']
            # Trazar el modelo Minimum Jerk con líneas punteadas "hotpink"
            for i, subm in enumerate(minjerk_submovs):
                t_model = subm['t_segment_model']
                v_sm = subm['v_sm']
                if i == 0:
                    # Solo en la primera iteración se agrega la etiqueta con el conteo total
                    label_minjerk = f"MinJerk ({len(minjerk_submovs)} submovs)"
                    ax_submov.plot(t_model, v_sm, ls=':', color='hotpink', alpha=0.9, label=label_minjerk)
                else:
                    ax_submov.plot(t_model, v_sm, ls=':', color='hotpink', alpha=0.9)


            for i_sub, subm in enumerate(submovements):
                gauss_list = subm.get('gaussians', [])
                for j, g in enumerate(gauss_list):
                    A_g = g['A_gauss']
                    mu_g = g['mu_gauss']
                    sigma_g = g['sigma_gauss']
                    left_g = mu_g - 3 * sigma_g
                    right_g = mu_g + 3 * sigma_g
                    t_gm = np.linspace(left_g, right_g, 200)
                    gauss_curve = gaussian(t_gm, A_g, mu_g, sigma_g)
                    # Usar el mismo color de la paleta global:
                    color_gauss = GAUSSIAN_PALETTE[j % len(GAUSSIAN_PALETTE)]
                    ax_submov.plot(t_gm, gauss_curve, '--', color=color_gauss, alpha=0.7,
                                label='Gauss' if (i_sub==0 and j==0) else "")
                    """
                # En Panel 4, después de dibujar el modelo Minimum Jerk (si existe)
                for subm in submovements:
                    if subm.get('MovementType') == 'MinimumJerk':
                        ax_submov.plot(subm['t_segment_model'], subm['v_sm'], ':', color='lightgreen',
                                    label='MinJerk' if 'MinJerk' not in [l.get_label() for l in ax_submov.lines] else "")
    
                    """
                
            ax_submov.set_xlabel('Tiempo (s)')
            ax_submov.set_ylabel('Vel. (px/s)')
            ax_submov.set_xlim(0, max_time)
            ax_submov.set_ylim(0, max_vel + 5)
            ax_submov.legend(fontsize=8, loc='upper right')

            # Panel 5: Perfil del Estímulo
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
        dist_str = f"_Dist_{dist_ic}" if dist_ic is not None else ""
        out_filename = f"Dia_{day_str}_{body_part}_{stimulus_key}{dist_str}_Group_{fig_index+1}.png"
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

def extract_submovement_metrics(submovement_details):
    """
    Extrae un DataFrame con las métricas de cada submovimiento (solo durante el estímulo)
    de forma unificada para los tres modelos.
    """
    rows = []
    for rec in submovement_details:
        mt = rec.get('MovementType', '')
        common_data = {
            'Ensayo': rec.get('Ensayo'),
            'Dia experimental': rec.get('Dia experimental'),
            'body_part': rec.get('body_part'),
            'Estímulo': rec.get('Estímulo'),
            'MovementType': mt,
            'Coordenada_x': rec.get('Coordenada_x'),
            'Coordenada_y': rec.get('Coordenada_y'),
            'Distancia Intracortical': rec.get('Distancia Intracortical')
        }
        if mt == 'Threshold-based':
            inicio = rec.get('Latencia al Inicio (s)')
            dur_total = rec.get('Duración Total (s)')
            common_data.update({
                'Inicio (s)': inicio,
                'Fin (s)': inicio + dur_total if dur_total is not None else np.nan,
                'Duración (s)': dur_total,
                'Valor Pico (velocidad)': rec.get('Valor Pico (velocidad)'),
                'Latencia Pico (s)': rec.get('Latencia al Pico (s)')
            })
        elif mt == 'Gaussian-based':
            mu = rec.get('mu_gauss')
            sigma = rec.get('sigma_gauss')
            inicio = mu - 2 * sigma if (mu is not None and sigma is not None) else np.nan
            fin = mu + 2 * sigma if (mu is not None and sigma is not None) else np.nan
            duracion = 4 * sigma if sigma is not None else np.nan
            common_data.update({
                'Inicio (s)': inicio,
                'Fin (s)': fin,
                'Duración (s)': duracion,
                'Valor Pico (velocidad)': rec.get('A_gauss'),
                'Centro (s)': mu
            })
        elif mt == 'MinimumJerk':
            t_start = rec.get('t_start')
            t_end = rec.get('t_end')
            common_data.update({
                'Inicio (s)': t_start,
                'Fin (s)': t_end,
                'Duración (s)': t_end - t_start if (t_start is not None and t_end is not None) else np.nan,
                'Valor Pico (velocidad)': rec.get('valor_pico'),
                'Tiempo Pico (s)': rec.get('t_peak')
            })
        rows.append(common_data)
    return pd.DataFrame(rows)

# =====================================================
# Función para calcular el threshold (umbral) por día y body part
# (ignora la distancia intracortical, ya que se supone que la coordenada es la misma en ese día)
# =====================================================
def compute_thresholds_by_day_bodypart(stimuli_info):
    """
    Para cada día y cada body part, se toman los valores de velocidad (ya filtrados y suavizados)
    de los primeros 100 frames (pre-estímulo) de todos los ensayos.
    En lugar de usar media y STD, se usa la mediana y la desviación absoluta mediana (MAD) para
    calcular el umbral: umbral = mediana + 2 * MAD.
    Esto ayuda a evitar que valores extremos inflen el umbral.
    """
    thresholds_dict = {}
    for dia, day_df in stimuli_info.groupby('Dia experimental'):
        for part in body_parts:
            pre_stim_values = []
            for _, row in day_df.iterrows():
                csv_path = row.get('csv_path')
                if csv_path and os.path.exists(csv_path):
                    velocidades, _ = calcular_velocidades(csv_path)
                    if part in velocidades and len(velocidades[part]) > 0:
                        # Tomamos los primeros 100 frames del período pre-estímulo
                        pre_stim = velocidades[part][:100]
                        pre_stim = pre_stim[~np.isnan(pre_stim)]
                        pre_stim_values.extend(pre_stim)
            if len(pre_stim_values) < 10:
                logging.warning(f"Datos insuficientes para threshold para {part} en día {dia}")
                thresholds_dict[(dia, part)] = {'median': None, 'mad': None, 'threshold': None}
            else:
                median_val = np.median(pre_stim_values)
                mad_val = np.median(np.abs(pre_stim_values - median_val))
                threshold = median_val + 2 * mad_val
                thresholds_dict[(dia, part)] = {'median': median_val, 'mad': mad_val, 'threshold': threshold}
                logging.info(f"Threshold para {part} en día {dia}: Mediana={median_val:.4f}, MAD={mad_val:.4f}, Umbral={threshold:.4f}")
    return thresholds_dict



# =====================================================
# Función principal: recopilar datos de velocidad y submovimientos
# =====================================================
def collect_velocity_threshold_data():
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")

    # --- CALCULO DEL THRESHOLD (UMBRAL) GLOBAL POR DÍA Y BODY PART ---
    thresholds_by_day = compute_thresholds_by_day_bodypart(stimuli_info)

    all_movement_data = []
    thresholds_data = []
    processed_combinations = set()
    movement_ranges_all = []
    global submovement_details  
    submovement_details = []

    # Se agrupa según la agrupación original (incluye la "Distancia Intracortical")
    grouped_data = stimuli_info.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical'])

    for (dia_experimental, coord_x, coord_y, dist_ic), day_df in grouped_data:
        print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}')
        print(f'Número de ensayos en este grupo: {len(day_df)}')
        logging.info(f'Número de ensayos en este grupo: {len(day_df)}')

        # Asegurarse de que "Forma del Pulso" esté en minúsculas y asignar trial_number
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()
        if 'trial_number' not in day_df.columns:
            day_df['trial_number'] = day_df.index + 1

        for part in body_parts:
            logging.info(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}, Articulación {part}')
            print(f'Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}, Articulación {part}')
            processed_combinations.add((dia_experimental, coord_x, coord_y, dist_ic, 'All_Stimuli', part))

            # --- USAR EL THRESHOLD GLOBAL CALCULADO POR DÍA Y BODY PART ---
            threshold_data = thresholds_by_day.get((dia_experimental, part))
            if threshold_data is None or threshold_data['threshold'] is None:
                logging.warning(f"No se pudo calcular threshold para {part} en día {dia_experimental}. Se omite este body part.")
                continue
            threshold = threshold_data['threshold']
            global_mean = threshold_data['median']
            global_std = threshold_data['mad']

            logging.info(f"Utilizando threshold para {part} en día {dia_experimental}: {threshold:.4f}")

            # Almacenar estos valores en thresholds_data para referencia en resúmenes
            thresholds_data.append({
                'body_part': part,
                'Dia experimental': dia_experimental,
                'Coordenada_x': coord_x,
                'Coordenada_y': coord_y,
                'Distancia Intracortical': dist_ic,
                'threshold': threshold,
                'mean_pre_stim': global_mean,
                'std_pre_stim': global_std,
                'num_pre_stim_values': None  # Ya que el cálculo global usa todos los ensayos del día
            })

            # ----- Procesar cada estímulo para este grupo y body part -----
            all_stimuli_data = {}
            group_velocities = []
            group_positions = {'x': [], 'y': []}
            group_trial_indices = []
            movement_trials_in_selected = 0
            trials_passed = []
            trial_counter = 0

            unique_stimuli = day_df.drop_duplicates(subset=['Forma del Pulso', 'Duración (ms)'], keep='first')[['Forma del Pulso', 'Duración (ms)', 'trial_number']]
            for _, stim in unique_stimuli.iterrows():
                forma_pulso = stim['Forma del Pulso'].lower()
                duracion_ms = stim.get('Duración (ms)', None)
                stimulus_key = f"{forma_pulso.capitalize()}, {duracion_ms} ms"

                if duracion_ms is not None:
                    stim_df = day_df[(day_df['Forma del Pulso'] == forma_pulso) & (day_df['Duración (ms)'] == duracion_ms)]
                else:
                    stim_df = day_df[day_df['Forma del Pulso'] == forma_pulso]

                if stim_df.empty:
                    continue

                amplitudes = stim_df['Amplitud (microA)'].unique()
                amplitude_movement_counts = {}

                for amplitude in amplitudes:
                    amplitude_trials = stim_df[stim_df['Amplitud (microA)'] == amplitude]
                    movement_trials = 0
                    total_trials_part = 0
                    max_velocities = []

                    for _, rowAmp in amplitude_trials.iterrows():
                        csv_path = rowAmp.get('csv_path')
                        if csv_path and os.path.exists(csv_path):
                            velocidades, posiciones = calcular_velocidades(csv_path)
                            if part not in velocidades or len(velocidades[part]) == 0:
                                continue
                            start_frame = 100
                            # Usamos los primeros 100 frames de cada trial
                            vel_pre = velocidades[part][:start_frame]
                            vel_pre = vel_pre[~np.isnan(vel_pre)]
                            if len(vel_pre) == 0:
                                logging.warning("Trial sin datos pre-stim.")
                                continue
                            # Aquí usamos la media global (global_mean) para filtrar outliers
                            if np.nanmean(vel_pre) > global_mean + 3 * global_std:
                                logging.info("Descartando trial por exceso pre-stim.")
                                continue

                            total_trials_part += 1
                            frames = np.arange(len(velocidades[part]))
                            above_thresh = (velocidades[part] > threshold)
                            indices_above = frames[above_thresh]
                            if len(indices_above) > 0:
                                segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)
                                for seg in segments:
                                    if start_frame <= seg[0]:
                                        movement_trials += 1
                            maxVel = np.max(velocidades[part])
                            max_velocities.append(maxVel)
                            rowAmp['trial_number'] = rowAmp.get('trial_number', rowAmp.name + 1)
                    prop_movement = 1 if movement_trials > 0 else 0
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
                        'trial_number': rowAmp.get('trial_number'),
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'no_movement_trials': total_trials_part - movement_trials,
                        'proportion_movement': prop_movement
                    })

                if not amplitude_movement_counts:
                    logging.debug(f"No hay amplitudes con movimiento para {part}, día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    continue

                max_proportion = max(mdata['proportion_movement'] for mdata in amplitude_movement_counts.values())
                selected_amplitudes = [amp for amp, mdata in amplitude_movement_counts.items() if mdata['proportion_movement'] == max_proportion]
                selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amplitudes)]
                print(f"Amplitudes selec. {selected_amplitudes} => prop mov={max_proportion:.2f} para {part}, día={dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                logging.info(f"Amplitudes selec. {selected_amplitudes} con prop mov={max_proportion:.2f}")

                max_velocities = []
                for ampSel in selected_amplitudes:
                    data_amp = amplitude_movement_counts.get(ampSel, {})
                    max_velocities.extend(data_amp.get('max_velocities', []))
                y_max_velocity = np.mean(max_velocities) + np.std(max_velocities) if max_velocities else 50

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

                for _, rowStim in selected_trials.iterrows():
                    csv_path = rowStim.get('csv_path')
                    if csv_path and os.path.exists(csv_path):
                        velocidades, posiciones = calcular_velocidades(csv_path)
                        if part in velocidades:
                            vel = velocidades[part]
                            pos = posiciones[part]
                            if len(vel) == 0:
                                continue
                            start_frame = 100
                            amplitude_list, duration_list = generar_estimulo_desde_parametros(
                                rowStim['Forma del Pulso'],
                                rowStim['Amplitud (microA)'] * 1000,
                                (duracion_ms * 1000 if duracion_ms else 1000000),
                                rowStim['Frecuencia (Hz)'] if frequency is None else frequency,
                                200,
                                compensar=False if duracion_ms == 1000 else True
                            )
                            current_frame = int(start_frame + sum(duration_list))
                            vel_pre = vel[:start_frame]
                            vel_pre = vel_pre[~np.isnan(vel_pre)]
                            if len(vel_pre) == 0:
                                logging.warning(f"Trial {trial_counter} sin datos pre-stim.")
                                continue
                            if np.nanmean(vel_pre) > global_mean + 3 * global_std:
                                logging.info(f"Descartado trial {trial_counter} por pre-stim alto.")
                                continue

                            if np.any(vel[start_frame:current_frame] > threshold):
                                movement_trials_in_selected += 1
                            group_velocities.append(vel)
                            group_positions['x'].append(pos['x'])
                            group_positions['y'].append(pos['y'])
                            group_trial_indices.append(trial_counter)

                            frames_vel = np.arange(len(vel))
                            above_threshold = (vel > threshold)
                            indices_above = frames_vel[above_threshold]
                            submovements = []
                            if len(indices_above) > 0:
                                segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)
                                for seg in segments:
                                    movement_start = seg[0]
                                    movement_end = seg[-1]
                                    if movement_start < start_frame + 3:
                                        continue
                                    if (movement_end - movement_start) < 3:
                                        continue
                                    if movement_start < start_frame:
                                        periodo = 'Pre-Estímulo'
                                    elif start_frame <= movement_start <= current_frame:
                                        periodo = 'Durante Estímulo'
                                    else:
                                        periodo = 'Post-Estímulo'
                                    
                                    movement_data = {
                                        'Ensayo': trial_counter + 1,
                                        'Dia experimental': dia_experimental,
                                        'body_part': part,
                                        'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                        'MovementType': 'Threshold-based',
                                        'Periodo': periodo,
                                        'Inicio Movimiento (Frame)': movement_start,
                                        'Fin Movimiento (Frame)': movement_end,
                                        'Latencia al Inicio (s)': (movement_start - start_frame) / 100.0,
                                        'Latencia al Pico (s)': (movement_start + np.argmax(vel[movement_start:movement_end+1]) - start_frame) / 100.0,
                                        'Valor Pico (velocidad)': float(vel[movement_start + np.argmax(vel[movement_start:movement_end+1])]),
                                        'Duración Total (s)': (movement_end - movement_start) / 100.0,
                                        'Duración durante Estímulo (s)': max(0, min(movement_end, current_frame) - max(movement_start, start_frame)) / 100.0,
                                        'Coordenada_x': coord_x,
                                        'Coordenada_y': coord_y,
                                        'Distancia Intracortical': dist_ic
                                    }
                                    movement_ranges.append(movement_data)
                                    movement_ranges_all.append(movement_data)
                                    if periodo == 'Durante Estímulo':
                                        submovement_details.append(movement_data)
                                    
                                    if periodo == 'Durante Estímulo':
                                        t_segment = np.arange(movement_start, movement_end + 1) / 100.0
                                        if t_segment[0] < (start_frame/100.0 + 0.03):
                                            continue
                                        vel_segment = vel[movement_start:movement_end+1]
                                        vel_segment_filtrada = aplicar_moving_average(vel_segment, window_size=10)
                                        submov_peak_indices = detectar_submovimientos_en_segmento(vel_segment_filtrada, threshold)
                                        if len(submov_peak_indices) > 0:
                                            gaussians = fit_gaussians_submovement(t_segment, vel_segment_filtrada, threshold, start_frame/100.0)
                                            if gaussians:
                                                for g in gaussians:
                                                    rec = {
                                                        'Ensayo': trial_counter + 1,
                                                        'Dia experimental': dia_experimental,
                                                        'body_part': part,
                                                        'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                                        'MovementType': 'Gaussian-based',
                                                        'A_gauss': g.get('A_gauss'),
                                                        'mu_gauss': g.get('mu_gauss'),
                                                        'sigma_gauss': g.get('sigma_gauss'),
                                                        'Coordenada_x': coord_x,
                                                        'Coordenada_y': coord_y,
                                                        'Distancia Intracortical': dist_ic
                                                    }
                                                    submovement_details.append(rec)
                                                submovements.append({'gaussians': gaussians, 'MovementType': 'Gaussian-based'})

                                        for rep_peak in submov_peak_indices:
                                            window = 5  # Puedes ajustar este valor según tus datos
                                            local_start = max(0, rep_peak - window)
                                            local_end = min(len(t_segment) - 1, rep_peak + window)
                                            t_local = t_segment[local_start: local_end + 1]
                                            vel_local = vel_segment[local_start: local_end + 1]
                                            # Solo realizar el ajuste si la ventana tiene suficientes puntos
                                            if len(t_local) >= 3:
                                                result_local = robust_fit_velocity_profile(t_local, vel_local, 1)
                                                if result_local is not None:
                                                    params_local = result_local.x
                                                    v_minjerk_local = minimum_jerk_velocity(t_local, params_local[0], params_local[1], params_local[2])
                                                    local_peak_index = int(np.argmax(v_minjerk_local))
                                                    # Registrar cada submovimiento ajustado por MinimumJerk
                                                    rec = {
                                                        'Ensayo': trial_counter + 1,
                                                        'Dia experimental': dia_experimental,
                                                        'body_part': part,
                                                        'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                                        'MovementType': 'MinimumJerk',
                                                        't_start': t_local[0],
                                                        't_end': t_local[-1],
                                                        't_peak': t_local[local_peak_index],
                                                        'valor_pico': float(v_minjerk_local[local_peak_index]),
                                                        'Coordenada_x': coord_x,
                                                        'Coordenada_y': coord_y,
                                                        'Distancia Intracortical': dist_ic
                                                    }
                                                    submovement_details.append(rec)
                                                    # Agregar el ajuste a la lista de submovimientos para este ensayo
                                                    submovements.append({
                                                        't_segment_model': t_local,
                                                        'v_sm': v_minjerk_local,
                                                        'MovementType': 'MinimumJerk'
                                                    })

                            trial_data = {
                                'velocity': vel,
                                'positions': pos,
                                'trial_index': trial_counter,
                                'submovements': submovements,
                                'movement_ranges': [md for md in movement_ranges if md['Ensayo'] == (trial_counter + 1)],
                                'amplitude_list': amplitude_list,
                                'duration_list': duration_list,
                                'start_frame': start_frame,
                                'current_frame': current_frame,
                                'threshold': threshold,
                                'mean_vel_pre': global_mean,
                                'std_vel_pre': global_std,
                                'Ensayo': trial_counter + 1,
                                'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                'csv_filename': rowStim.get('csv_filename', 'Archivo desconocido')
                            }
                            trial_data['minjerk_count'] = sum(1 for s in submovements if s.get('MovementType') == 'MinimumJerk')
                            trials_data.append(trial_data)
                            trial_counter += 1

                if len(trials_data) == 0:
                    logging.debug(f"No hay datos de velocidad para {part} en día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    print(f"No hay datos de velocidad para {part} en día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    continue

                all_stimuli_data[stimulus_key] = {
                    'velocities': group_velocities,
                    'positions': group_positions,
                    'threshold': threshold,
                    'amplitude_list': amplitude_list,
                    'duration_list': duration_list,
                    'start_frame': start_frame,
                    'current_frame': current_frame,
                    'mean_vel_pre': global_mean,
                    'std_vel_pre': global_std,
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
                    'Order': stim.get('trial_number'),
                    'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                    'trials_data': trials_data
                }

                # Se puede llamar a la función de graficación para cada estímulo
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

    if submovement_details:
        submovement_df = pd.DataFrame(submovement_details)
        submovement_df['Submov_Num'] = submovement_df.groupby(['Ensayo', 'MovementType']).cumcount() + 1
        submovement_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
        submovement_df.to_csv(submovement_path, index=False)
        print(f"Datos detallados de submovimientos guardados en: {submovement_path}")

    submovement_metrics_df = extract_submovement_metrics(submovement_details)
    submovement_metrics_path = os.path.join(output_comparisons_dir, 'submovement_metrics_summary.csv')
    submovement_metrics_df.to_csv(submovement_metrics_path, index=False)
    print(f"Métricas unificadas de submovimientos guardadas en: {submovement_metrics_path}")

    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        print(f"Día: {combo[0]}, X={combo[1]}, Y={combo[2]}, Dist={combo[3]}, {combo[4]}, {combo[5]}")

    print("Finalizada la recopilación de datos de umbral de velocidad.")
    return counts_df


def plot_summary_movement_data_by_bodypart(submovements_df, output_dir):
    """
    For each experimental day and body part, creates a summary figure (2 rows of panels)
    where, for each metric, for each stimulus, three boxplots (one per model) are shown
    and annotated with the actual p‑values obtained from the Tukey test.
    """
    df = submovements_df[submovements_df['Periodo'] == 'Durante Estímulo'].copy()
    if df.empty:
        logging.info("No hay movimientos DURANTE estímulo para graficar (por body_part).")
        return
    if 'Ensayo' not in df.columns or 'Estímulo' not in df.columns:
        df['Ensayo'] = range(1, len(df) + 1)
        df['Estímulo'] = df['Forma del Pulso'].str.capitalize() + ', ' + df['Duración (ms)'].astype(str) + ' ms'
    
    unique_days = df['Dia experimental'].unique()
    unique_bodyparts = df['body_part'].unique()
    
    # Use the updated aggregation function
    agg_df = aggregate_trial_metrics_extended(df)
    
    pulse_duration_dict = {
        'Rectangular': [500, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa Ascendente': [1000],
        'Rampa Descendente': [1000],
        'Triple Rombo': [700]
    }
    ordered_stimuli = [f"{shape}, {dur} ms" for shape, durations in pulse_duration_dict.items() for dur in durations]
    
    movement_types = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    offset_dict = {'Threshold-based': 0, 'Gaussian-based': 0.5, 'MinimumJerk': 0.75}
    gap_between = 1.5
    
    metrics_dict = {
        'lat_inicio_ms': "Lat. Inicio (ms)",
        'lat_primer_pico_ms': "Lat. Primer Pico (ms)",
        'lat_pico_max_ms': "Lat. Pico Máx. (ms)",
        'dur_total_ms': "Duración Total (ms)",
        'valor_pico_inicial': "Valor Pico Inicial",
        'valor_pico_max': "Valor Pico Máx.",
        'num_movs': "N° Movimientos",
        'lat_inicio_mayor_ms': "Lat. Inicio Mayor (ms)",
        'lat_pico_mayor_ms': "Lat. Pico Mayor (ms)",
        'delta_valor_pico': "Delta Valor Pico"
    }
    metric_keys = list(metrics_dict.keys())
    n_cols = math.ceil(len(metric_keys) / 2)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 2 * 5), squeeze=False)
    
    # For simplicity, take the first day and the first body part
    day = list(unique_days)[0]
    body_part = list(unique_bodyparts)[0]
    df_subset = agg_df[(agg_df['Dia experimental'] == day) & (agg_df['body_part'] == body_part)]
    
    positions_by_stim = {}
    for idx, metric_key in enumerate(metric_keys):
        r, c = idx // n_cols, idx % n_cols
        ax = axs[r, c]
        boxplot_data = []
        x_positions = []
        group_centers = []
        labels = []
        current_pos = 0
        for stim in ordered_stimuli:
            df_stim = df_subset[df_subset['Estímulo'] == stim]
            if df_stim.empty:
                continue
            model_positions = {}
            for mtype in movement_types:
                data = df_stim[df_stim['MovementType'] == mtype][metric_key].dropna().values
                if len(data) == 0:
                    continue
                boxplot_data.append(data)
                pos = current_pos + offset_dict[mtype]
                x_positions.append(pos)
                model_positions[mtype] = pos
            if model_positions:
                center = np.mean(list(model_positions.values()))
                group_centers.append(center)
                labels.append(stim)
                positions_by_stim[stim] = model_positions
                current_pos = max(x_positions) + gap_between
            else:
                current_pos += len(movement_types)*0.6 + gap_between
        if not boxplot_data:
            ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
            continue
        bp_obj = ax.boxplot(boxplot_data, positions=x_positions, widths=0.6/2.2,
                            patch_artist=True, showfliers=False, whis=(0, 100))
        box_colors = sns.color_palette("Set3", n_colors=len(x_positions))
        for patch, color in zip(bp_obj['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Estímulo")
        ax.set_ylabel(metrics_dict[metric_key])
        ax.set_title(metrics_dict[metric_key])
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0], y_lims[1] + 0.35*(y_lims[1]-y_lims[0]))
        
        for stim in positions_by_stim:
            pval_matrix = get_tukey_pvals_for_stimulus(df_subset, stim, metric_key)
            if pval_matrix is not None:
                box_positions = positions_by_stim[stim]
                pairs = list(itertools.combinations(sorted(box_positions.keys()), 2))
                add_significance_brackets(ax, pairs, pval_matrix, box_positions,
                                          y_offset=0.1, line_height=0.05, font_size=10)
    
    custom_handles = [Patch(facecolor=color, edgecolor='black', label=mtype) for mtype, color in zip(movement_types, sns.color_palette("Set3", n_colors=len(movement_types)))]
    fig.legend(handles=custom_handles, loc='upper right', title='Modelos')
    
    fig.suptitle(f"{body_part} - Día {day} (Resumen por Body Part)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fname = f"summary_by_bp_{body_part}_{sanitize_filename(str(day))}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Resumen por body_part para {body_part} guardado en: {out_path}")







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
    h_base = y_lim[1] + (y_lim[1] - y_lim[0]) * y_offset  # posición base un poco por encima
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
        # Si p < 0.05 se añade un asterisco
        if pval < 0.05:
            p_text = f"* p={pval:.3g}"
        else:
            p_text = f"p={pval:.3g}"
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
    Realiza ANOVA (dos factores: Forma del Pulso y Duración (ms)) y tests de Friedman
    utilizando las métricas extendidas (las que se generan en aggregate_trial_metrics_extended).
    Se realizan además pruebas post‑hoc (Tukey) para cada métrica.
    """
    if output_dir is None:
        output_dir = output_comparisons_dir

    # Primero, agregamos las métricas extendidas:
    agg_df = aggregate_trial_metrics_extended(movement_ranges_df)

    agg_df['Forma del Pulso'] = agg_df['Estímulo'].apply(
        lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan
    )
    agg_df['Duración (ms)'] = agg_df['Estímulo'].apply(
        lambda s: s.split(', ')[1].replace(' ms', '') if isinstance(s, str) and ', ' in s else np.nan
    ).astype(float)

    
    # Usamos directamente la DataFrame agregada para los tests:
    df = agg_df.copy()
    
    # Definir las métricas a analizar (estas columnas existen en df)
    metrics = ['lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_max_ms',
               'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
               'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms']

    # El agrupamiento se hará por "Dia experimental", "body_part" y "MovementType"
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []
    results_friedman = []
    posthoc_results = []
    posthoc_signifs = []

    grouped_stats = df.groupby(grouping)
    for (dia, bp, movtype), df_sub in grouped_stats:
        if len(df_sub) < 3:
            continue
        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue
            # La fórmula usa 'Forma del Pulso' y 'Duración (ms)'
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

                # Interaction post-hoc (by Forma del Pulso)
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

        # Friedman test (optional)
        try:
            metric_friedman = 'lat_inicio_ms'
            df_sub['Condicion'] = df_sub['Forma del Pulso'] + "_" + df_sub['Duración (ms)'].astype(str)
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

    # Save results
    anova_df_out = pd.DataFrame(results_anova)
    if not anova_df_out.empty:
        anova_path = os.path.join(output_dir, 'anova_twofactor_results.csv')
        anova_df_out.to_csv(anova_path, index=False)
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


def do_significance_tests_detailed(submovements_df, output_dir=None):
    """
    Realiza pruebas de significancia (ANOVA y post‑hoc) utilizando los submovimientos detallados.
    Se agrupan los submovimientos detallados (por ejemplo, del archivo submovement_detailed_summary)
    por Ensayo, Estímulo, MovementType, Día experimental y body_part, y se calculan métricas extendidas.
    Luego se realizan pruebas ANOVA (usando 'Forma del Pulso' y 'Duración (ms)' extraídos de 'Estímulo')
    para cada combinación de Día experimental, body_part y MovementType, y se generan gráficos post‑hoc
    con los p‑valores reales obtenidos del test de Tukey.
    
    Los títulos y leyendas se han ajustado para dejar claro que se está trabajando con submovimientos detallados.
    """
    if output_dir is None:
        output_dir = output_comparisons_dir

    # Agregar métricas extendidas a partir de los submovimientos detallados
    agg_df = aggregate_trial_metrics_extended(submovements_df)
    
    # Extraer "Forma del Pulso" y "Duración (ms)" de la columna 'Estímulo'
    agg_df['Forma del Pulso'] = agg_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    agg_df['Duración (ms)'] = agg_df['Estímulo'].apply(
        lambda s: s.split(', ')[1].replace(' ms', '') if isinstance(s, str) and ', ' in s else np.nan
    ).astype(float)
    
    # Se trabaja sobre este DataFrame agregado
    df = agg_df.copy()
    
    # Definir las métricas a analizar (estas columnas ya han sido calculadas en agg_df)
    metrics = [
        'lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_max_ms',
        'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
        'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms'
    ]
    
    # Agrupar por Día experimental, body_part y MovementType
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []
    results_friedman = []
    posthoc_results = []
    posthoc_signifs = []
    
    grouped_stats = df.groupby(grouping)
    for (dia, bp, movtype), df_sub in grouped_stats:
        # Si no hay suficientes registros, omitir
        if len(df_sub) < 3:
            continue
        for metric in metrics:
            data_metric = df_sub.dropna(subset=[metric])
            if len(data_metric) < 3:
                continue
            # Usamos 'Forma del Pulso' y 'Duración (ms)' como factores independientes
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
    
                # Post-hoc para "Forma del Pulso"
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
                                # Generar gráfico de heatmap para la matriz de p-valores
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
                                # Generar boxplot con anotaciones reales
                                plot_boxplot_metric_with_posthoc(
                                    data_metric, metric, 'Forma del Pulso',
                                    tukey_df=tk_df, pvals_matrix=pvals_mat,
                                    output_dir=output_dir
                                )
                except KeyError:
                    pass
    
                # Post-hoc para "Duración (ms)"
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
    
                # Post-hoc de interacción (por Forma del Pulso, considerando Duración)
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
                logging.warning(f"Fallo ANOVA en {dia}, {bp}, {movtype}, {metric}: {e}")
                continue
    
        # Test de Friedman (opcional)
        try:
            metric_friedman = 'lat_inicio_ms'
            df_sub['Condicion'] = df_sub['Forma del Pulso'] + "_" + df_sub['Duración (ms)'].astype(str)
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
            logging.warning(f"Fallo Friedman en {dia}, {bp}, {movtype}: {e}")
    
    # Guardar resultados en archivos CSV
    anova_df_out = pd.DataFrame(results_anova)
    if not anova_df_out.empty:
        anova_path = os.path.join(output_dir, 'anova_twofactor_results_detailed.csv')
        anova_df_out.to_csv(anova_path, index=False)
        print(f"Resultados ANOVA (detallados) guardados en {anova_path}")
    else:
        print("No se generaron resultados ANOVA (detallados).")
    
    friedman_df = pd.DataFrame(results_friedman)
    if not friedman_df.empty:
        friedman_path = os.path.join(output_dir, 'friedman_results_detailed.csv')
        friedman_df.to_csv(friedman_path, index=False)
        print(f"Resultados Friedman (detallados) guardados en {friedman_path}")
    else:
        print("No se generaron resultados Friedman (detallados).")
    
    if posthoc_results:
        posthoc_all = pd.concat(posthoc_results, ignore_index=True)
        posthoc_path = os.path.join(output_dir, 'posthoc_tukey_results_detailed.csv')
        posthoc_all.to_csv(posthoc_path, index=False)
        print(f"Resultados Post-hoc (Tukey) detallados guardados en {posthoc_path}")
    
    if posthoc_signifs:
        pvals_concat = pd.concat(posthoc_signifs, ignore_index=True)
        pvals_path = os.path.join(output_dir, 'posthoc_significance_matrices_detailed.csv')
        pvals_concat.to_csv(pvals_path, index=False)
        print(f"Matrices de p-values detalladas guardadas en {pvals_path}")

# FIN de do_significance_tests_detailed



# --- NUEVO: Función para graficar un boxplot con líneas de significancia basadas en el post-hoc
# --- Función para graficar un boxplot (usando 100% de los datos para calcular la caja)
# pero que muestre únicamente la caja central (IQR) y la mediana,
# junto con los brackets de significancia obtenidos del post-hoc.
def plot_boxplot_metric_with_posthoc(data_metric, metric, factor_name, tukey_df, pvals_matrix, output_dir,
                                     grouping_cols=['Dia experimental', 'body_part', 'MovementType']):
    import os
    # Order the levels (for example, alphabetically)
    factor_levels = sorted(data_metric[factor_name].unique())
    # Define spacing: each group will be placed 2.5 units apart
    spacing = 2.5
    positions = [i * spacing for i in range(len(factor_levels))]
    
    fig, ax = plt.subplots(figsize=(max(8, len(factor_levels)*2.5), 6))
    # Get lists of data for each level in order:
    grouped_lists = data_metric.groupby(factor_name)[metric].apply(list)
    bp = ax.boxplot(
        [grouped_lists[lvl] for lvl in factor_levels],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100)
    )
    
    # Hide whiskers and caps
    for part in ('whiskers', 'caps'):
        for line in bp[part]:
            line.set_visible(False)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # <<-- FIX: Define box_colors so that its length matches the number of groups
    box_colors = sns.color_palette('Set3', n_colors=len(factor_levels))
    
    # Assign colors to each box:
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(factor_levels, rotation=45, ha='right')
    ax.set_xlabel(factor_name)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {factor_name} (post-hoc)")
    
    # Annotate each box with the median and count
    group_stats = data_metric.groupby(factor_name)[metric].agg(['median', 'count'])
    for i, lvl in enumerate(factor_levels):
        med = group_stats.loc[lvl, 'median']
        cnt = group_stats.loc[lvl, 'count']
        median_y = bp['medians'][i].get_ydata()[0]
        y_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(positions[i], median_y + y_offset, f"{med:.2f}\n(n={cnt})", ha='center', va='bottom', fontsize=10)
    
    # Build a dictionary for bracket positions
    box_positions = {lvl: pos for lvl, pos in zip(factor_levels, positions)}
    p_values_dict = {}
    for row in tukey_df.itertuples():
        g1 = getattr(row, 'group1')
        g2 = getattr(row, 'group2')
        pval = getattr(row, 'p_adj')
        p_values_dict[(g1, g2)] = pval
    pairs = list(itertools.combinations(factor_levels, 2))
    
    # Add significance brackets
    add_significance_brackets(ax, pairs, p_values_dict, box_positions,
                              y_offset=0.1, line_height=0.05, font_size=10)
    
    fname = f"boxplot_posthoc_{sanitize_filename(factor_name)}_{sanitize_filename(metric)}.png"
    out_path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Boxplot with post-hoc saved in: {out_path}")



def aggregate_trial_metrics_extended(submovements_df):
    """
    Groups submovements (from the submovement_detailed_summary file) by Trial, Stimulus,
    MovementType, Experimental Day, and Body Part and calculates 10 metrics for each model.
    For 'Threshold-based', the duration is now calculated as the difference between the start 
    of the first submovement and the finish of the last submovement that has initiated during the stimulus.
    """
    fs = 100.0

    def agg_func(group):
        mt = group['MovementType'].iloc[0]
        keys = ["lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_max_ms",
                "dur_total_ms", "valor_pico_inicial", "valor_pico_max",
                "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_valor_pico"]
        if group.empty:
            return pd.Series({k: 0 for k in keys})
        
        if mt == 'Threshold-based':
            # Use only submovements during stimulus
            if group['Latencia al Inicio (s)'].dropna().empty:
                return pd.Series({k: 0 for k in keys})
            first_start = group['Inicio Movimiento (Frame)'].min()
            # Instead of taking max of Fin Movimiento, we select the one that initiated last.
            idx_last = group['Inicio Movimiento (Frame)'].idxmax()
            last_frame = group.loc[idx_last, 'Fin Movimiento (Frame)']
            dur_total_ms = ((last_frame - first_start) / fs) * 1000

            lat_inicio_ms = group['Latencia al Inicio (s)'].min() * 1000
            lat_primer_pico_ms = group['Latencia al Pico (s)'].min() * 1000
            lat_pico_max_ms = group['Latencia al Pico (s)'].max() * 1000
            idx_min = group['Latencia al Inicio (s)'].idxmin()
            valor_pico_inicial = group.loc[idx_min, 'Valor Pico (velocidad)'] if pd.notnull(idx_min) else 0
            valor_pico_max = group['Valor Pico (velocidad)'].max()
            num_movs = len(group)
            idx_max = group['Valor Pico (velocidad)'].idxmax() if not group['Valor Pico (velocidad)'].dropna().empty else None
            lat_inicio_mayor_ms = group.loc[idx_max, 'Latencia al Inicio (s)'] * 1000 if idx_max is not None else 0
            lat_pico_mayor_ms = group.loc[idx_max, 'Latencia al Pico (s)'] * 1000 if idx_max is not None else 0
            delta_valor_pico = valor_pico_max - valor_pico_inicial
            return pd.Series({
                "lat_inicio_ms": lat_inicio_ms,
                "lat_primer_pico_ms": lat_primer_pico_ms,
                "lat_pico_max_ms": lat_pico_max_ms,
                "dur_total_ms": dur_total_ms,
                "valor_pico_inicial": valor_pico_inicial,
                "valor_pico_max": valor_pico_max,
                "num_movs": num_movs,
                "lat_inicio_mayor_ms": lat_inicio_mayor_ms,
                "lat_pico_mayor_ms": lat_pico_mayor_ms,
                "delta_valor_pico": delta_valor_pico
            })
        elif mt == 'Gaussian-based':
            if group[['mu_gauss', 'sigma_gauss', 'A_gauss']].dropna().empty:
                return pd.Series({k: 0 for k in keys})
            start_times = group['mu_gauss'] - 2 * group['sigma_gauss']
            end_times = group['mu_gauss'] + 2 * group['sigma_gauss']
            lat_inicio_ms = start_times.min() * 1000
            lat_primer_pico_ms = group['mu_gauss'].min() * 1000
            lat_pico_max_ms = group['mu_gauss'].max() * 1000
            dur_total_ms = (end_times.max() - start_times.min()) * 1000
            idx_min = start_times.idxmin()
            valor_pico_inicial = group.loc[idx_min, 'A_gauss'] if pd.notnull(idx_min) else 0
            valor_pico_max = group['A_gauss'].max()
            num_movs = len(group)
            idx_max = group['A_gauss'].idxmax() if not group['A_gauss'].dropna().empty else None
            lat_inicio_mayor_ms = ((group.loc[idx_max, 'mu_gauss'] - 2 * group.loc[idx_max, 'sigma_gauss']) * 1000) if idx_max is not None else 0
            lat_pico_mayor_ms = group.loc[idx_max, 'mu_gauss'] * 1000 if idx_max is not None else 0
            delta_valor_pico = valor_pico_max - valor_pico_inicial
            return pd.Series({
                "lat_inicio_ms": lat_inicio_ms,
                "lat_primer_pico_ms": lat_primer_pico_ms,
                "lat_pico_max_ms": lat_pico_max_ms,
                "dur_total_ms": dur_total_ms,
                "valor_pico_inicial": valor_pico_inicial,
                "valor_pico_max": valor_pico_max,
                "num_movs": num_movs,
                "lat_inicio_mayor_ms": lat_inicio_mayor_ms,
                "lat_pico_mayor_ms": lat_pico_mayor_ms,
                "delta_valor_pico": delta_valor_pico
            })
        elif mt == 'MinimumJerk':
            if group[['t_start', 't_peak', 't_end', 'valor_pico']].dropna().empty:
                return pd.Series({k: 0 for k in keys})
            lat_inicio_ms = group['t_start'].min() * 1000
            lat_primer_pico_ms = group['t_peak'].min() * 1000
            lat_pico_max_ms = group['t_peak'].max() * 1000
            dur_total_ms = (group['t_end'].max() - group['t_start'].min()) * 1000
            idx_min = group['t_start'].idxmin()
            valor_pico_inicial = group.loc[idx_min, 'valor_pico'] if pd.notnull(idx_min) else 0
            valor_pico_max = group['valor_pico'].max()
            num_movs = len(group)
            idx_max = group['valor_pico'].idxmax() if not group['valor_pico'].dropna().empty else None
            lat_inicio_mayor_ms = group.loc[idx_max, 't_start'] * 1000 if idx_max is not None else 0
            lat_pico_mayor_ms = group.loc[idx_max, 't_peak'] * 1000 if idx_max is not None else 0
            delta_valor_pico = valor_pico_max - valor_pico_inicial
            return pd.Series({
                "lat_inicio_ms": lat_inicio_ms,
                "lat_primer_pico_ms": lat_primer_pico_ms,
                "lat_pico_max_ms": lat_pico_max_ms,
                "dur_total_ms": dur_total_ms,
                "valor_pico_inicial": valor_pico_inicial,
                "valor_pico_max": valor_pico_max,
                "num_movs": num_movs,
                "lat_inicio_mayor_ms": lat_inicio_mayor_ms,
                "lat_pico_mayor_ms": lat_pico_mayor_ms,
                "delta_valor_pico": delta_valor_pico
            })
        else:
            return pd.Series({k: 0 for k in keys})

    agg = submovements_df.groupby(
        ['Ensayo', 'Estímulo', 'MovementType', 'Dia experimental', 'body_part']
    ).apply(agg_func).reset_index()

    trial_keys = submovements_df[['Ensayo', 'Estímulo', 'Dia experimental', 'body_part']].drop_duplicates().reset_index(drop=True)
    all_rows = []
    for _, trial in trial_keys.iterrows():
        for mt in ['Threshold-based', 'Gaussian-based', 'MinimumJerk']:
            row = agg[
                (agg['Ensayo'] == trial['Ensayo']) &
                (agg['Estímulo'] == trial['Estímulo']) &
                (agg['Dia experimental'] == trial['Dia experimental']) &
                (agg['body_part'] == trial['body_part']) &
                (agg['MovementType'] == mt)
            ]
            if row.empty:
                new_row = trial.to_dict()
                new_row['MovementType'] = mt
                new_row.update({key: 0 for key in keys})
                all_rows.append(new_row)
            else:
                all_rows.append(row.iloc[0].to_dict())
    aggregated = pd.DataFrame(all_rows)
    return aggregated










def get_tukey_pvals_for_stimulus(agg_df, stim, metric):
    """
    Para un estímulo (stim) y una métrica dada, calcula la matriz de p‐valores
    comparando los tres MovementType usando un test de Tukey.
    Devuelve una matriz pandas (DataFrame) o None si no hay suficientes grupos.
    """
    df_sub = agg_df[agg_df['Estímulo'] == stim]
    if df_sub['MovementType'].nunique() < 2:
        return None
    tukey_res = pairwise_tukeyhsd(endog=df_sub[metric].values,
                                  groups=df_sub['MovementType'].values,
                                  alpha=0.05)
    tk_df = pd.DataFrame(data=tukey_res._results_table.data[1:], 
                         columns=tukey_res._results_table.data[0])
    tk_df = tk_df.rename(columns={'p-adj': 'p_adj'})
    groups = sorted(df_sub['MovementType'].unique())
    pval_matrix = build_significance_matrix_from_arrays(groups, tk_df)
    return pval_matrix

def compute_model_pvals(agg_df, metric):
    models = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    pvals = {}
    for mt in models:
        df_model = agg_df[agg_df['MovementType'] == mt]
        if len(df_model) < 3:
            pvals[mt] = (np.nan, np.nan, np.nan)
        else:
            try:
                mod = ols(f"Q('{metric}') ~ C(Q('Forma del Pulso')) * C(Q('Duración (ms)'))", data=df_model).fit()
                anova_res = anova_lm(mod, typ=2)
                p_shape = anova_res.loc["C(Q('Forma del Pulso'))", "PR(>F)"] if "C(Q('Forma del Pulso'))" in anova_res.index else np.nan
                p_dur = anova_res.loc["C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Duración (ms)'))" in anova_res.index else np.nan
                p_int = anova_res.loc["C(Q('Forma del Pulso')):C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Forma del Pulso')):C(Q('Duración (ms)'))" in anova_res.index else np.nan
                pvals[mt] = (p_shape, p_dur, p_int)
            except Exception as e:
                pvals[mt] = (np.nan, np.nan, np.nan)
    return pvals

def plot_global_summary_with_significance_all_extended(submovements_df, output_dir):
    # Aggregate data
    agg_df = aggregate_trial_metrics_extended(submovements_df)
    agg_df['Forma del Pulso'] = agg_df['Estímulo'].apply(
        lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    agg_df['Duración (ms)'] = agg_df['Estímulo'].apply(
        lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)
    
    # Define ordered stimuli
    pulse_duration_dict = {
        'Rectangular': [500, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa Ascendente': [1000],
        'Rampa Descendente': [1000],
        'Triple Rombo': [700]
    }
    ordered_stimuli = [f"{shape}, {dur} ms" for shape, durations in pulse_duration_dict.items() for dur in durations]
    
    movement_types = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    offset_dict = {'Threshold-based': 0, 'Gaussian-based': 0.5, 'MinimumJerk': 1.0}
    gap_between = 1.5
    metrics_dict = {
        "lat_inicio_ms": "Lat. Inicio (ms)",
        "lat_primer_pico_ms": "Lat. Primer Pico (ms)",
        "lat_pico_max_ms": "Lat. Pico Máx. (ms)",
        "dur_total_ms": "Duración Total (ms)",
        "valor_pico_inicial": "Valor Pico Inicial",
        "valor_pico_max": "Valor Pico Máx.",
        "num_movs": "N° Movimientos",
        "lat_inicio_mayor_ms": "Lat. Inicio Mayor (ms)",
        "lat_pico_mayor_ms": "Lat. Pico Mayor (ms)",
        "delta_valor_pico": "Delta Valor Pico"
    }
    metric_keys = list(metrics_dict.keys())
    n_cols = math.ceil(len(metric_keys) / 2)
    
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 4, 2 * 4), squeeze=False)
    positions_by_stim = {}
    
    for idx, metric_key in enumerate(metric_keys):
        r, c = idx // n_cols, idx % n_cols
        ax = axs[r, c]
        boxplot_data = []
        x_positions = []
        group_centers = []
        labels = []
        current_pos = 0
        
        for stim in ordered_stimuli:
            df_stim = agg_df[agg_df['Estímulo'] == stim]
            if df_stim.empty:
                continue
            model_positions = {}
            for mtype in movement_types:
                data = df_stim[df_stim['MovementType'] == mtype][metric_key].dropna().values
                if len(data) == 0:
                    continue
                boxplot_data.append(data)
                pos = current_pos + offset_dict[mtype]
                x_positions.append(pos)
                model_positions[mtype] = pos
            if model_positions:
                center = np.mean(list(model_positions.values()))
                group_centers.append(center)
                labels.append(stim)
                positions_by_stim[stim] = model_positions
                current_pos = max(x_positions) + gap_between
            else:
                current_pos += len(movement_types) * 0.6 + gap_between
        
        if not boxplot_data:
            ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
            continue
        
        bp_obj = ax.boxplot(boxplot_data, positions=x_positions, widths=0.6/2.2,
                            patch_artist=True, showfliers=False, whis=(0, 100))
        box_colors = sns.color_palette("Set3", n_colors=len(x_positions))
        for patch, color in zip(bp_obj['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Estímulo")
        ax.set_ylabel(metrics_dict[metric_key])
        ax.set_title(metrics_dict[metric_key])
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0], y_lims[1] + 0.35*(y_lims[1]-y_lims[0]))
        
        # (Optional) Add Tukey significance brackets for each stimulus here as before...
        for stim in positions_by_stim:
            pval_matrix = get_tukey_pvals_for_stimulus(agg_df, stim, metric_key)
            if pval_matrix is not None:
                box_positions = positions_by_stim[stim]
                pairs = list(itertools.combinations(sorted(box_positions.keys()), 2))
                add_significance_brackets(ax, pairs, pval_matrix, box_positions,
                                          y_offset=0.1, line_height=0.05, font_size=10)
        
        # Now, compute p-values per model for this metric and add a subtitle text
        model_pvals = compute_model_pvals(agg_df, metric_key)
        subtitle_text = (
            f"Threshold-based: p_shape={model_pvals['Threshold-based'][0]:.3f}, "
            f"p_dur={model_pvals['Threshold-based'][1]:.3f}, "
            f"p_int={model_pvals['Threshold-based'][2]:.3f}\n"
            f"Gaussian-based: p_shape={model_pvals['Gaussian-based'][0]:.3f}, "
            f"p_dur={model_pvals['Gaussian-based'][1]:.3f}, "
            f"p_int={model_pvals['Gaussian-based'][2]:.3f}\n"
            f"MinimumJerk: p_shape={model_pvals['MinimumJerk'][0]:.3f}, "
            f"p_dur={model_pvals['MinimumJerk'][1]:.3f}, "
            f"p_int={model_pvals['MinimumJerk'][2]:.3f}"
        )
        ax.text(0.95, 0.05, subtitle_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    custom_handles = [Patch(facecolor=color, edgecolor='black', label=mtype) 
                      for mtype, color in zip(movement_types, sns.color_palette("Set3", n_colors=len(movement_types)))]
    fig.legend(handles=custom_handles, loc='upper right', title='Modelos')
    
    fig.suptitle("Resumen Global Extendido por Métrica\n(Comparación de modelos para cada estímulo)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    out_filename = "global_summary_extended.png"
    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Resumen Global Extendido guardado en: {out_path}")



def plot_anova_significance_summary(anova_df, output_dir):
    """
    From the ANOVA results DataFrame (generated in do_significance_tests) that contains,
    for each combination of:
      - 'Dia experimental', 'body_part', 'MovementType',
      - 'Metric' (the new key, e.g. 'lat_inicio_ms') and 'Factor'
    it groups the results by 'MovementType', 'Metric' and 'Factor' and calculates the median
    of the p-value and the median number of observations.
    Then for each model (MovementType) and each metric, it generates a bar plot showing the
    median p-value for each factor, annotating above with the p-value and sample size.
    A horizontal line at p = 0.05 is drawn to indicate the significance threshold.
    """
    anova_df["PR(>F)"] = pd.to_numeric(anova_df["PR(>F)"], errors='coerce')
    
    summary = anova_df.groupby(['MovementType', 'Metric', 'Factor']).agg({
        'PR(>F)': 'median',
        'Num_observations': 'median'
    }).reset_index()
    
    # Define an order for the factors; adjust if needed.
    factor_order = {
        "C(Q('Forma del Pulso'))": 1,
        "C(Q('Duración (ms)'))": 2,
        "C(Q('Forma del Pulso')):C(Q('Duración (ms)'))": 3
    }
    summary['order'] = summary['Factor'].map(factor_order)
    summary = summary.sort_values('order')
    
    for mt in summary['MovementType'].unique():
        df_mt = summary[summary['MovementType'] == mt]
        for metric in df_mt['Metric'].unique():
            df_plot = df_mt[df_mt['Metric'] == metric].copy()
            df_plot = df_plot.sort_values('order')
            
            num_levels = len(df_plot)
            spacing = 2.5
            fig_width = max(6, num_levels * spacing)
            plt.figure(figsize=(fig_width, 4))
            
            ax = sns.barplot(x='Factor', y='PR(>F)', data=df_plot, palette='viridis')
            ax.axhline(0.05, ls='--', color='red', label='p = 0.05')
            # Get the maximum p-value, and replace non-finite values with a fallback value
            pval_max = df_plot['PR(>F)'].max()
            if not np.isfinite(pval_max):
                pval_max = 0.1  # or 1.0, depending on what makes sense in your context
            ax.set_ylim(0, pval_max * 1.15)

            ax.set_title(f"ANOVA for {metric}\nModel: {mt}")
            ax.set_ylabel("Median p-value")
            ax.set_xlabel("Factor")
            ax.legend()
            
            for patch, (_, row) in zip(ax.patches, df_plot.iterrows()):
                x_center = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                n_obs = row['Num_observations']
                star = " *" if row['PR(>F)'] < 0.05 else ""
                annotation = f"{row['PR(>F)']:.3f}{star}\n(n={int(n_obs)})"
                ax.text(x_center, y + 0.01 * ax.get_ylim()[1], annotation,
                        ha='center', va='bottom', fontsize=10)
            
            fname = f"anova_significance_{sanitize_filename(mt)}_{sanitize_filename(metric)}.png"
            out_path = os.path.join(output_dir, fname)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[Plot] ANOVA significance summary saved in: {out_path}")







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

    # Pruebas de hipótesis ANOVA + Friedman
    movement_ranges_df_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
    if not os.path.exists(movement_ranges_df_path):
        print(f"No se encontró {movement_ranges_df_path}. No se pueden hacer pruebas de hipótesis.")
        sys.exit()    

    movement_ranges_df = pd.read_csv(movement_ranges_df_path)
    do_significance_tests(movement_ranges_df, output_dir=output_comparisons_dir)

    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
    if os.path.exists(submov_path):

        submovements_df = pd.read_csv(submov_path)

        do_significance_tests_detailed(submovements_df, output_dir=output_comparisons_dir)
        plot_global_summary_with_significance_all_extended(submovements_df, output_comparisons_dir)
        plot_summary_movement_data_by_bodypart(submovements_df, output_comparisons_dir)
    else:
        print("No se encontró el archivo submovement_detailed_summary.csv para generar los resúmenes.")

    # Después de do_significance_tests, por ejemplo:
    anova_results_path = os.path.join(output_comparisons_dir, 'anova_twofactor_results.csv')
    if os.path.exists(anova_results_path):
        anova_df = pd.read_csv(anova_results_path)
        plot_anova_significance_summary(anova_df, output_comparisons_dir)
    else:
        print("No se encontraron resultados ANOVA para generar el resumen de significancia.")


    print("Proceso completo finalizado.")