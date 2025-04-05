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
    level=logging.DEBUG,  # DEBUG para información detallada
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Añadir un StreamHandler para ver los logs en la consola
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
# Reducir la verbosidad de los logs de fuentes:
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Añadir la ruta a Stimulation.py
stimulation_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\scripts\GUI_pattern_generator'
if not os.path.exists(stimulation_path):
    logging.error(f"La ruta a Stimulation.py no existe: {stimulation_path}")
    print(f"La ruta a Stimulation.py no existe: {stimulation_path}")
    sys.exit(f"La ruta a Stimulation.py no existe: {stimulation_path}")
sys.path.append(stimulation_path)
logging.info(f"Ruta añadida al PATH: {stimulation_path}")
print(f"Ruta añadida al PATH: {stimulation_path}")

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
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials'

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
verificar_archivo(stimuli_info_path, 'Stimuli_information_extended.csv')

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

print("Filtrando entradas donde 'Descartar' es 'No'")
logging.info("Filtrando entradas donde 'Descartar' es 'No'")
if 'Descartar' not in stimuli_info.columns:
    logging.error("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
    print("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
    sys.exit("La columna 'Descartar' no se encontró en 'Stimuli_information.csv'.")
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']
logging.info(f"'Stimuli_information.csv' después del filtrado tiene {len(stimuli_info)} filas.")
print(f"'Stimuli_information.csv' después del filtrado tiene {len(stimuli_info)} filas.")

print("Normalizando 'Forma del Pulso' y generando la columna 'Estímulo'")
logging.info("Normalizando 'Forma del Pulso' y generando la columna 'Estímulo'")
if 'Forma del Pulso' not in stimuli_info.columns:
    logging.error("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
    print("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
    sys.exit("La columna 'Forma del Pulso' no se encontró en 'Stimuli_information.csv'.")
stimuli_info['Forma del Pulso'] = stimuli_info['Forma del Pulso'].str.lower()
stimuli_info['Estímulo'] = stimuli_info.apply(
    lambda r: f"{r['Forma del Pulso'].strip().capitalize()}, {int(round(float(r['Duración (ms)']), 0))} ms"
              if pd.notnull(r['Forma del Pulso']) and pd.notnull(r['Duración (ms)']) else None,
    axis=1
)
logging.info("Columna 'Estímulo' creada y estandarizada.")
print("Columna 'Estímulo' creada y estandarizada.")

if stimuli_info.empty:
    logging.error("El DataFrame stimuli_info está vacío después del filtrado.")
    print("El DataFrame stimuli_info está vacío después del filtrado.")
    sys.exit("El DataFrame stimuli_info está vacío. No hay datos para procesar.")
logging.info("El DataFrame stimuli_info no está vacío después del filtrado.")
print("El DataFrame stimuli_info no está vacío después del filtrado.")

body_parts_specific_colors = {
    'Frente': 'blue',
    'Hombro': 'orange',
    'Codo': 'green',
    'Muneca': 'red',  # Reemplazar 'ñ' por 'n'
    'Braquiradial': 'grey',
    'Bicep': 'brown'
}
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

# Definimos parámetros para el suavizado de submovimientos (para modelos y panel 4)
SAVGOL_WINDOW_SUBMOV = 31
SAVGOL_POLY_SUBMOV = 3

def suavizar_velocidad_savgol(vel, window_length=SAVGOL_WINDOW_SUBMOV, polyorder=SAVGOL_POLY_SUBMOV):
    if len(vel) < window_length:
        return vel
    return savgol_filter(vel, window_length=window_length, polyorder=polyorder)

def detectar_submovimientos_en_segmento(vel_segment, threshold):
    """
    Detecta picos locales (submovimientos) dentro de un segmento de velocidad 
    usando find_peaks, asegurando duración mínima.
    """
    # Suavizar el segmento con SavGol
    vel_suav_seg = suavizar_velocidad_savgol(vel_segment)
    
    peak_indices, _ = find_peaks(vel_suav_seg, height=threshold)

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


def fit_gaussians_submovement(t_segment, v_segment, threshold, stim_start, peak_limit):
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    logging.debug(f"Inicio de fit_gaussians_submovement con peak_limit={peak_limit}")
    peaks, _ = find_peaks(v_segment, height=threshold)
    if len(peaks) == 0:
        logging.warning("No se detectaron picos en el segmento para ajustar gaussianas.")
        return []

    gaussians = []
    max_sigma = 0.5 * (t_segment[-1] - t_segment[0])
    peak_limit_used = min(peak_limit, np.percentile(v_segment, 95))
    logging.debug(f"peak_limit_used = {peak_limit_used} (np.max(v_segment) = {np.max(v_segment)})")
    
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

        # Cambiado: ahora se descarta si la gaussiana inicia antes de 20 ms después del estímulo
        if (mu_init - 2 * sigma_init) < (stim_start + 0.02):
            logging.info(f"Gaussiana descartada: inicio {mu_init - 2*sigma_init:.3f}s < {stim_start + 0.02:.3f}s")
            continue

        if 4 * sigma_init > 0.8:
            logging.info(f"Gaussiana descartada: duración {4*sigma_init:.3f}s > 0.8s")
            continue

        p0 = [A_init, mu_init, sigma_init]
        lb = [0, t_win[0], 1e-4]
        ub = [peak_limit_used, t_win[-1], max_sigma]
        try:
            popt, _ = curve_fit(gaussian, t_win, v_win, p0=p0, bounds=(lb, ub))
            modelo_pico = gaussian(mu_init, *popt)
            epsilon = 1e-6
            scale = v_segment[peak] / (modelo_pico + epsilon)
            popt[0] *= scale

            # Actualizado: permitir inicio a 20 ms en lugar de 30 ms
            if (popt[1] - 2 * popt[2]) < (stim_start + 0.02):
                logging.info(f"Gaussiana descartada tras ajuste: inicio {popt[1]-2*popt[2]:.3f}s < {stim_start+0.02:.3f}s")
                continue
            if 4 * popt[2] > 0.8:
                logging.info(f"Gaussiana descartada tras ajuste: duración {4*popt[2]:.3f}s > 0.8s")
                continue

            gaussians.append({'A_gauss': popt[0], 'mu_gauss': popt[1], 'sigma_gauss': popt[2]})
        except Exception as e:
            logging.warning(f"Fallo en curve_fit para pico en índice {peak}: {e}. Usando aproximación simple.")
            sigma_simple = sigma_init if sigma_init <= max_sigma else max_sigma
            if (mu_init - 2 * sigma_simple) < (stim_start + 0.02):
                logging.info(f"Gaussiana descartada en aproximación simple: inicio {mu_init-2*sigma_simple:.3f}s < {stim_start+0.02:.3f}s")
                continue
            if 4 * sigma_simple > 0.8:
                logging.info(f"Gaussiana descartada en aproximación simple: duración {4*sigma_simple:.3f}s > 0.8s")
                continue
            gaussians.append({'A_gauss': A_init, 'mu_gauss': mu_init, 'sigma_gauss': sigma_simple})

    # Filtrado mejorado de solapamientos: agrupar gaussianas solapadas y conservar la de mayor amplitud de cada grupo.
        # Filtrado mejorado de solapamientos: agrupar gaussianas solapadas y conservar la de mayor amplitud de cada grupo.
    gaussians = sorted(gaussians, key=lambda g: g['mu_gauss'])
    if not gaussians:  # Si la lista está vacía, retorna una lista vacía
        return []
    filtered_gaussians = []
    current_group = [gaussians[0]]
    for current in gaussians[1:]:
        last_in_group = current_group[-1]
        # Si la diferencia entre el centro de la gaussiana actual y la última del grupo es menor
        # que 0.5 veces la suma de sus sigma, la consideramos solapada.
        if abs(current['mu_gauss'] - last_in_group['mu_gauss']) < 0.5 * (last_in_group['sigma_gauss'] + current['sigma_gauss']):
            current_group.append(current)
        else:
            # Termina el grupo; conservar la gaussiana con mayor amplitud
            best = max(current_group, key=lambda g: g['A_gauss'])
            filtered_gaussians.append(best)
            current_group = [current]
    # No olvidar el último grupo
    if current_group:
        best = max(current_group, key=lambda g: g['A_gauss'])
        filtered_gaussians.append(best)
    return filtered_gaussians





def minimum_jerk_velocity(t, A, t0, T):
    """Standard minimum-jerk velocity profile."""
    if T <= 0:
        return np.zeros_like(t)
    tau = (t - t0) / T
    valid_idx = (tau >= 0) & (tau <= 1)
    v = np.zeros_like(t)
    v[valid_idx] = (30 * A / T) * (tau[valid_idx]**2) * (1 - tau[valid_idx])**2
    return v

def sum_of_minimum_jerk(t, *params):
    """Sum of multiple minimum-jerk submovements."""
    n_submovements = len(params) // 3
    v_total = np.zeros_like(t)
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        v_total += minimum_jerk_velocity(t, A, t0, T)
    return v_total

def regularized_residuals(p, t, observed_velocity, lambda_reg, A_target):
    """Compute residuals with a penalty on amplitudes."""
    residual = sum_of_minimum_jerk(t, *p) - observed_velocity
    amplitudes = p[0::3]
    penalty = np.sqrt(lambda_reg) * (amplitudes - A_target)
    return np.concatenate([residual, penalty])

def robust_fit_velocity_profile(t, observed_velocity, n_submovements,
                                n_restarts=5,
                                lambda_reg=0.1, loss='soft_l1', f_scale=0.5, peak_limit=np.inf):
    """
    Robustly fits a sum of minimum jerk profiles to an observed velocity profile.
    
    To help avoid solutions that capture only the onset (or offset), we modify the initial
    guess for the onset (t₀) and, importantly, we enforce a minimum duration for the submovement.
    Here, the lower bound for T is set as:
    
       min_T = max(0.05, 0.5 * (t[-1] - t[0]))
    
    This forces each minimum-jerk submovement to have a duration that is at least half the total
    segment duration (or 50 ms, whichever is larger).
    
    Parameters:
      t : array_like
          Time vector (seconds).
      observed_velocity : array_like
          The observed velocity profile.
      n_submovements : int
          Number of submovements to fit.
      n_restarts : int, optional
          Number of random restarts for robustness.
      lambda_reg, loss, f_scale : parameters for the least_squares call.
      peak_limit : float
          Maximum allowed amplitude.
    
    Returns:
      best_result : OptimizeResult
          The least_squares optimization result with the lowest cost.
    """
    def initial_fit(t, observed_velocity, n_submovements, lambda_reg, loss, f_scale, peak_limit):
        # Find peaks in the observed velocity.
        peak_indices, _ = find_peaks(observed_velocity, height=np.mean(observed_velocity), distance=10)
        if len(peak_indices) == 0:
            peak_times = np.array([(t[0] + t[-1]) / 2])
            peak_amplitudes = np.array([np.max(observed_velocity)])
        else:
            peak_times = t[peak_indices]
            peak_amplitudes = observed_velocity[peak_indices]
    
        # If not enough peaks, add extra guesses evenly over the time interval.
        if len(peak_amplitudes) < n_submovements:
            extras = n_submovements - len(peak_amplitudes)
            extra_times = np.linspace(t[0], t[-1], extras)
            extra_amplitudes = np.full(extras, np.max(observed_velocity) / n_submovements)
            peak_times = np.concatenate([peak_times, extra_times])
            peak_amplitudes = np.concatenate([peak_amplitudes, extra_amplitudes])
        else:
            # Choose the largest n_submovements peaks.
            top_indices = np.argsort(peak_amplitudes)[-n_submovements:]
            peak_times = peak_times[top_indices]
            peak_amplitudes = peak_amplitudes[top_indices]
    
        total_time = t[-1] - t[0]
        delta = 0.1 * total_time  # 10% of the total time.
        midpoint = (t[0] + t[-1]) / 2
        # Adjust peak times if they are too close to boundaries.
        adjusted_peak_times = np.array([
            pt if (pt >= t[0] + delta and pt <= t[-1] - delta) else midpoint
            for pt in peak_times
        ])
    
        params_init = []
        for i in range(n_submovements):
            A_init = peak_amplitudes[i]
            t0_init = adjusted_peak_times[i]
            # Use the equal split as initial guess for T.
            T_init = total_time / n_submovements
            params_init.extend([A_init, t0_init, T_init])
    
        lower_bounds = []
        upper_bounds = []
        epsilon = 1e-3
        segment_duration = total_time
        # Enforce a minimum T that is at least half the segment duration or 50 ms.
        min_T = max(0.05, 0.5 * segment_duration)
        for i in range(n_submovements):
            lower_bounds.extend([epsilon, t[0] + delta, min_T])
            upper_bounds.extend([peak_limit, t[-1] - delta, segment_duration])
    
        params_init = np.maximum(params_init, lower_bounds)
        params_init = np.minimum(params_init, upper_bounds)
    
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
            logging.error(f"Initial fitting failed: {e}")
            return None
        return result

    base_result = initial_fit(t, observed_velocity, n_submovements, lambda_reg, loss, f_scale, peak_limit)
    if base_result is None:
        logging.error("Initial fitting failed.")
        return None
    best_result = base_result
    best_cost = base_result.cost

    epsilon = 1e-3
    for i in range(n_restarts):
        perturbation = np.random.uniform(0.9, 1.1, size=len(base_result.x))
        perturbed_init = base_result.x * perturbation
        perturbed_init = np.maximum(perturbed_init, epsilon)
        try:
            result = least_squares(
                lambda p: regularized_residuals(p, t, observed_velocity, lambda_reg, A_target=np.max(observed_velocity)),
                x0=perturbed_init,
                bounds=(
                    [epsilon for _ in base_result.x],
                    [peak_limit if j % 3 == 0 else np.inf for j in range(len(base_result.x))]
                ),
                loss=loss,
                f_scale=f_scale
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception as e:
            logging.warning(f"Restart {i} failed: {e}")
            continue

    logging.info(f"Robust fit result: cost = {best_cost}")
    logging.info(f"Fitted parameters: {best_result.x}")
    return best_result

def filter_overlapping_minjerk(segments, overlap_threshold=0.5):
    """
    Filtra una lista de submovimientos MinimumJerk eliminando aquellos que se solapan en más
    del porcentaje dado (por defecto 50% de la duración del segmento más corto). Se conserva
    el segmento que tenga mayor pico (máximo valor en 'v_sm').
    """
    if not segments:
        return segments
    segments_sorted = sorted(segments, key=lambda s: s['t_segment_model'][0])
    filtered = [segments_sorted[0]]
    for seg in segments_sorted[1:]:
        last_seg = filtered[-1]
        last_start, last_end = last_seg['t_segment_model'][0], last_seg['t_segment_model'][-1]
        seg_start, seg_end = seg['t_segment_model'][0], seg['t_segment_model'][-1]
        # Calcular el solapamiento
        overlap = max(0, min(last_end, seg_end) - max(last_start, seg_start))
        # Duración del segmento más corto
        min_duration = min(last_end - last_start, seg_end - seg_start)
        if min_duration > 0 and (overlap / min_duration) > overlap_threshold:
            # Conservar el que tenga mayor pico
            if max(last_seg['v_sm']) >= max(seg['v_sm']):
                continue  # descartar seg
            else:
                filtered[-1] = seg  # reemplazar
        else:
            filtered.append(seg)
    return filtered



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
            median_vel_pre = trial.get('median_vel_pre', 0.0)
            mad_vel_pre  = trial.get('mad_vel_pre', 0.0)
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
            ax_vel.plot(t_vel, vel, color=VELOCITY_COLOR, alpha=0.8, label='Velocidad')
            ax_vel.axhline(median_vel_pre, color='lightcoral', ls='-', label=f'Median Pre={median_vel_pre:.1f}')
            ax_vel.fill_between(t_vel, median_vel_pre - mad_vel_pre, median_vel_pre + mad_vel_pre,
                                color='lightcoral', alpha=0.1, label='±1MAD Pre')
            ax_vel.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')

            ax_vel.axhline(threshold, color='k', ls='--', label='Umbral')

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
            
            # Panel 4: Velocidad observada + Modelo Minimum Jerk
            ax_submov.plot(t_vel, vel, color=VELOCITY_COLOR, label='Velocidad')
            ax_submov.axhline(threshold, color='k', ls='--', label='Umbral')
            ax_submov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Estim. window')

            # Usamos directamente los MinimumJerk filtrados ya guardados
            minjerk_submovs = trial.get('minjerk_submovs', [])
            for i, subm in enumerate(minjerk_submovs):
                t_model = subm['t_segment_model']
                v_sm = subm['v_sm']
                if i == 0:
                    label_minjerk = f"MinJerk ({len(minjerk_submovs)} submovs)"
                    ax_submov.plot(t_model, v_sm, ls=':', color='lightgreen', alpha=0.9, label=label_minjerk)
                else:
                    ax_submov.plot(t_model, v_sm, ls=':', color='lightgreen', alpha=0.9)





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
    """
    Recolecta y procesa los datos de velocidad y submovimientos a partir del CSV de información
    de estímulos (stimuli_info). Se filtran los ensayos (Descartar == 'No'), se agrupan por día y
    coordenadas, se calcula el threshold (umbral) para cada parte del cuerpo y se extrae la
    información detallada de cada ensayo. Los registros generados se usan para crear el CSV
    'expanded_trials_detailed.csv' y otros resúmenes.

    Devuelve:
       counts_df: DataFrame con resumen de movimientos (para su posterior análisis).
    """
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")

    # Cargar el CSV original y filtrar ensayos donde 'Descartar' es 'No'
    df_all = pd.read_csv(stimuli_info_path)
    logging.info(f"Archivo original cargado con {len(df_all)} filas.")
    df_filter = df_all[df_all['Descartar'] == 'No'].copy()
    logging.info(f"Ensayos a procesar (Descartar == 'No'): {len(df_filter)} filas.")

    # Calcular thresholds globales por día y parte del cuerpo
    thresholds_by_day = compute_thresholds_by_day_bodypart(df_filter)

    all_movement_data = []
    thresholds_data = []
    processed_combinations = set()
    movement_ranges_all = []
    global submovement_details  
    submovement_details = []
    expanded_trials = []  # Aquí se guardarán todos los registros detallados

    # Agrupar por día, coordenadas y distancia intracortical
    grouped_data = df_filter.groupby(['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical'])
    global_stimuli_data = {}

    for (dia_experimental, coord_x, coord_y, dist_ic), day_df in grouped_data:
        logging.info(f"Procesando: Día {dia_experimental}, X={coord_x}, Y={coord_y}, Dist={dist_ic}. Ensayos: {len(day_df)}")
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()
        if 'trial_number' not in day_df.columns:
            day_df['trial_number'] = day_df.index + 1

        # Iterar por cada parte del cuerpo
        for part in body_parts:
            current_part = part  # Valor explícito de la parte del cuerpo
            logging.info(f"Procesando: Día {dia_experimental} - Articulación: {current_part}")
            print(f"Procesando: Día {dia_experimental} - Articulación: {current_part}")
            processed_combinations.add((dia_experimental, coord_x, coord_y, dist_ic, 'All_Stimuli', current_part))

            # Obtener el threshold global para la parte actual
            threshold_data = thresholds_by_day.get((dia_experimental, current_part))
            if threshold_data is None or threshold_data['threshold'] is None:
                logging.warning(f"No se pudo calcular threshold para {current_part} en día {dia_experimental}. Se omite este body part.")
                continue
            threshold = threshold_data['threshold']
            global_median = threshold_data['median']
            global_mad = threshold_data['mad']
            logging.info(f"Threshold para {current_part} en día {dia_experimental}: {threshold:.4f}")

            thresholds_data.append({
                'body_part': current_part,
                'Dia experimental': dia_experimental,
                'Coordenada_x': coord_x,
                'Coordenada_y': coord_y,
                'threshold': threshold,
                'median_pre_stim': global_median,
                'mad_pre_stim': global_mad
            })

            key_global = (dia_experimental, current_part)
            if key_global not in global_stimuli_data:
                global_stimuli_data[key_global] = {}
            all_stimuli_data = global_stimuli_data[key_global]

            movement_trials_in_selected = 0
            trial_counter = 0

            # Obtener estímulos únicos (combinación de Forma del Pulso y Duración)
            unique_stimuli = day_df.drop_duplicates(subset=['Forma del Pulso', 'Duración (ms)'], keep='first')[['Forma del Pulso', 'Duración (ms)', 'trial_number']]
            for _, stim in unique_stimuli.iterrows():
                forma_pulso = stim['Forma del Pulso'].lower()
                duracion_ms = stim.get('Duración (ms)', None)
                stimulus_key = f"{forma_pulso.capitalize()}, {int(round(float(duracion_ms), 0))} ms"

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
                            # Usar la parte del cuerpo actual de forma explícita
                            if current_part not in velocidades or len(velocidades[current_part]) == 0:
                                continue
                            start_frame = 100
                            vel_pre = velocidades[current_part][:start_frame]
                            vel_pre = vel_pre[~np.isnan(vel_pre)]
                            if len(vel_pre) == 0:
                                logging.warning("Trial sin datos pre-stim.")
                                continue
                            if np.nanmedian(vel_pre) > global_median + 3 * global_mad:
                                logging.info("Descartando trial por exceso pre-stim robusto.")
                                continue

                            total_trials_part += 1
                            frames = np.arange(len(velocidades[current_part]))
                            amp_list, time_list = generar_estimulo_desde_parametros(
                                rowAmp['Forma del Pulso'],
                                rowAmp['Amplitud (microA)'] * 1000,
                                (duracion_ms * 1000 if duracion_ms else 1000000),
                                rowAmp['Frecuencia (Hz)'],
                                200,
                                compensar=False if duracion_ms == 1000 else True
                            )
                            stim_duration_frames = int(sum(time_list))
                            current_frame = start_frame + stim_duration_frames

                            indices_above = frames[velocidades[current_part] > threshold]
                            trial_has_movement = 1 if len(indices_above) > 0 else 0
                            movement_trials += trial_has_movement
                            maxVel = np.max(velocidades[current_part])
                            max_velocities.append(maxVel)
                            rowAmp['trial_number'] = rowAmp.get('trial_number', rowAmp.name + 1)

                    prop_movement = movement_trials / total_trials_part if total_trials_part > 0 else np.nan
                    amplitude_movement_counts[amplitude] = {
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'max_velocities': max_velocities,
                        'proportion_movement': prop_movement
                    }
                    all_movement_data.append({
                        'Ensayo_Key': rowAmp.get('Ensayo_Key', rowAmp.get('csv_filename', 'Desconocido')),
                        'Dia experimental': dia_experimental,
                        'body_part': current_part,
                        'Forma del Pulso': forma_pulso,
                        'Duración (ms)': duracion_ms,
                        'Amplitud (microA)': amplitude,
                        'movement_trials': movement_trials,
                        'total_trials': total_trials_part,
                        'proportion_movement': prop_movement,
                    })

                if not amplitude_movement_counts:
                    logging.debug(f"No hay amplitudes con movimiento para {current_part}, día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    continue

                max_proportion = max(mdata['proportion_movement'] for mdata in amplitude_movement_counts.values())
                selected_amplitudes = [amp for amp, mdata in amplitude_movement_counts.items() if mdata['proportion_movement'] == max_proportion]
                selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amplitudes)]
                print(f"Amplitudes selec. {selected_amplitudes} => prop mov={max_proportion:.2f} para {current_part}, día={dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                logging.info(f"Amplitudes selec. {selected_amplitudes} con prop mov={max_proportion:.2f}")

                max_velocities = []
                for ampSel in selected_amplitudes:
                    data_amp = amplitude_movement_counts.get(ampSel, {})
                    max_velocities.extend(data_amp.get('max_velocities', []))
                y_max_velocity = np.percentile(max_velocities, 90) if max_velocities else 50
                y_max_velocity = min(y_max_velocity, 640)
                frequencies = selected_trials['Frecuencia (Hz)'].unique()
                if len(frequencies) == 1:
                    frequency = frequencies[0]
                elif len(frequencies) > 1:
                    frequency = frequencies[0]
                    logging.warning("Múltiples frecuencias => usando la primera.")
                else:
                    frequency = None

                movement_trials_in_selected = 0
                trial_counter = 0
                movement_ranges = []
                trials_data = []

                for _, rowStim in selected_trials.iterrows():
                    csv_path = rowStim.get('csv_path')
                    if csv_path and os.path.exists(csv_path):
                        velocidades, posiciones = calcular_velocidades(csv_path)
                        if current_part in velocidades:
                            vel = velocidades[current_part]
                            pos = posiciones[current_part]
                            if len(vel) == 0:
                                continue
                            start_frame = 100
                            amp_list, duration_list = generar_estimulo_desde_parametros(
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
                            if np.nanmedian(vel_pre) > global_median + 3 * global_mad:
                                logging.info(f"Descartado trial {trial_counter} por exceso pre-stim robusto.")
                                trial_data = {
                                    'csv_filename': rowStim.get('csv_filename', 'Archivo desconocido'),
                                    'trial_index': trial_counter + 1,
                                    'start_frame': start_frame,
                                    'current_frame': current_frame,
                                    'threshold': threshold,
                                    'median_vel_pre': global_median,
                                    'mad_vel_pre': global_mad,
                                    'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                    'Dia experimental': dia_experimental,
                                    'body_part': current_part,
                                    'csv_path': csv_path,
                                    'Descartar': 'Sí'
                                }
                                trials_data.append(trial_data)
                                expanded_trials.append(trial_data)
                                trial_counter += 1
                                continue

                            if np.any(vel[start_frame:current_frame] > threshold):
                                movement_trials_in_selected += 1
                            frames_vel = np.arange(len(vel))
                            indices_above = frames_vel[vel > threshold]
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

                                    match_key = re.search(r'(\d{8}_\d{9}_\d+)', csv_path)
                                    ensayo_key = match_key.group(1) if match_key else csv_path

                                    movement_data = {
                                        'Ensayo_Key': ensayo_key,
                                        'Ensayo': trial_counter + 1,
                                        'Dia experimental': dia_experimental,
                                        'body_part': current_part,
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
                                    
                                    # Procesar submovimientos (por Gaussian y MinimumJerk)
                                    if periodo == 'Durante Estímulo':
                                        t_segment = np.arange(movement_start, movement_end + 1) / 100.0
                                        if t_segment[0] < (start_frame/100.0 + 0.03):
                                            continue
                                        vel_segment = vel[movement_start:movement_end+1]
                                        # vel_segment_filtrada = aplicar_moving_average(vel_segment, window_size=10)
                                        submov_peak_indices = detectar_submovimientos_en_segmento(vel_segment, threshold)
                                        if len(submov_peak_indices) > 0:
                                            gaussians = fit_gaussians_submovement(t_segment, vel_segment, threshold, start_frame/100.0, y_max_velocity)
                                            if gaussians:
                                                for g in gaussians:
                                                    rec = {
                                                        'Ensayo_Key': ensayo_key,
                                                        'Ensayo': trial_counter + 1,
                                                        'Dia experimental': dia_experimental,
                                                        'body_part': current_part,
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
                                            window = 5
                                            local_start = max(0, rep_peak - window)
                                            local_end = min(len(t_segment) - 1, rep_peak + window)
                                            t_local = t_segment[local_start: local_end + 1]
                                            vel_local = vel_segment[local_start: local_end + 1]
                                            if len(t_local) >= 3:
                                                result_local = robust_fit_velocity_profile(t_local, vel_local, 1, peak_limit=y_max_velocity)
                                                if result_local is not None:
                                                    params_local = result_local.x
                                                    v_minjerk_local = minimum_jerk_velocity(t_local, params_local[0], params_local[1], params_local[2])
                                                    local_peak_index = int(np.argmax(v_minjerk_local))
                                                    rec = {
                                                        'Ensayo_Key': ensayo_key,
                                                        'Ensayo': trial_counter + 1,
                                                        'Dia experimental': dia_experimental,
                                                        'body_part': current_part,
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
                                'amplitude_list': amp_list,
                                'duration_list': duration_list,
                                'start_frame': start_frame,
                                'current_frame': current_frame,
                                'threshold': threshold,
                                'median_vel_pre': global_median,
                                'mad_vel_pre': global_mad,
                                'Ensayo': trial_counter + 1,
                                'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms",
                                'csv_filename': rowStim.get('csv_filename', 'Archivo desconocido'),
                                'body_part': current_part
                            }
                            if not any(md.get('Periodo', '') == 'Durante Estímulo' for md in trial_data.get('movement_ranges', [])):
                                trial_data['Descartar'] = 'Sí'
                                logging.info(f"Trial {rowStim.get('csv_filename','Desconocido')} no tuvo modelo en el tercer panel. Marcado para descartar.")
                            else:
                                trial_data['Descartar'] = 'No'

                            trials_data.append(trial_data)
                            expanded_trials.append(trial_data)
                            trial_counter += 1

                if len(trials_data) == 0:
                    logging.debug(f"No hay datos de velocidad para {current_part} en día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    print(f"No hay datos de velocidad para {current_part} en día {dia_experimental}, {forma_pulso} {duracion_ms} ms.")
                    continue

                all_stimuli_data[stimulus_key] = {
                    'Ensayo_Key': f"{dia_experimental}_{current_part}_{stimulus_key}",
                    'Dia experimental': dia_experimental,
                    'body_part': current_part,
                    'Forma del Pulso': forma_pulso,
                    'Duración (ms)': duracion_ms,
                    'Amplitud (microA)': selected_amplitudes,
                    'movement_trials': movement_trials_in_selected,
                    'total_trials': len(trials_data),
                    'proportion_movement': movement_trials_in_selected / len(trials_data) if len(trials_data) > 0 else np.nan,
                    'y_max_velocity': y_max_velocity,
                    'threshold': threshold,
                    'median_vel_pre': global_median,
                    'mad_vel_pre': global_mad,
                    'trials_data': trials_data,
                    'Estímulo': f"{forma_pulso.capitalize()}, {duracion_ms} ms"
                }

                for stimulus_key, dataSt in all_stimuli_data.items():
                    plot_trials_side_by_side(
                        stimulus_key=stimulus_key,
                        data=dataSt,
                        body_part=current_part,
                        dia_experimental=dia_experimental,
                        output_dir=output_comparisons_dir,
                        coord_x=coord_x,
                        coord_y=coord_y,
                        dist_ic=dist_ic
                    )

    # Guardar CSV detallado de ensayos (por parte del cuerpo)
    expanded_df = pd.DataFrame(expanded_trials)
    expanded_csv_path = os.path.join(output_comparisons_dir, 'expanded_trials_detailed.csv')
    expanded_df.to_csv(expanded_csv_path, index=False)
    print(f"CSV detallado de ensayos (por parte del cuerpo) guardado en: {expanded_csv_path}")

    # Preparar counts_df a partir de all_movement_data
    counts_df = pd.DataFrame(all_movement_data)
    counts_df['Estímulo'] = counts_df.apply(
        lambda r: f"{r['Forma del Pulso'].strip().capitalize()}, {int(round(float(r['Duración (ms)']), 0))} ms",
        axis=1
    )

    if global_stimuli_data:
        stimuli_df = pd.DataFrame([entry for subdict in global_stimuli_data.values() for entry in subdict.values()])
        counts_df['Estímulo'] = counts_df['Estímulo'].str.strip().str.lower()
        stimuli_df['Estímulo'] = stimuli_df['Estímulo'].str.strip().str.lower()
        counts_df['body_part'] = counts_df['body_part'].str.strip().str.lower()
        stimuli_df['body_part'] = stimuli_df['body_part'].str.strip().str.lower()
        print("Valores de y_max_velocity:", stimuli_df['y_max_velocity'].unique())
        counts_df = pd.merge(
            counts_df,
            stimuli_df[['Dia experimental', 'body_part', 'Estímulo', 'y_max_velocity']],
            on=['Dia experimental', 'body_part', 'Estímulo'],
            how='left'
        )

    counts_path = os.path.join(output_comparisons_dir, 'movement_counts_summary.csv')
    counts_df.to_csv(counts_path, index=False)
    print(f"Datos de movimiento (con y_max_velocity) guardados en: {counts_path}")

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
        submovement_df['Submov_Num'] = submovement_df.groupby(['Ensayo_Key', 'MovementType']).cumcount() + 1
        submovement_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
        submovement_df.to_csv(submovement_path, index=False)
        print(f"Datos detallados de submovimientos guardados en: {submovement_path}")

    print("Combinaciones procesadas:")
    for combo in processed_combinations:
        print(f"Día: {combo[0]}, X={combo[1]}, Y={combo[2]}, Dist={combo[3]}, {combo[4]}, {combo[5]}")

    print("Finalizada la recopilación de datos de umbral de velocidad.")
    return counts_df





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




#########################
# ETAPA 2: PRUEBAS DE HIPÓTESIS CON LAS MÉTRICAS AGREGADAS
#########################

def do_significance_tests_aggregated(aggregated_df, output_dir=None):
    """
    Realiza pruebas de hipótesis (ANOVA, post‑hoc y Friedman) usando la tabla
    de métricas agregadas por ensayo (producto de aggregate_trial_metrics_extended).
    Se agrupa por (Dia experimental, body_part, MovementType) y para cada métrica se:
      - Ejecuta un ANOVA de dos factores (Forma del Pulso y Duración (ms) extraídas de la columna 'Estímulo')
      - Calcula el partial eta-squared para cada factor.
      - Si el p-value es significativo (< 0.05) se ejecuta un post‑hoc (Tukey).
    Los resultados se guardan en archivos CSV.
    """
    if output_dir is None:
        output_dir = output_comparisons_dir

    # Primero, extraemos los factores de 'Estímulo'
    aggregated_df['Forma del Pulso'] = aggregated_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    aggregated_df['Duración (ms)'] = aggregated_df['Estímulo'].apply(lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)

    # Definir las métricas a analizar
    metrics = [
        'lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_max_ms',
        'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
        'num_movs', 'lat_inicio_mayor_ms', 'lat_pico_mayor_ms', 'delta_valor_pico'
    ]
    grouping = ['Dia experimental', 'body_part', 'MovementType']
    results_anova = []
    results_friedman = []
    posthoc_results = []
    posthoc_signifs = []

    for (dia, bp, movtype), df_sub in aggregated_df.groupby(grouping):
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
                eta_sq = {}
                for row in anova_res.index:
                    if row == 'Residual':
                        continue
                    eta_sq[row] = calc_partial_eta_sq(anova_res, factor_row=row, resid_row='Residual')
                    results_anova.append({
                        'Dia experimental': dia,
                        'body_part': bp,
                        'MovementType': movtype,
                        'Metric': metric,
                        'Factor': row,
                        'sum_sq': anova_res.loc[row, 'sum_sq'],
                        'df': anova_res.loc[row, 'df'],
                        'F': anova_res.loc[row, 'F'],
                        'PR(>F)': anova_res.loc[row, 'PR(>F)'],
                        'Partial_Eta_Sq': eta_sq[row],
                        'Num_observations': len(data_metric)
                    })
                # Ejecutar post-hoc solo si los p-values del factor principal son significativos
                # (Aquí se pueden agregar condiciones según lo que necesites)
                # … (Se puede reutilizar la función do_posthoc_tests sobre data_metric)
            except Exception as e:
                logging.warning(f"Fallo ANOVA en {dia}, {bp}, {movtype}, {metric}: {e}")
                continue

        # (Opcional) Test de Friedman si tienes múltiples condiciones repetidas
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

    # Guardar resultados en CSV
    anova_df_out = pd.DataFrame(results_anova)
    if not anova_df_out.empty:
        anova_path = os.path.join(output_dir, 'anova_twofactor_results_aggregated.csv')
        anova_df_out.to_csv(anova_path, index=False)
        print(f"Resultados ANOVA (agregados) guardados en {anova_path}")
    else:
        print("No se generaron resultados ANOVA (agregados).")

    friedman_df = pd.DataFrame(results_friedman)
    if not friedman_df.empty:
        friedman_path = os.path.join(output_dir, 'friedman_results_aggregated.csv')
        friedman_df.to_csv(friedman_path, index=False)
        print(f"Resultados Friedman (agregados) guardados en {friedman_path}")
    else:
        print("No se generaron resultados Friedman (agregados).")

    # (El post-hoc se puede complementar si es necesario)
    
    return  # Se pueden retornar los DataFrames si se desean para posteriores gráficos







#########################
# ETAPA 1: AGREGACIÓN DE MÉTRICAS POR ENSAYO
#########################

def aggregate_trial_metrics_extended(submovements_df):
    """
    A partir del CSV detallado de submovimientos (submovement_detailed_summary),
    agrupa los submovimientos por Ensayo (y otros campos relevantes) y calcula, por cada modelo,
    las siguientes métricas (convertidas a ms o unidades convenidas):
      Para el modelo "Threshold-based":
         - lat_inicio_ms: Latencia mínima (en ms) de inicio entre los submovimientos.
         - lat_primer_pico_ms: Latencia del primer pico.
         - lat_pico_max_ms: Latencia del submov con mayor valor pico.
         - dur_total_ms: Duración total del movimiento (desde el inicio del primer submov hasta el fin del último).
         - valor_pico_inicial: Valor mínimo del pico.
         - valor_pico_max: Valor máximo del pico.
         - num_movs: Número de submovimientos.
         - lat_inicio_mayor_ms: Latencia de inicio del submov con mayor pico.
         - lat_pico_mayor_ms: Latencia del pico máximo.
         - delta_valor_pico: Diferencia entre el valor pico máximo y el mínimo.
      Se hacen cálculos análogos para "Gaussian-based" (usando mu_gauss, sigma y A_gauss)
      y para "MinimumJerk" (usando t_start, t_peak, t_end y valor_pico).

    Devuelve un DataFrame con las columnas:
       Ensayo_Key, Ensayo, Estímulo, MovementType, Dia experimental, body_part,
       lat_inicio_ms, lat_primer_pico_ms, lat_pico_max_ms, dur_total_ms,
       valor_pico_inicial, valor_pico_max, num_movs, lat_inicio_mayor_ms, lat_pico_mayor_ms, delta_valor_pico.
    """
    fs = 100.0  # frecuencia de muestreo (para convertir frames a segundos)
    
    # Definimos funciones de agregación para cada modelo
    def agg_threshold(group):
        return pd.Series({
            "lat_inicio_ms": group["Latencia al Inicio (s)"].min() * 1000,
            "lat_primer_pico_ms": group["Latencia al Pico (s)"].min() * 1000,
            "lat_pico_max_ms": group["Latencia al Pico (s)"].max() * 1000,
            "dur_total_ms": ((group["Fin Movimiento (Frame)"].max() - group["Inicio Movimiento (Frame)"].min()) / fs) * 1000,
            "valor_pico_inicial": group["Valor Pico (velocidad)"].min(),
            "valor_pico_max": group["Valor Pico (velocidad)"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": group.loc[group["Valor Pico (velocidad)"].idxmax(), "Latencia al Inicio (s)"] * 1000,
            "lat_pico_mayor_ms": group["Latencia al Pico (s)"].max() * 1000,
            "delta_valor_pico": group["Valor Pico (velocidad)"].max() - group["Valor Pico (velocidad)"].min()
        })
        
    def agg_gaussian(group):
        return pd.Series({
            "lat_inicio_ms": (group["mu_gauss"] - 2 * group["sigma_gauss"]).min() * 1000,
            "lat_primer_pico_ms": group["mu_gauss"].min() * 1000,
            "lat_pico_max_ms": group["mu_gauss"].max() * 1000,
            "dur_total_ms": (group["mu_gauss"].max() + 2 * group["sigma_gauss"].max() - (group["mu_gauss"].min() - 2 * group["sigma_gauss"].min())) * 1000,
            "valor_pico_inicial": group["A_gauss"].min(),
            "valor_pico_max": group["A_gauss"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": ((group.loc[group["A_gauss"].idxmax(), "mu_gauss"] - 2 * group.loc[group["A_gauss"].idxmax(), "sigma_gauss"]) * 1000),
            "lat_pico_mayor_ms": group["mu_gauss"].max() * 1000,
            "delta_valor_pico": group["A_gauss"].max() - group["A_gauss"].min()
        })
        
    def agg_minjerk(group):
        return pd.Series({
            "lat_inicio_ms": group["t_start"].min() * 1000,
            "lat_primer_pico_ms": group["t_peak"].min() * 1000,
            "lat_pico_max_ms": group["t_peak"].max() * 1000,
            "dur_total_ms": (group["t_end"].max() - group["t_start"].min()) * 1000,
            "valor_pico_inicial": group["valor_pico"].min(),
            "valor_pico_max": group["valor_pico"].max(),
            "num_movs": group.shape[0],
            "lat_inicio_mayor_ms": group.loc[group["valor_pico"].idxmax(), "t_start"] * 1000,
            "lat_pico_mayor_ms": group.loc[group["valor_pico"].idxmax(), "t_peak"] * 1000,
            "delta_valor_pico": group["valor_pico"].max() - group["valor_pico"].min()
        })
        
    # Los campos de agrupación (clave) son los de identificación del ensayo y condiciones:
    grouping_cols = ['Ensayo_Key', 'Ensayo', 'Estímulo', 'MovementType', 'Dia experimental', 'body_part']
    grouped = submovements_df.groupby(grouping_cols)
    
    agg_list = []
    for name, group in grouped:
        movement_type = name[3]
        if movement_type == 'Threshold-based':
            agg_metrics = agg_threshold(group)
        elif movement_type == 'Gaussian-based':
            agg_metrics = agg_gaussian(group)
        elif movement_type == 'MinimumJerk':
            agg_metrics = agg_minjerk(group)
        else:
            agg_metrics = pd.Series({k: np.nan for k in [
                "lat_inicio_ms", "lat_primer_pico_ms", "lat_pico_max_ms",
                "dur_total_ms", "valor_pico_inicial", "valor_pico_max",
                "num_movs", "lat_inicio_mayor_ms", "lat_pico_mayor_ms", "delta_valor_pico"
            ]})
        # Agregar los campos de clave al resultado
        for i, col in enumerate(grouping_cols):
            agg_metrics[col] = name[i]
        agg_list.append(agg_metrics)
    aggregated_df = pd.DataFrame(agg_list)
    return aggregated_df




# ==============================================================================
# Función para generar un resumen por Body Part (una figura por combinación Día–BodyPart)
# ==============================================================================

def plot_summary_movement_data_by_bodypart(submovements_df, output_dir):
    """
    Para cada combinación de Día experimental y body_part, genera un gráfico resumen con subplots.
    Cada subplot corresponde a una métrica y muestra, por cada estímulo, tres boxplots (Threshold-based,
    Gaussian-based y MinimumJerk) dispuestos con un pequeño offset; el color de cada grupo se asigna
    según el estímulo. Se añade un subtítulo con los p‑valores (p_shape, p_dur y p_int) por cada modelo.
    """
    # Agregar las métricas extendidas
    agg_df = aggregate_trial_metrics_extended(submovements_df)
    agg_df['Forma del Pulso'] = agg_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    agg_df['Duración (ms)'] = agg_df['Estímulo'].apply(lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)
    
    unique_days = agg_df['Dia experimental'].unique()
    unique_bodyparts = agg_df['body_part'].unique()
    
    # Definir el orden de los estímulos (se puede modificar según las necesidades)
    pulse_duration_dict = {
        'Rectangular': [500, 1000],
        'Rombo': [500, 750, 1000],
        'Rampa Ascendente': [1000],
        'Rampa Descendente': [1000],
        'Triple Rombo': [700]
    }
    ordered_stimuli = [f"{shape}, {dur} ms" for shape, durations in pulse_duration_dict.items() for dur in durations]
    
    # Fijamos el orden de los modelos y sus offsets para el gráfico
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
    
    # Por simplicidad se toma el primer día y el primer body_part
    day = list(unique_days)[0]
    body_part = list(unique_bodyparts)[0]
    df_subset = agg_df[(agg_df['Dia experimental'] == day) & (agg_df['body_part'] == body_part)]
    
    # Se asigna un color único para cada estímulo (usando Set3)
    stim_colors = dict(zip(ordered_stimuli, sns.color_palette("Set3", n_colors=len(ordered_stimuli))))
    
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 2 * 5), squeeze=False)
    # Se prepara un diccionario para ubicar el centro de cada grupo (para agregar brackets si se desea)
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
            # Para cada estímulo, se buscan los datos de cada modelo
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
        # Asignar a cada caja el color del estímulo (se determina a partir de la posición central)
        for pos, patch in zip(x_positions, bp_obj['boxes']):
            idx_grp = int(pos // gap_between)  # aproximación
            # Se determina el estímulo correspondiente buscando el grupo central más cercano
            diffs = {stim: abs((positions_by_stim[stim]['Threshold-based'] if 'Threshold-based' in positions_by_stim[stim] else positions_by_stim[stim][movement_types[0]]) - pos) for stim in positions_by_stim}
            stim_key = min(diffs, key=diffs.get)
            color = stim_colors.get(stim_key, 'lightgrey')
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Estímulo")
        ax.set_ylabel(metrics_dict[metric_key])
        ax.set_title(metrics_dict[metric_key])
        y_lims = ax.get_ylim()
        ax.set_ylim(y_lims[0], y_lims[1] + 0.35*(y_lims[1]-y_lims[0]))
        
        # Agregar como subtítulo los 9 p-values (3 por modelo)
        model_pvals = compute_model_pvals(df_subset, metric_key)
        subtitle_text = (
            f"Threshold: p_shape={model_pvals['Threshold-based'][0]:.3f}, "
            f"p_dur={model_pvals['Threshold-based'][1]:.3f}, "
            f"p_int={model_pvals['Threshold-based'][2]:.3f}\n"
            f"Gaussian: p_shape={model_pvals['Gaussian-based'][0]:.3f}, "
            f"p_dur={model_pvals['Gaussian-based'][1]:.3f}, "
            f"p_int={model_pvals['Gaussian-based'][2]:.3f}\n"
            f"MinJerk: p_shape={model_pvals['MinimumJerk'][0]:.3f}, "
            f"p_dur={model_pvals['MinimumJerk'][1]:.3f}, "
            f"p_int={model_pvals['MinimumJerk'][2]:.3f}"
        )
        ax.text(0.95, 0.05, subtitle_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        # (Opcional) Se pueden agregar brackets de significancia comparando los 3 modelos por estímulo
        for stim in positions_by_stim:
            pval_matrix = get_tukey_pvals_for_stimulus(df_subset, stim, metric_key)
            if pval_matrix is not None:
                box_positions = positions_by_stim[stim]
                # Para cada estímulo se asume que la comparación se hace entre las tres etiquetas de movimiento
                pairs = list(itertools.combinations(sorted(box_positions.keys()), 2))
                add_significance_brackets(ax, pairs, pval_matrix, box_positions,
                                          y_offset=0.1, line_height=0.05, font_size=10)
    
    # Agregar una leyenda que indique el orden de los modelos
    custom_handles = [Patch(facecolor='white', edgecolor='black', label=mt) for mt in movement_types]
    fig.legend(handles=custom_handles, loc='upper right', title='Modelos (offsets fijos)')
    
    fig.suptitle(f"{body_part} - Día {day} (Resumen por Body Part)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fname = f"summary_by_bp_{body_part}_{sanitize_filename(str(day))}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Resumen por Body Part para {body_part} - Día {day} guardado en: {out_path}")





# ==============================================================================
# Función para crear un composite de posthoc heatmaps (una figura con subplots para todas las métricas)
# ==============================================================================
# ==============================================================================
# Función para crear un composite de posthoc heatmaps (una figura con subplots para todas las métricas)
# ==============================================================================
def plot_posthoc_heatmaps_composite(posthoc_signifs_df, output_dir, movement_type, day, body_part):
    """
    Para un movimiento (movement_type), día y body_part, crea una figura compuesta
    con subplots que muestran las matrices de p-values (heatmaps) para cada métrica.
    """
    subset = posthoc_signifs_df[(posthoc_signifs_df['MovementType'] == movement_type) &
                                (posthoc_signifs_df['Dia experimental'] == day) &
                                (posthoc_signifs_df['body_part'] == body_part)]
    if subset.empty:
        print(f"No hay datos de posthoc para {movement_type}, {day}, {body_part}")
        return
    metrics = sorted(subset['Metric'].unique())
    n_cols = math.ceil(len(metrics) / 2)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols*4, 2*4), squeeze=False)
    for i, metric in enumerate(metrics):
        r = i // n_cols
        c = i % n_cols
        ax = axs[r, c]
        # Pivot: usamos las columnas 'group1', 'group2' y 'p-value'
        pivot = subset[subset['Metric'] == metric].pivot(index='group1', columns='group2', values='p-value')
        sns.heatmap(pivot, annot=True, cmap='Reds', cbar_kws={'label': 'p-value'}, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{metric}")
    plt.suptitle(f"Posthoc Heatmaps Composite\n{movement_type} - {body_part} - Día {day}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"posthoc_heatmaps_composite_{movement_type}_{body_part}_{sanitize_filename(str(day))}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Posthoc Heatmaps Composite guardado en: {out_path}")



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
    """
    Para cada modelo (Threshold-based, Gaussian-based, MinimumJerk), se calcula la ANOVA 
    con dos factores (Forma del Pulso y Duración (ms)) sobre la métrica dada y se extraen:
      - p_shape: p-value para la forma
      - p_dur: p-value para la duración
      - p_int: p-value para la interacción
    Devuelve un diccionario con 3 valores por modelo.
    """
    models = ['Threshold-based', 'Gaussian-based', 'MinimumJerk']
    pvals = {}
    for mt in models:
        df_model = agg_df[agg_df['MovementType'] == mt]
        if len(df_model) < 3:
            pvals[mt] = (np.nan, np.nan, np.nan)
        else:
            try:
                # Se asume que la columna 'Estímulo' tiene el formato "Forma, Duración ms"
                mod = ols(f"Q('{metric}') ~ C(Q('Forma del Pulso')) * C(Q('Duración (ms)'))", data=df_model).fit()
                anova_res = anova_lm(mod, typ=2)
                p_shape = anova_res.loc["C(Q('Forma del Pulso'))", "PR(>F)"] if "C(Q('Forma del Pulso'))" in anova_res.index else np.nan
                p_dur = anova_res.loc["C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Duración (ms)'))" in anova_res.index else np.nan
                p_int = anova_res.loc["C(Q('Forma del Pulso')):C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Forma del Pulso')):C(Q('Duración (ms)'))" in anova_res.index else np.nan
                pvals[mt] = (p_shape, p_dur, p_int)
            except Exception as e:
                logging.warning(f"Error en ANOVA para modelo {mt} y métrica {metric}: {e}")
                pvals[mt] = (np.nan, np.nan, np.nan)
    return pvals


#########################
# ETAPA 3: GRÁFICOS A PARTIR DE LAS MÉTRICAS AGREGADAS
#########################

def plot_global_summary_with_significance_all_extended(aggregated_df, output_dir):
    """
    Con la tabla agregada de métricas por ensayo (producto de aggregate_trial_metrics_extended),
    se generan gráficos globales (composite) que muestren, para cada métrica, boxplots
    comparando los diferentes estímulos (extraídos de 'Estímulo') para cada modelo.
    Se añaden anotaciones con los p-values obtenidos en los tests.
    """
    # Extraer factores de 'Estímulo'
    aggregated_df['Forma del Pulso'] = aggregated_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    aggregated_df['Duración (ms)'] = aggregated_df['Estímulo'].apply(lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)
    
    # Aquí se definiría el orden de los estímulos, modelos, etc.
    # Por simplicidad se puede crear un gráfico para cada modelo.
    models = aggregated_df['MovementType'].unique()
    for model in models:
        df_model = aggregated_df[aggregated_df['MovementType'] == model]
        # Crear un gráfico compuesto para todas las métricas
        metrics = ['lat_inicio_ms', 'lat_primer_pico_ms', 'lat_pico_max_ms',
                   'dur_total_ms', 'valor_pico_inicial', 'valor_pico_max',
                   'num_movs', 'delta_valor_pico']
        n_cols = math.ceil(len(metrics) / 2)
        fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 10), squeeze=False)
        for idx, metric in enumerate(metrics):
            ax = axs[idx // n_cols, idx % n_cols]
            sns.boxplot(x='Estímulo', y=metric, data=df_model, ax=ax)
            ax.set_title(f"{metric} - {model}")
            ax.tick_params(axis='x', rotation=45)
        fig.suptitle(f"Resumen Global Agregado - {model}", fontsize=16)
        out_path = os.path.join(output_dir, f"global_summary_{model}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Plot] Resumen global para {model} guardado en: {out_path}")









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
    
    '''
    # Para pruebas de hipótesis usaremos el CSV detallado de submovimientos:
    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_summary.csv')
    if os.path.exists(submov_path):
        submovements_df = pd.read_csv(submov_path)
        
        # ETAPA 1: Agregar métricas por ensayo a partir de los submovimientos detallados
        aggregated_df = aggregate_trial_metrics_extended(submovements_df)
        aggregated_df.to_csv(os.path.join(output_comparisons_dir, 'aggregated_metrics.csv'), index=False)
        print("Tabla agregada de métricas guardada en 'aggregated_metrics.csv'")
        
        # ETAPA 2: Pruebas de hipótesis (ANOVA, Friedman, etc.)
        do_significance_tests_aggregated(aggregated_df, output_dir=output_comparisons_dir)
        
        # ETAPA 3: Generar gráficos globales a partir de las métricas agregadas
        plot_global_summary_with_significance_all_extended(aggregated_df, output_comparisons_dir)
    else:
        print("No se encontró el archivo submovement_detailed_summary.csv para generar los análisis.")

    print("Proceso completo finalizado.")
    '''