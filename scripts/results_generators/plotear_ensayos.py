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
from matplotlib.lines import Line2D

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

from patsy.contrasts import Sum

from matplotlib.ticker import MultipleLocator, FuncFormatter

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
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\plot_trials4mad'

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

# ——— Constantes de “Periodo” ———
PERIODO_PRE     = 'Pre-Estímulo'
PERIODO_DURANTE = 'Durante Estímulo'
PERIODO_POST    = 'Post-Estímulo'

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
    
    peak_indices, _ = find_peaks(vel_suav_seg, height=threshold, distance = 5)

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
        ub = [min(peak_limit_used, np.max(v_win)), t_win[-1], max_sigma]

        try:
            popt, _ = curve_fit(gaussian, t_win, v_win, p0=p0, bounds=(lb, ub))
            modelo_pico = gaussian(popt[1], *popt)
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
        # que 0.7 veces la suma de sus sigma, la consideramos solapada.
        if abs(current['mu_gauss'] - last_in_group['mu_gauss']) < 0.7 * (last_in_group['sigma_gauss'] + current['sigma_gauss']):
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
    # residual de velocidad
    residual = sum_of_minimum_jerk(t, *p) - observed_velocity

    # penalización de amplitudes
    amplitudes = p[0::3]
    amp_penalty = np.sqrt(lambda_reg) * (amplitudes - A_target)

    # penalización de duraciones demasiado cortas
    durations = p[2::3]
    ideal_dur = t[-1] - t[0]
    gamma = lambda_reg * 10
    dur_penalty = np.sqrt(gamma) * (ideal_dur - durations)

    return np.concatenate([residual, amp_penalty, dur_penalty])


def robust_fit_velocity_profile(t, observed_velocity, n_submovements,
                                n_restarts=5,
                                lambda_reg=0.1, loss='soft_l1', f_scale=0.5, peak_limit=np.inf):
    def initial_fit(t, observed_velocity, n_submovements, lambda_reg, loss, f_scale, peak_limit):
        # 1) detecto picos
        peak_indices, _ = find_peaks(observed_velocity, height=np.mean(observed_velocity), distance=10)
        if len(peak_indices) == 0:
            peak_times = np.array([(t[0] + t[-1]) / 2])
            peak_amplitudes = np.array([np.max(observed_velocity)])
        else:
            peak_times = t[peak_indices]
            peak_amplitudes = observed_velocity[peak_indices]

        # 2) completo con tiempos uniformes si faltan
        if len(peak_amplitudes) < n_submovements:
            extras = n_submovements - len(peak_amplitudes)
            extra_times = np.linspace(t[0], t[-1], extras)
            extra_amps = np.full(extras, np.max(observed_velocity) / n_submovements)
            peak_times = np.concatenate([peak_times, extra_times])
            peak_amplitudes = np.concatenate([peak_amplitudes, extra_amps])
        else:
            top = np.argsort(peak_amplitudes)[-n_submovements:]
            peak_times = peak_times[top]
            peak_amplitudes = peak_amplitudes[top]

        total_time = t[-1] - t[0]
        delta = 0.1 * total_time

        # 3) cálculo de T_init por FWHM si es posible
        half_max = np.max(observed_velocity) / 2
        idx = observed_velocity >= half_max
        if idx.sum() >= 2:
            fwhm = t[idx].max() - t[idx].min()
            T_init = fwhm
        else:
            T_init = total_time / n_submovements

        # 4) bounds para T
        min_T = total_time * 0.2

        max_T = total_time * 1.2

        # 5) armo x0, lb, ub
        params_init, lb, ub = [], [], []
        for A0, t0 in zip(peak_amplitudes, peak_times):
            params_init += [A0, t0, T_init]
            lb    += [1e-3, t[0]+delta, min_T]
            ub    += [min(peak_limit, np.max(observed_velocity)), t[-1]-delta, max_T]

        # ajustar a bounds
        params_init = np.maximum(params_init, lb)
        params_init = np.minimum(params_init, ub)

        # 6) llamada a least_squares
        return least_squares(
            lambda p: regularized_residuals(p, t, observed_velocity, lambda_reg, np.max(observed_velocity)),
            x0=params_init,
            bounds=(lb, ub),
            loss=loss,
            f_scale=f_scale
        )

    # primer ajuste
    base = initial_fit(t, observed_velocity, n_submovements, lambda_reg, loss, f_scale, peak_limit)
    best, best_cost = base, base.cost

    # re-arranques
    for _ in range(n_restarts):
        perturb = base.x * np.random.uniform(0.9, 1.1, size=base.x.size)
        perturb = np.maximum(perturb, 1e-3)
        try:
            res = least_squares(
                lambda p: regularized_residuals(p, t, observed_velocity, lambda_reg, np.max(observed_velocity)),
                x0=perturb,
                bounds=([1e-3]*base.x.size, [peak_limit if i%3==0 else np.inf for i in range(base.x.size)]),
                loss=loss,
                f_scale=f_scale
            )
            if res.cost < best_cost:
                best, best_cost = res, res.cost
        except:
            pass

    return best


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

def configurar_eje_ms(ax, max_time):
    """
    Eje-X en ms, ticks mayores cada 500 ms y menores cada 250 ms,
    pero sólo etiqueta los mayores que sean múltiplos de 1000 ms.
    """
    ax.set_xlim(0, max_time)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))    # 500 ms
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))   # 250 ms
    def fmt(x, pos):
        ms = int(x*1000)
        return str(ms) if (ms % 1000)==0 else ""
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.tick_params(axis='x', which='major', length=7)
    ax.tick_params(axis='x', which='minor', length=4)
    ax.set_xlabel('Tiempo (ms)')


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

    Ahora se incluyen también los ensayos descartados. En la visualización se marca
    con "(Descartado)" en el título del panel 1 y se pueden aplicar otros cambios de estilo.
    """
    trials_data = data.get('trials_data', [])
    if not trials_data:
        logging.warning(
            f"No hay datos de ensayos (trials_data) para {stimulus_key} en {body_part} día {dia_experimental}"
        )
        return

    max_per_figure = 15
    fs = 100.0

    # 1) Ordenar los ensayos según el índice (u otro criterio que prefieras)
    trials_data = sorted(trials_data, key=lambda d: d.get('trial_index', 0))

    # 2) Dividir en batches de tamaño max_per_figure
    batches = [
        trials_data[i:i + max_per_figure]
        for i in range(0, len(trials_data), max_per_figure)
    ]

    day_str  = str(dia_experimental).replace('/', '-')
    dist_str = f"_Dist_{dist_ic}" if dist_ic is not None else ""

    # 3) Iterar sobre cada batch y numerarlo
    for batch_num, subset in enumerate(batches, start=1):
        # Calcular máximos para ejes
        # calcular máximos para ejes
        max_time = 0
        max_vel  = 0

        for td in subset:
            vel_ensayo = td.get('velocity', [])
            # sólo si viene array/lista no vacía
            if vel_ensayo is not None and len(vel_ensayo) > 0:
                max_time = max(max_time, len(vel_ensayo) / fs)
                max_vel  = max(max_vel, np.nanmax(vel_ensayo))

        fig_height = 25
        fig_width  = len(subset) * 5
        height_ratios = [2, 2, 2, 0.5, 2]
        fig, axes = plt.subplots(
            5, len(subset),
            figsize=(fig_width, fig_height),
            sharey=False,
            gridspec_kw={'height_ratios': height_ratios}
        )
        if len(subset) == 1:
            axes = axes.reshape(5, 1)

        # Aplicamos la configuración de ejes X‐ms a todos los subplots
        for row in axes:
            for ax in row:
                configurar_eje_ms(ax, max_time)

        for idx_col, trial in enumerate(subset):
            ax_disp   = axes[0, idx_col]
            ax_vel    = axes[1, idx_col]
            ax_submov = axes[2, idx_col]   # panel DETALLE de submovimientos
            ax_mov    = axes[3, idx_col]   # panel RANGOS de movimiento
            ax_stim   = axes[4, idx_col]


            vel = trial.get('velocity', [])
            pos = trial.get('positions', {'x': [], 'y': []})
            #submovements = trial.get('submovements', [])
            mov_ranges   = trial.get('movement_ranges', [])
            subm = trial['submovements']
            th_segments = [m for m in mov_ranges if m['Periodo']==PERIODO_DURANTE]
            gauss_list  = [g for s in subm if s['MovementType']=='Gaussian-based' and s['Periodo']==PERIODO_DURANTE
                        for g in s['gaussians']]
            mj_list     = [(s['t_segment_model'], s['v_sm']) for s in subm
                        if s['MovementType']=='MinimumJerk' and s['Periodo']==PERIODO_DURANTE]

            threshold      = trial.get('threshold_refined',
                                      trial.get('threshold', 0.0))
            median_vel_pre = trial.get('median_vel_pre_refined',
                                      trial.get('median_vel_pre', 0.0))
            mad_vel_pre    = trial.get('mad_vel_pre_refined',
                                      trial.get('mad_vel_pre', 0.0))

            start_frame   = trial.get('start_frame', 100)
            current_frame = trial.get('current_frame', 200)
            amplitude_list = trial.get('amplitude_list', [])
            duration_list  = trial.get('duration_list', [])
            
            frames_vel = np.arange(len(vel))
            t_vel      = frames_vel / fs


            stim_start_s = start_frame / fs
            stim_end_s   = current_frame / fs

            # Detectamos si el ensayo fue descartado
            is_discarded = trial.get('Descartar', 'No') == 'Sí'
            motivo_descarte = trial.get('motivo_descarte', '')

            # Panel 1: Desplazamiento
            if len(pos['x']) == len(pos['y']) and len(pos['x']) > 0:
                displacement = np.sqrt((pos['x'] - pos['x'][0])**2 + (pos['y'] - pos['y'][0])**2)
                t_disp = np.arange(len(displacement)) / fs
                ax_disp.plot(t_disp, displacement, color='blue', label='Desplaz. (px)')
                if len(displacement) > 0:
                    ax_disp.set_ylim(0, np.max(displacement) + 10)
            ax_disp.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Ventana Estím.')
            ax_disp.set_ylabel('Desplaz. (px)')
            ax_disp.set_xlabel('Tiempo (ms)')
            ax_disp.set_xlim(0, max_time)
            csv_name = trial.get('csv_filename', 'Archivo desconocido')
            if csv_name != 'Archivo desconocido':
                m = re.search(r'(\d{8}_\d{9}_\d+)', csv_name)
                key = m.group(1) if m else csv_name
            else:
                key = csv_name
            # Se añade la etiqueta "(Descartado)" si corresponde, y si hay motivo se puede incluir
            title_text = f"Ensayo {trial.get('trial_index', 0) + 1}\n{key}"
            if is_discarded:
                title_text += "\n(Descartado" + (f": {motivo_descarte}" if motivo_descarte else "") + ")"
            ax_disp.set_title(title_text)

            ax_disp.legend(fontsize=7, loc='upper left')


            # Panel 2: Velocidad + Umbral
            ax_vel.plot(t_vel, vel, color=VELOCITY_COLOR, alpha=0.8, label='Velocidad')
            ax_vel.axhline(median_vel_pre, color='lightcoral', ls='-',label=f'Mediana Pre-Estím.={median_vel_pre:.1f}')
            ax_vel.fill_between(t_vel, median_vel_pre - mad_vel_pre, median_vel_pre + mad_vel_pre,
                                color='lightcoral', alpha=0.1, label='±1MAD Pre-Estím')
            ax_vel.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1, label='Ventana Estím.')
            ax_vel.axhline(threshold, color='k', ls='--', label='Umbral')
            ax_vel.axhline(median_vel_pre + 4 * mad_vel_pre, color='orange', ls='-.', label='Descartar (4*MAD)')
            ax_vel.set_xlabel('Tiempo (ms)')
            ax_vel.set_ylabel('Vel. (px/s)')
            ax_vel.set_xlim(0, max_time)
            ax_vel.set_ylim(0, max_vel + 5)
            ax_vel.legend(fontsize=7, loc='upper left')

            # --- antes del panel 3 y 4 ----
            # Extraemos todas las gaussianas "Durante Estimulo"
            gauss_list = []
            gauss_list = [g for s in subm if s['MovementType']=='Gaussian-based' and s['Periodo']==PERIODO_DURANTE for g in s['gaussians']]
            mj_list = [(s['t_segment_model'], s['v_sm']) for s in subm if s['MovementType']=='MinimumJerk' and s['Periodo']==PERIODO_DURANTE]
            n_gauss = len(gauss_list); n_mj = len(mj_list)

            # Colores consistentes
            TH_COLOR = 'red'
            MJ_COLOR = 'lightgreen'
            GAUSS_COLORS = [GAUSSIAN_PALETTE[i % len(GAUSSIAN_PALETTE)] for i in range(n_gauss)]


            


            # ——— Panel 4: Detalle de submovimientos ———
            ax_submov.plot(t_vel, vel, color=VELOCITY_COLOR, lw=1, label='Velocidad')
            ax_submov.axhline(threshold, color='k', lw=1, ls='--', label='Umbral')
            ax_submov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # Points de Threshold peak
            for mv in th_segments:
                i0, i1 = mv['Inicio Movimiento (Frame)'], mv['Fin Movimiento (Frame)']
                seg = vel[i0:i1+1]
                if seg.size:
                    rel = np.argmax(seg)
                    abs_pk = i0 + rel
                    t_pk, v_pk = abs_pk/fs, seg[rel]
                    #ax_submov.plot(t_pk, v_pk, 'o', color=TH_COLOR, ms=6, label='Threshold peak')
                    #ax_submov.text(t_pk, v_pk + 2, f"{v_pk:.1f}", ha='center', va='bottom', fontsize=6)

            # Gaussian curves (más gruesas)
            for i, g in enumerate(gauss_list):
                xs = np.linspace(g['mu_gauss']-3*g['sigma_gauss'], g['mu_gauss']+3*g['sigma_gauss'], 200)
                curve = gaussian(xs, g['A_gauss'], g['mu_gauss'], g['sigma_gauss'])
                ax_submov.plot(xs, curve, ls='--', lw=2, color=GAUSS_COLORS[i], label='Gaussianas')

            # MinimumJerk curves (más gruesas)
            for i, (t_seg, v_seg) in enumerate(mj_list):
                ax_submov.plot(t_seg, v_seg, ls=':', lw=2, color=MJ_COLOR, label='MinimumJerk')

            ax_submov.set(xlim=(0, max_time), ylim=(0, max_vel+5))
            ax_submov.set_xlabel('Tiempo (ms)')
            ax_submov.set_ylabel('Vel. (px/s)')

            legend_elems = [
                Line2D([0],[0], color=VELOCITY_COLOR, lw=1,      label='Velocidad'),
                Line2D([0],[0], color='k',           lw=1, ls='--', label='Umbral')
                #,Line2D([0],[0], color=TH_COLOR,      marker='o', ls='', label='Threshold peak')

            ]
            if GAUSS_COLORS:
                legend_elems.append(
                    Line2D([0],[0], color=GAUSS_COLORS[0], lw=2, ls='--', label='Gaussianas')
                )
            if mj_list:
                legend_elems.append(
                    Line2D([0],[0], color=MJ_COLOR, lw=2, ls=':', label='MinimumJerk')
                )
            ax_submov.legend(handles=legend_elems, fontsize=8, loc='upper left')

            # ——— Panel 3: Rangos de movimiento ———
            ax_mov = ax_mov

            # recortamos el eje Y a la mitad inferior (0–0.5)
            ax_mov.set(xlim=(0, max_time), ylim=(0.0, 0.5))
            ax_mov.axvspan(stim_start_s, stim_end_s, color='green', alpha=0.1)

            # extraemos los segmentos
  #          th_segments = [m for m in mov_ranges if m['Periodo']=='Durante Estimulo']
 #           gauss_list  = [g for s in subm if s['MovementType']=='Gaussian-based' and s['Periodo']=='Durante Estimulo' for g in s['gaussians']]
#            mj_list     = [(s['t_segment_model'], s['v_sm']) for s in subm if s['MovementType']=='MinimumJerk' and s['Periodo']=='Durante Estimulo']

            # definimos niveles bien separados en la mitad baja
            y_levels = {
                'Threshold-based': 0.4,
                'Gaussian-based':  0.3,
                'MinimumJerk':      0.2
            }

            # dibujar Threshold-based en y=0.4
            for mv in th_segments:
                x0, x1 = mv['Inicio Movimiento (Frame)']/fs, mv['Fin Movimiento (Frame)']/fs
                y0 = y_levels['Threshold-based']
                ax_mov.hlines(y0, x0, x1, color=TH_COLOR, lw=2)
                abs_pk = mv['Inicio Movimiento (Frame)'] + np.argmax(vel[mv['Inicio Movimiento (Frame)']:mv['Fin Movimiento (Frame)']+1])
                t_pk, v_pk = abs_pk/fs, mv['Valor Pico (velocidad)']
                ax_mov.plot(t_pk, y0, 'o', color=TH_COLOR, ms=6)
                ax_mov.text(t_pk, y0 - 0.02, f"{v_pk:.1f}", ha='center', va='bottom', fontsize=4)

            # dibujar Gaussian-based en y=0.3
            for i, g in enumerate(gauss_list):
                y1 = y_levels['Gaussian-based']
                L, R = g['mu_gauss']-2*g['sigma_gauss'], g['mu_gauss']+2*g['sigma_gauss']
                ax_mov.hlines(y1, L, R, lw=2, color=GAUSS_COLORS[i])
                ax_mov.plot(g['mu_gauss'], y1, 'o', color=GAUSS_COLORS[i], ms=6)
                ax_mov.text(g['mu_gauss'], y1 - 0.02, f"{g['A_gauss']:.1f}", ha='center', va='bottom', fontsize=4)

            # dibujar MinimumJerk en y=0.2
            for t_seg, v_seg in mj_list:
                y2 = y_levels['MinimumJerk']
                ax_mov.hlines(y2, t_seg[0], t_seg[-1], lw=2, color=MJ_COLOR)
                rel = np.argmax(v_seg)
                t_pk = t_seg[rel]
                ax_mov.plot(t_pk, y2, 'o', color=MJ_COLOR, ms=6)
                ax_mov.text(t_pk, y2 - 0.02, f"{v_seg.max():.1f}", ha='center', va='bottom', fontsize=4)

            ax_mov.set_yticks([])
            ax_mov.set_ylabel('Rangos de mov.')


            # construir leyenda sólo si hay algo
            legend_handles = []
            if th_segments:
                legend_handles.append(Line2D([0],[0], color=TH_COLOR, lw=2, label=f"Seg Umbral ({len(th_segments)})"))
            if gauss_list:
                legend_handles.append(Line2D([0],[0], color=GAUSS_COLORS[0], lw=2, label=f"Seg Gauss ({len(gauss_list)})"))
            if mj_list:
                legend_handles.append(Line2D([0],[0], color=MJ_COLOR, lw=2, label=f"Seg MinJerk ({len(mj_list)})"))

            if legend_handles:
                ax_mov.legend(handles=legend_handles, fontsize=6, loc='upper left')

            # Panel 5: Perfil del Estímulo
            ax_stim.set_xlabel('Tiempo (ms)')
            ax_stim.set_ylabel('Amplitud (µA)')
            ax_stim.set_xlim(0, max_time)
            ax_stim.set_ylim(-170, 170)
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
                ax_stim.step(x_vals, y_vals, color='purple', where='pre', linewidth=0.05, label='Estímulo')

        fig.suptitle(
            f"Día {dia_experimental}, {body_part}, {stimulus_key} - Grupo {batch_num}/{len(batches)}",
            fontsize=12
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        out_filename = (
            f"Dia_{day_str}_{body_part}_{stimulus_key}"
            f"{dist_str}_Group_{batch_num}.png"
        )
        out_path = os.path.join(output_dir, out_filename)
        if os.path.exists(out_path):
            os.remove(out_path)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')

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


def procesar_trial(row, current_part, duracion_ms, global_median, global_mad, threshold, frequency, default_start_frame=100):
    """
    Procesa un ensayo individual y retorna un diccionario con toda la información relevante,
    incluyendo un trial_id único y la bandera 'Descartar' (con motivo si se descarta).
    
    Parámetros:
      row: una fila del DataFrame de ensayos (del CSV de estímulos)
      current_part: cadena con la parte del cuerpo a procesar (ej. 'Codo')
      duracion_ms: duración en ms asociada al estímulo de la fila
      global_median, global_mad: valores calculados globalmente para el pre-estímulo
      threshold: umbral de velocidad para detectar movimiento
      frequency: frecuencia asociada al estímulo
      default_start_frame: número de frame que se asume como inicio del estímulo (por defecto 100)
    
    Retorna:
      Un diccionario con la información completa del ensayo, o None si no se pueden procesar los datos.
    """
    csv_path = row.get('csv_path')
    if not (csv_path and os.path.exists(csv_path)):
        logging.warning(f"CSV path no existe para el ensayo: {row}")
        return None

    # Calcular velocidades y posiciones para la parte del cuerpo
    velocidades, posiciones = calcular_velocidades(csv_path)
    if current_part not in velocidades or len(velocidades[current_part]) == 0:
        logging.warning(f"No hay datos de velocidad para {current_part} en el archivo {csv_path}")
        return None
    vel = velocidades[current_part]
    pos = posiciones[current_part]

    # Definir el frame de inicio y calcular la velocidad pre-estímulo
    start_frame = default_start_frame
    vel_pre = vel[:start_frame]
    vel_pre = vel_pre[~np.isnan(vel_pre)]
    if len(vel_pre) == 0:
        logging.warning("Ensayo sin datos pre-estímulo")
        return None

    # Calcular la lista de amplitudes y duración del estímulo usando la función existente
    # (Esta función ya incluye la conversión de tiempo: us_to_frames)
    amp_list, duration_list = generar_estimulo_desde_parametros(
        row['Forma del Pulso'],
        row['Amplitud (microA)'],
        (duracion_ms * 1000 if duracion_ms else 1000000),
        row['Frecuencia (Hz)'] if frequency is None else frequency,
        200,
        compensar=False if duracion_ms == 1000 else True
    )
    current_frame = int(start_frame + sum(duration_list))

    # Definir un trial id único (por ejemplo, combinando el nombre del archivo y el trial_number)
    trial_num = row.get('trial_number', row.name)
    trial_id = f"{row.get('csv_filename', 'ArchivoDesconocido')}_{trial_num}"

    # Armar el diccionario con la información del ensayo
    trial_data = {
        'trial_id': trial_id,
        'csv_filename': row.get('csv_filename', 'Archivo desconocido'),
        'trial_index': trial_num,
        'start_frame': start_frame,
        'current_frame': current_frame,
        'threshold': threshold,
        'median_vel_pre': global_median,
        'mad_vel_pre': global_mad,
        'Estímulo': row['Estímulo'],
        'Dia experimental': row.get('Dia experimental', None),
        'body_part': current_part,
        'csv_path': csv_path,
        'amplitude_list': amp_list,
        'duration_list': duration_list,
        'velocity': vel,
        'positions': pos
    }

    # Evaluar si se descarta el ensayo según la mediana pre-estímulo
    if np.nanmedian(vel_pre) > global_median + 4 * global_mad:
        trial_data['Descartar'] = 'Sí'
        trial_data['motivo_descarte'] = 'Exceso de velocidad pre-estímulo (4*MAD)'
    else:
        trial_data['Descartar'] = 'No'

    return trial_data

def detect_movements_and_submovements(trial):
    """
    Detección de segmentos y submovimientos sobre trial['velocity'],
    usando trial['threshold_refined'], trial['start_frame'], trial['current_frame'].
    Rellena trial['movement_ranges'] y trial['submovements'] con:
      - Threshold-based segments
      - Gaussian-based submovements (fit_gaussians_submovement)
      - MinimumJerk submovements (robust_fit_velocity_profile)
    """
    trial['movement_ranges'] = []
    trial['submovements']    = []

    vel     = trial['velocity']
    th      = trial['threshold_refined']
    start   = trial['start_frame']
    end     = trial['current_frame']
    fs      = 100.0

    movement_ranges = []
    submovements    = []

    frames = np.arange(len(vel))
    above  = frames[vel > th]
    if len(above) == 0:
        trial['movement_ranges'] = movement_ranges
        trial['submovements']    = submovements
        return

    # 1) Detectar segmentos contínuos por threshold
    segments = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
    for seg in segments:
        i0, i1 = seg[0], seg[-1]
        # descartar muy cortos / arranque antes de start+3
        if (i1 - i0) < 3 or i0 < start + 3:
            continue

        # determinar periodo
        if   i1 < start:           periodo = PERIODO_PRE
        elif i0 <= end:            periodo = PERIODO_DURANTE
        else:                      periodo = PERIODO_POST

        # 1.a) Threshold‑based: tomar el pico
        rel_pk     = np.argmax(vel[i0:i1+1])
        abs_pk     = i0 + rel_pk
        peak_vel   = float(vel[abs_pk])
        peak_lat   = (abs_pk - start) / fs

        movement_ranges.append({
            'MovementType': 'Threshold-based',
            'Periodo': periodo,
            'Inicio Movimiento (Frame)': i0,
            'Fin Movimiento (Frame)':    i1,
            'Latencia al Inicio (s)':     (i0 - start)/fs,
            'Latencia al Pico (s)':       peak_lat,
            'Valor Pico (velocidad)':     peak_vel,
            'Duración Total (s)':         (i1 - i0)/fs,
            'Duración durante Estímulo (s)': 
                max(0, min(i1, end) - max(i0, start))/fs
        })

        # 2) Gaussian‑based submovements
        t_seg = np.arange(i0, i1+1) / fs
        v_seg = vel[i0:i1+1]
        # solo si hay algo sobre th
        gauss = fit_gaussians_submovement(
            t_seg, v_seg, th, start/fs,
            peak_limit = np.percentile(vel, 90)  # o y_max_velocity si lo tienes
        )
        # 2) Gaussian‑based submovements
        if gauss:
            submovements.append({
                'MovementType': 'Gaussian-based',
                'Periodo': periodo,
                'gaussians': gauss
            })

        # 3) MinimumJerk submovements
        peaks = detectar_submovimientos_en_segmento(v_seg, th)
        mj_candidates = []
        for pk in peaks:
            w0 = max(0, pk-5)
            w1 = min(len(v_seg)-1, pk+5)
            t_loc = t_seg[w0:w1+1]
            v_loc = v_seg[w0:w1+1]
            if len(t_loc) < 3:
                continue
            res = robust_fit_velocity_profile(
                t_loc, v_loc, n_submovements=1,
                peak_limit=np.percentile(vel,90)
            )
            if res is not None:
                A, t0, T = res.x
                v_mj = minimum_jerk_velocity(t_loc, A, t0, T)
                mj_candidates.append({
                'MovementType': 'MinimumJerk',
                'Periodo': periodo,
                't_segment_model': t_loc,
                'v_sm': v_mj
            })
        if mj_candidates:
            # 1) extraer centro, duración y amplitud
            mj_info = []
            for sm in mj_candidates:
                t_loc = sm['t_segment_model']
                v_sm  = sm['v_sm']
                rel   = np.argmax(v_sm)
                t_peak = t_loc[rel]
                dur    = t_loc[-1] - t_loc[0]
                amp    = v_sm.max()
                mj_info.append((t_peak, dur, amp, sm))

            #  si no hay nada tras esa extracción, salto todo
            if not mj_info:
                continue

            # 2) agrupar solapados y conservar el de mayor amplitud
            mj_info.sort(key=lambda x: x[0])
            filtered_mj = []
            group = [mj_info[0]]
            for info in mj_info[1:]:
                last = group[-1]
                if abs(info[0] - last[0]) < 0.7 * (info[1] + last[1]):
                    group.append(info)
                else:
                    best = max(group, key=lambda x: x[2])
                    filtered_mj.append(best[3])
                    group = [info]
            # no olvidar el último grupo
            best = max(group, key=lambda x: x[2])
            filtered_mj.append(best[3])

            # 3) si tras filtrar no hay candidatos, salto
            if not filtered_mj:
                continue

            # 4) sólo añado los submovimientos válidos
            for sm in filtered_mj:
                submovements.append(sm)

        trial['movement_ranges'] = movement_ranges
        trial['submovements']    = submovements

# =====================================================
# Función principal: recopilar datos de velocidad y submovimientos
# =====================================================
def collect_velocity_threshold_data():
    """
    Recolecta y procesa los datos de velocidad y submovimientos a partir del CSV de información
    de estímulos (stimuli_info). Se filtran los ensayos (Descartar == 'No'), se agrupan por día y
    coordenadas, se calcula el threshold para cada parte del cuerpo y se extrae la información detallada
    de cada ensayo.

    Para cada ensayo se utiliza la función 'procesar_trial' para asignar un trial_id y obtener
    los datos básicos (velocity, positions, lista de amplitudes y tiempos del estímulo, etc.).
    Posteriormente se detectan los segmentos (submovimientos de tipo Threshold-based) y se procesan
    mediante ajustes Gaussian-based y MinimumJerk, siempre usando el umbral **refinado**.

    Devuelve:
        counts_df: DataFrame con resumen de movimientos.
    """
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")

    # Copia inicial de stimuli_info
    df_filter = stimuli_info.copy()

    # 1) Calcular thresholds globales por día y parte del cuerpo
    thresholds_by_day = compute_thresholds_by_day_bodypart(df_filter)

    # Contenedores
    all_movement_data       = []
    thresholds_dict         = {}
    processed_combinations  = set()
    movement_ranges_all     = []
    submovement_details_all = []
    submovement_details_valid = []
    expanded_trials         = []
    global_stimuli_data     = {}

    # 2) Agrupar por día, coordenadas y distancia intracortical
    grouped_data = df_filter.groupby(
        ['Dia experimental', 'Coordenada_x', 'Coordenada_y', 'Distancia Intracortical']
    )

    # 3) Iterar sobre cada grupo
    for (dia_exp, x_coord, y_coord, dist_ic), day_df in grouped_data:
        day_df = day_df.copy()
        day_df['Forma del Pulso'] = day_df['Forma del Pulso'].str.lower()
        if 'trial_number' not in day_df.columns:
            day_df['trial_number'] = day_df.index + 1

        for part in body_parts:
            current_part = part
            processed_combinations.add((dia_exp, x_coord, y_coord, dist_ic, current_part))

            # Threshold global
            thr_data = thresholds_by_day.get((dia_exp, current_part), {})
            threshold      = thr_data.get('threshold')
            global_median  = thr_data.get('median')
            global_mad     = thr_data.get('mad')
            if threshold is None:
                logging.warning(f"No hay threshold para {current_part} en día {dia_exp}, se omite.")
                continue

            thresholds_dict[(dia_exp, x_coord, y_coord, part)] = {
                'body_part': part,
                'Dia experimental': dia_exp,
                'Coordenada_x': x_coord,
                'Coordenada_y': y_coord,
                'threshold_original': threshold,
                'median_pre_original': global_median,
                'mad_pre_original': global_mad,
                'threshold_refined': np.nan,
                'median_pre_refined': np.nan,
                'mad_pre_refined': np.nan
            }

            # Preparar dict para graficar
            key_global = (dia_exp, current_part)
            global_stimuli_data.setdefault(key_global, {})
            all_stimuli_data = global_stimuli_data[key_global]

            # Identificar estímulos
            unique_stimuli = day_df.drop_duplicates(
                subset=['Forma del Pulso', 'Duración (ms)'],
                keep='first'
            )[['Forma del Pulso', 'Duración (ms)']]

            for _, stim in unique_stimuli.iterrows():
                forma_pulso = stim['Forma del Pulso']
                duracion_ms = stim['Duración (ms)']
                stimulus_key = f"{forma_pulso.capitalize()}, {int(round(float(duracion_ms),0))} ms"

                stim_df = day_df[
                    (day_df['Forma del Pulso'] == forma_pulso) &
                    (day_df['Duración (ms)'] == duracion_ms)
                ]
                if stim_df.empty:
                    continue

                # 4) Elegir amplitud con mayor proporción de movimiento
                amplitude_movement_counts = {}
                for amp, amp_trials in stim_df.groupby('Amplitud (microA)'):
                    mov_count = total_trials = 0
                    max_vels = []
                    for _, rowAmp in amp_trials.iterrows():
                        csv_path = rowAmp.get('csv_path')
                        if not (csv_path and os.path.exists(csv_path)):
                            continue
                        velocidades, _ = calcular_velocidades(csv_path)
                        v = velocidades.get(current_part, np.array([]))
                        if v.size == 0:
                            continue
                        pre = v[:100]
                        pre = pre[~np.isnan(pre)]
                        if pre.size == 0 or np.nanmedian(pre) > global_median + 3*global_mad:
                            continue
                        total_trials += 1
                        if (v > threshold).any():
                            mov_count += 1
                        max_vels.append(np.nanmax(v))
                    prop = mov_count/total_trials if total_trials else np.nan
                    amplitude_movement_counts[amp] = {
                        'movement_trials': mov_count,
                        'total_trials': total_trials,
                        'max_velocities': max_vels,
                        'proportion_movement': prop
                    }
                    all_movement_data.append({
                        'Dia experimental': dia_exp,
                        'body_part': current_part,
                        'Forma del Pulso': forma_pulso,
                        'Duración (ms)': duracion_ms,
                        'Amplitud (microA)': amp,
                        'movement_trials': mov_count,
                        'total_trials': total_trials,
                        'proportion_movement': prop
                    })

                if not amplitude_movement_counts:
                    continue

                max_prop = max(d['proportion_movement'] for d in amplitude_movement_counts.values())
                selected_amps = [amp for amp,d in amplitude_movement_counts.items() if d['proportion_movement']==max_prop]
                selected_trials = stim_df[stim_df['Amplitud (microA)'].isin(selected_amps)]

                # 5) Procesar cada trial seleccionado
                trials_data = []
                for _, rowStim in selected_trials.iterrows():
                    td = procesar_trial(
                        rowStim, current_part, duracion_ms,
                        global_median, global_mad, threshold, None
                    )
                    if td:
                        trials_data.append(td)
                        expanded_trials.append(td)
                if not trials_data:
                    continue

                # 6) Refinar umbral con pre-estímulo
                pre_vals = []
                for td in trials_data:
                    if td['Descartar']=='No':
                        pv = td['velocity'][:td['start_frame']]
                        pre_vals.extend(pv[~np.isnan(pv)])
                if len(pre_vals)>=10:
                    med_v = np.median(pre_vals)
                    mad_v = np.median(np.abs(pre_vals-med_v))
                    thr_ref = med_v + 2*mad_v
                else:
                    med_v, mad_v, thr_ref = global_median, global_mad, threshold

                # 7) Re-detectar usando umbral refinado
                for td in trials_data:
                    td['threshold_refined']      = thr_ref
                    td['median_vel_pre_refined'] = med_v
                    td['mad_vel_pre_refined']    = mad_v
                    detect_movements_and_submovements(td)

                    # Volcar en resumen
                    for mv in td.get('movement_ranges', []):
                        if mv.get('Periodo')!=PERIODO_DURANTE: 
                            continue
                        movement_ranges_all.append({
                            **mv,
                            'Ensayo_Key': td['trial_id'],
                            'Dia experimental': td['Dia experimental'],
                            'body_part': td['body_part'],
                            'Estímulo': td['Estímulo'],
                            'Coordenada_x': x_coord,
                            'Coordenada_y': y_coord,
                            'Distancia Intracortical': dist_ic
                        })
                    for sm in td.get('submovements', []):
                        if sm.get('Periodo')!=PERIODO_DURANTE:
                            continue
                        if sm['MovementType'] == 'Gaussian-based':
                            for g in sm['gaussians']:
                                entry = {
                                    'Ensayo_Key':              td['trial_id'],
                                    'Dia experimental':        td['Dia experimental'],
                                    'body_part':               td['body_part'],
                                    'Estímulo':                td['Estímulo'],
                                    'Coordenada_x':            x_coord,
                                    'Coordenada_y':            y_coord,
                                    'Distancia Intracortical': dist_ic,
                                    'MovementType':            'Gaussian-based',
                                    'Periodo':                 'Durante Estímulo',
                                    'A_gauss':                 g['A_gauss'],
                                    'mu_gauss':                g['mu_gauss'],
                                    'sigma_gauss':             g['sigma_gauss'],
                                    # columnas “históricas”:
                                    't_start':                 g['mu_gauss'] - 2*g['sigma_gauss'],
                                    't_peak':                  g['mu_gauss'],
                                    't_end':                   g['mu_gauss'] + 2*g['sigma_gauss'],
                                    'valor_pico':              g['A_gauss']
                                }
                                submovement_details_all.append(entry)
                                if td['Descartar']=='No':
                                    submovement_details_valid.append(entry)

                        elif sm['MovementType'] == 'MinimumJerk':
                            t_seg = sm['t_segment_model']
                            v_sm  = sm['v_sm']
                            if len(t_seg)>0 and len(v_sm)>0:
                                idx_peak = int(np.argmax(v_sm))
                                entry = {
                                    'Ensayo_Key':              td['trial_id'],
                                    'Dia experimental':        td['Dia experimental'],
                                    'body_part':               td['body_part'],
                                    'Estímulo':                td['Estímulo'],
                                    'Coordenada_x':            x_coord,
                                    'Coordenada_y':            y_coord,
                                    'Distancia Intracortical': dist_ic,
                                    'MovementType':            'MinimumJerk',
                                    'Periodo':                 'Durante Estímulo',
                                    't_start':                 t_seg[0],
                                    't_peak':                  t_seg[idx_peak],
                                    't_end':                   t_seg[-1],
                                    'valor_pico':              float(np.max(v_sm))
                                }
                                submovement_details_all.append(entry)
                                if td['Descartar']=='No':
                                    submovement_details_valid.append(entry)

                # 8) Graficar
                all_stimuli_data[stimulus_key] = {
                    'trials_data': trials_data,
                    'y_max_velocity': min(np.percentile(
                        sum([d['max_velocities'] for d in amplitude_movement_counts.values()], []), 90
                    ), 640),
                    'threshold': threshold,
                    'Estímulo': stimulus_key
                }
                for sk,d in all_stimuli_data.items():
                    plot_trials_side_by_side(
                        stimulus_key=sk,
                        data=d,
                        body_part=current_part,
                        dia_experimental=dia_exp,
                        output_dir=output_comparisons_dir,
                        coord_x=x_coord,
                        coord_y=y_coord,
                        dist_ic=dist_ic
                    )

    # ——— Guardado de CSV finales ———
    # Renombramos trial_id → Ensayo antes de guardar
    expanded_df = pd.DataFrame(expanded_trials)
    expanded_df.rename(columns={'trial_id': 'Ensayo'}, inplace=True)
    expanded_df.to_csv(
         os.path.join(output_comparisons_dir, 'expanded_trials_detailed.csv'),
         index=False
    )

    counts_df = pd.DataFrame(all_movement_data)
    # Aseguramos acento en Estímulo y luego normalizamos para el merge
    counts_df['Estímulo'] = counts_df.apply(
         lambda r: f"{r['Forma del Pulso'].strip().capitalize()}, {int(round(float(r['Duración (ms)']),0))} ms",
         axis=1
    )
    # Merge y_max_velocity
    if global_stimuli_data:
        # Reconstruimos una lista de registros que SÍ incluyan body_part
        stim_list = []
        for (dia_exp, bp), stim_dict in global_stimuli_data.items():
            for stim_key, d in stim_dict.items():
                stim_list.append({
                    'Dia experimental': dia_exp,
                    'body_part': bp,
                    'Estímulo':   d['Estímulo'],
                    'y_max_velocity': d['y_max_velocity']
                })
        stimuli_df = pd.DataFrame(stim_list)

        # Normalizamos para merge
        # Normalizamos para merge (mismas columnas en ambos DFs)
        stimuli_df['Estímulo']  = stimuli_df['Estímulo'].str.lower().str.strip()
        stimuli_df['body_part'] = stimuli_df['body_part'].str.lower().str.strip()
        counts_df['Estímulo']   = counts_df['Estímulo'].str.lower().str.strip()
        counts_df['body_part']  = counts_df['body_part'].str.lower().str.strip()

        counts_df = counts_df.merge(
            stimuli_df[['Dia experimental','body_part','Estímulo','y_max_velocity']],
            on=['Dia experimental','body_part','Estímulo'],
            how='left'
        )


    counts_df.to_csv(
        os.path.join(output_comparisons_dir, 'movement_counts_summary.csv'),
        index=False
    )
    pd.DataFrame(thresholds_dict.values()).to_csv(
        os.path.join(output_comparisons_dir, 'thresholds_summary.csv'), index=False
    )
    pd.DataFrame(movement_ranges_all).to_csv(
        os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv'), index=False
    )

    if submovement_details_all:
        df_all = pd.DataFrame(submovement_details_all)
        df_all['Submov_Num'] = df_all.groupby(['Ensayo_Key','MovementType']).cumcount()+1
        df_all.to_csv(
            os.path.join(output_comparisons_dir, 'submovement_detailed_all.csv'),
            index=False
        )
    if submovement_details_valid:
        df_val = pd.DataFrame(submovement_details_valid)
        df_val['Submov_Num'] = df_val.groupby(['Ensayo_Key','MovementType']).cumcount()+1
        df_val.to_csv(
            os.path.join(output_comparisons_dir, 'submovement_detailed_valid.csv'),
            index=False
        )

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


# --- NUEVO: Calcula partial eta-squared a partir de la tabla anova_lm (Type III)
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

    if 'Estímulo' not in aggregated_df.columns:
        if 'Estimulo' in aggregated_df.columns:
            aggregated_df = aggregated_df.rename(columns={'Estimulo':'Estímulo'})
        else:
            raise KeyError(
                f"No encontré ni 'Estímulo' ni 'Estimulo'.\n"
                f"Columnas recibidas: {aggregated_df.columns.tolist()}"
            )
    # Ahora ya podemos extraer Forma y Duración sin errores
    aggregated_df['Forma del Pulso'] = aggregated_df['Estímulo'].apply(
        lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan
    )
    aggregated_df['Duración (ms)'] = aggregated_df['Estímulo'].apply(
        lambda s: float(s.split(', ')[1].replace(' ms',''))
                  if isinstance(s, str) and ', ' in s else np.nan
    )

    # Primero, extraemos los factores de 'Estímulo'
    aggregated_df['Forma del Pulso'] = aggregated_df['Estímulo'].apply(lambda s: s.split(', ')[0] if isinstance(s, str) and ', ' in s else np.nan)
    aggregated_df['Duración (ms)'] = aggregated_df['Estímulo'].apply(lambda s: float(s.split(', ')[1].replace(' ms','')) if isinstance(s, str) and ', ' in s else np.nan)

    # Definir las métricas a analizar
    metrics = [
        'lat_inicio_ms',       # Latencia al inicio
        'lat_pico_mayor_ms',   # Latencia al pico mayor
        'valor_pico_max',      # Amplitud del pico máximo
        'dur_total_ms',        # Duración total
        'delta_t_pico',        # Diferencia entre primer y pico mayor
        'num_movs'             # Número de movimientos
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
            if data_metric.empty or len(data_metric) < 3:
                continue  # O registrar que este grupo no es válido para el ANOVA       

            if len(data_metric) < 3:
                continue
            formula = (
                f"Q('{metric}') ~ "
                f"C(Q('Forma del Pulso'), Sum) * C(Q('Duración (ms)'), Sum)"
            )
            try:
                model = ols(formula, data=data_metric).fit()
                anova_res = anova_lm(model, typ=3)
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
    fs = 100.0  # frames/s
    results = []

    # Agrupamos por ensayo y tipo de movimiento
    grouping_cols = [
        'Ensayo_Key','Estímulo','MovementType',
        'Dia experimental','body_part','Coordenada_x','Coordenada_y',
        'Distancia Intracortical'
    ]

    for (ensayo_key, estimulo, movement_type,
        dia_exp, body_part, coord_x, coord_y, dist_ic), group in submovements_df.groupby(grouping_cols):

        # now movement_type really is your MovementType
        mt = movement_type

        base = dict(zip(grouping_cols,
                        [ensayo_key, estimulo, movement_type,
                        dia_exp, body_part, coord_x, coord_y, dist_ic]))
        if mt == 'Threshold-based' and (group['Periodo']==PERIODO_DURANTE).any():
            g = group[group['Periodo']==PERIODO_DURANTE]
            idx_min_start = g['Latencia al Inicio (s)'].idxmin()
            idx_max_val   = g['Valor Pico (velocidad)'].idxmax()
            results.append({
                **base,
                'lat_inicio_ms': g.loc[idx_min_start, 'Latencia al Inicio (s)']*1000,
                'lat_pico_mayor_ms':   g.loc[idx_max_val,   'Latencia al Pico (s)']*1000,
                'valor_pico_max':      g['Valor Pico (velocidad)'].max(),
                'dur_total_ms':        ((g['Fin Movimiento (Frame)'].max()
                                         - g['Inicio Movimiento (Frame)'].min())/fs)*1000,
                'delta_t_pico':        ((g.loc[idx_max_val,'Latencia al Pico (s)']
                                         - g['Latencia al Pico (s)'].min())*1000),
                'num_movs':            g.shape[0]
            })
        elif mt == 'Gaussian-based':
            # mu_gauss and sigma_gauss
            starts = group['mu_gauss'] - 2*group['sigma_gauss']
            idx_max    = group['A_gauss'].idxmax()
            results.append({
                **base,
                'lat_inicio_ms':       (starts.min())*1000,
                'lat_pico_mayor_ms':   (group.loc[idx_max,'mu_gauss'])*1000,
                'valor_pico_max':      group['A_gauss'].max(),
                'dur_total_ms':        ((starts.max() + 2*group['sigma_gauss'].max()
                                         - starts.min())*1000),
                'delta_t_pico':        ((group.loc[idx_max,'mu_gauss']
                                         - group['mu_gauss'].min())*1000),
                'num_movs':            group.shape[0]
            })
        elif mt == 'MinimumJerk':
            idx_max = group['valor_pico'].idxmax()
            results.append({
                **base,
                'lat_inicio_ms':       group['t_start'].min()*1000,
                'lat_pico_mayor_ms':   group.loc[idx_max,'t_peak']*1000,
                'valor_pico_max':      group['valor_pico'].max(),
                'dur_total_ms':        ((group['t_end'].max()
                                         - group['t_start'].min())*1000),
                'delta_t_pico':        ((group.loc[idx_max,'t_peak']
                                         - group['t_peak'].min())*1000),
                'num_movs':            group.shape[0]
            })
        else:
            # si hay otro MovementType, lo ignoramos
            continue

    df = pd.DataFrame(results)
    # quitamos duplicados eventuales
    return df.loc[:,~df.columns.duplicated()]

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
                mod = ols(
                    f"Q('{metric}') ~ "
                    f"C(Q('Forma del Pulso'), Sum) * C(Q('Duración (ms)'), Sum)",
                    data=df_model
                ).fit()

                anova_res = anova_lm(mod, typ=3)
                p_shape = anova_res.loc["C(Q('Forma del Pulso'))", "PR(>F)"] if "C(Q('Forma del Pulso'))" in anova_res.index else np.nan
                p_dur = anova_res.loc["C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Duración (ms)'))" in anova_res.index else np.nan
                p_int = anova_res.loc["C(Q('Forma del Pulso')):C(Q('Duración (ms)'))", "PR(>F)"] if "C(Q('Forma del Pulso')):C(Q('Duración (ms)'))" in anova_res.index else np.nan
                pvals[mt] = (p_shape, p_dur, p_int)
            except Exception as e:
                logging.warning(f"Error en ANOVA para modelo {mt} y métrica {metric}: {e}")
                pvals[mt] = (np.nan, np.nan, np.nan)
    return pvals







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
    
    
    # Para pruebas de hipótesis usaremos el CSV detallado de submovimientos:
    submov_path = os.path.join(output_comparisons_dir, 'submovement_detailed_valid.csv')
    if os.path.exists(submov_path):
        # submovements_df = pd.read_csv(submov_path)
        submovements_df = pd.read_csv(submov_path, encoding='utf-8-sig')
        # si la lee como "Estimulo" sin tilde:
        if 'Estimulo' in submovements_df.columns and 'Estímulo' not in submovements_df.columns:
            submovements_df.rename(columns={'Estimulo': 'Estímulo'}, inplace=True)

        # ETAPA 1: Agregar métricas por ensayo a partir de los submovimientos detallados
        aggregated_df = aggregate_trial_metrics_extended(submovements_df)
        aggregated_df.to_csv(os.path.join(output_comparisons_dir, 'aggregated_metrics.csv'), index=False)
        print("Tabla agregada de métricas guardada en 'aggregated_metrics.csv'")
        
        # ETAPA 2: Pruebas de hipótesis (ANOVA, Friedman, etc.)
        do_significance_tests_aggregated(aggregated_df, output_dir=output_comparisons_dir)
        
    else:
        print("No se encontró el archivo submovement_detailed_summary.csv para generar los análisis.")

    from collections import Counter
    df = pd.read_csv(os.path.join(output_comparisons_dir, 'submovement_detailed_valid.csv'))
    print("Contador de tipos de movimiento:", Counter(df['MovementType']))
    

    print("Proceso completo finalizado.")