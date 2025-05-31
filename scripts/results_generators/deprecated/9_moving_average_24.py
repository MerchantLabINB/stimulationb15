# Importaciones y configuración inicial
import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
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
from matplotlib.patches import Patch
import textwrap

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

# NUEVA RUTA DE SALIDA PARA NO CONFUNDIR RESULTADOS
output_comparisons_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\triple_rombo_24_05_plots'

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

# FILTRO ESPECÍFICO
stimuli_info = stimuli_info[(stimuli_info['Dia experimental'] == '24/05') &
                            (stimuli_info['Forma del Pulso'] == 'triple rombo')]


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


import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import least_squares

def fit_velocity_profile(t, observed_velocity, max_submovements=10, prominence=0.01, distance=10, save_diagnostics=False, save_path=None):
    """
    Ajusta el perfil de velocidad utilizando una suma de submovimientos de mínimo jerk.

    Parámetros adicionales:
    - save_diagnostics (bool): Si es True, guarda un plot de diagnóstico de la detección de picos.
    - save_path (str): Ruta donde se guardará el plot de diagnóstico si save_diagnostics es True.
    """
    # Detectar picos en la velocidad para estimar el número de submovimientos
    peaks, properties = find_peaks(observed_velocity, prominence=prominence, distance=distance)
    n_submovements = min(len(peaks), max_submovements)

    if n_submovements == 0:
        logging.warning("No se detectaron picos en la velocidad. Saltando el ajuste.")
        return None

    # Opcional: Guardar plot de picos detectados
    if save_diagnostics and save_path:
        plt.figure(figsize=(10, 4))
        plt.plot(t, observed_velocity, label='Velocidad Observada')
        plt.plot(t[peaks], observed_velocity[peaks], "x", label='Picos Detectados')
        plt.title('Detección de Picos para Ajuste')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Velocidad (pixeles/s)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Diagnóstico de picos guardado en {save_path}")

    # Inicializar parámetros para submovimientos
    params_init = []
    lower_bounds = []
    upper_bounds = []

    for peak in peaks[:n_submovements]:
        A_init = observed_velocity[peak]  # Amplitud inicial basada en el pico
        t0_init = t[peak]                # Tiempo inicial del submovimiento
        T_init = 0.5                      # Duración inicial del submovimiento (ajustable)

        params_init.extend([A_init, t0_init, T_init])
        lower_bounds.extend([0, t[0], 0.1])   # Amplitud >=0, t0 >= inicio, T > 0
        upper_bounds.extend([np.inf, t[-1], t[-1] - t0_init])  # Amplitud sin límite, t0 <= final, T <= duración restante

    # Definir la función objetivo
    def objective(params):
        modeled_velocity = sum_of_minimum_jerk(t, *params)
        return modeled_velocity - observed_velocity

    # Realizar el ajuste de mínimos cuadrados
    try:
        result = least_squares(
            objective,
            x0=params_init,
            bounds=(lower_bounds, upper_bounds)
        )
    except ValueError as e:
        logging.error(f"Error en el ajuste de mínimos cuadrados: {e}")
        return None

    # Verificar el éxito del ajuste
    if not result.success:
        logging.warning(f"Ajuste no exitoso: {result.message}")
        return None

    return result

def sum_of_minimum_jerk(t, *params):
    """
    Calcula la suma de submovimientos de mínimo jerk.
    Cada submovimiento tiene 3 parámetros: A, t0, T.
    """
    velocity = np.zeros_like(t)
    n_submovements = len(params) // 3
    for i in range(n_submovements):
        A = params[3*i]
        t0 = params[3*i + 1]
        T = params[3*i + 2]
        velocity += minimum_jerk_velocity(t, A, t0, T)
    return velocity

def minimum_jerk_velocity(t, A, t0, T):
    """
    Calcula la velocidad de un submovimiento de mínimo jerk.
    """
    v = np.zeros_like(t)
    tau = (t - t0) / T
    valid_idx = (tau >= 0) & (tau <= 1)
    v[valid_idx] = A * 30 * (tau[valid_idx]**2) * (1 - tau[valid_idx])**2
    return v

def sanitize_filename(filename):
    """
    Reemplaza los caracteres inválidos por guiones bajos.
    """
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
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

# Función de suavizado usando Savitzky-Golay
# Función de suavizado usando un simple moving average
def suavizar_datos(data, window_size=21):
    """
    Aplica un simple moving average a los datos.

    Parámetros:
    - data: Array de datos a suavizar.
    - window_size: Tamaño de la ventana para el promedio móvil.

    Retorna:
    - Array de datos suavizados.
    """
    if len(data) < window_size:
        logging.warning(f"Datos demasiado cortos para aplicar moving average. Longitud={len(data)}, window_size={window_size}")
        return data
    try:
        # Utilizar 'same' para mantener el tamaño original
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    except Exception as e:
        logging.error(f'Error al aplicar moving average: {e}')
        return data




# Función para calcular velocidades y posiciones para cada articulación con suavizado
# Función para calcular velocidades y posiciones para cada articulación con suavizado
def calcular_velocidades(csv_path):
    logging.debug(f"Calculando velocidades para CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        logging.debug(f"Archivo CSV cargado: {csv_path}")
        
        # Aplanar las columnas de MultiIndex a single-level
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        logging.debug("Columnas aplanadas a single-level.")
        
        # Eliminar la columna 'scorer_bodyparts_coords' si existe
        if 'scorer_bodyparts_coords' in df.columns:
            df = df.drop(columns=['scorer_bodyparts_coords'])
            logging.debug("Columna 'scorer_bodyparts_coords' eliminada.")
        
        # Ajustar los nombres de las articulaciones para que coincidan con las columnas
        body_parts_adjusted = [part.replace('ñ', 'n').replace(' ', '_') for part in body_parts]
        
        velocidades_cruda = {}
        velocidades_suavizada = {}
        posiciones = {}
        
        for part_original, part in zip(body_parts, body_parts_adjusted):
            # Inicializar variables de columna
            x_col = None
            y_col = None
            likelihood_col = None

            # Encontrar columnas que correspondan a la articulación
            for col in df.columns:
                if col.endswith('_x') and part in col:
                    x_col = col
                elif col.endswith('_y') and part in col:
                    y_col = col
                elif col.endswith('_likelihood') and part in col:
                    likelihood_col = col

            # Verificar que todas las columnas necesarias fueron encontradas
            if not x_col or not y_col or not likelihood_col:
                logging.warning(f"Columns for {part_original} están incompletas en {csv_path}.")
                print(f"Columns for {part_original} están incompletas.")
                continue

            logging.debug(f"Usando columnas para {part_original}: x_col={x_col}, y_col={y_col}, likelihood_col={likelihood_col}")
            print(f"Usando columnas para {part_original}: x_col={x_col}, y_col={y_col}, likelihood_col={likelihood_col}")

            # Filtrar filas basadas en la probabilidad
            df_filtered = df[df[likelihood_col] > 0.1]
            valid_frames = len(df_filtered)
            logging.info(f'{part_original} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')
            print(f'{part_original} en {csv_path}: {valid_frames}/{len(df)} frames válidos después de filtrar por likelihood.')

            if df_filtered.empty:
                logging.warning(f'No hay datos suficientes para {part_original} en {csv_path} después de filtrar por likelihood.')
                print(f'No hay datos suficientes para {part_original} en {csv_path} después de filtrar por likelihood.')
                velocidades_cruda[part_original] = np.array([])
                velocidades_suavizada[part_original] = np.array([])
                posiciones[part_original] = {'x': np.array([]), 'y': np.array([])}
                continue

            # Extraer posiciones x y y
            x = df_filtered[x_col].values
            y = df_filtered[y_col].values

            # Calcular velocidades crudas
            delta_x = np.diff(x)
            delta_y = np.diff(y)
            distancias = np.hypot(delta_x, delta_y)
            delta_t = 1 / 100  # Asumiendo 100 fps
            velocidad_part_cruda = distancias / delta_t

            # Calcular velocidades suavizadas
            velocidad_part_suavizada = suavizar_datos(velocidad_part_cruda, window_size=21)

            logging.debug(f"Velocidades crudas y suavizadas para {part_original}.")
            print(f"Velocidades crudas y suavizadas para {part_original}.")

            # Almacenar velocidades y posiciones
            velocidades_cruda[part_original] = velocidad_part_cruda
            velocidades_suavizada[part_original] = velocidad_part_suavizada
            posiciones[part_original] = {'x': x, 'y': y}

        logging.info(f"Finalizado cálculo de velocidades para {csv_path}.")
        print(f"Finalizado cálculo de velocidades para {csv_path}.")
        return velocidades_cruda, velocidades_suavizada, posiciones

    except FileNotFoundError:
        logging.error(f"Archivo no encontrado: {csv_path}")
        print(f"Archivo no encontrado: {csv_path}")
        return {}, {}, {}
    except pd.errors.EmptyDataError:
        logging.error(f"Archivo vacío o corrupto: {csv_path}")
        print(f"Archivo vacío o corrupto: {csv_path}")
        return {}, {}, {}
    except Exception as e:
        logging.error(f'Error inesperado al calcular velocidades para {csv_path}: {e}')
        print(f'Error inesperado al calcular velocidades para {csv_path}: {e}')
        return {}, {}, {}



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

def plot_trials_side_by_side(stimulus_key, data, body_part, dia_experimental, output_dir, max_trials_per_row=15):
    """
    Genera gráficos organizados por filas (5 filas: desplazamiento, velocidad, duración de movimiento,
    submovimientos y estímulo) y columnas (hasta max_trials_per_row ensayos por fila).

    Parámetros:
    - stimulus_key (str): Clave única para el estímulo.
    - data (dict): Diccionario con datos asociados al estímulo (velocidades, posiciones, submovimientos, etc.).
    - body_part (str): Articulación específica a graficar.
    - dia_experimental (str/int): Día experimental.
    - output_dir (str): Directorio donde se guardarán las figuras.
    - max_trials_per_row (int): Número máximo de ensayos por fila en la figura.
    """

    # Crear subdirectorio específico para el estímulo, día y articulación
    output_dir = os.path.join(
        output_dir,
        f'Dia_{sanitize_filename(str(dia_experimental))}',
        f'{sanitize_filename(stimulus_key)}',
        f'{sanitize_filename(body_part)}'
    )
    os.makedirs(output_dir, exist_ok=True)

    trials_data = data.get('trials_data', [])
    if not trials_data:
        logging.warning(f"No hay datos de ensayos para {stimulus_key} en {body_part} día {dia_experimental}")
        print(f"No hay datos de ensayos para {stimulus_key} en {body_part} día {dia_experimental}")
        return

    # Filtrar ensayos con datos de velocidad válidos
    non_empty_trials = [trial for trial in trials_data if len(trial['velocity_cruda']) > 0]
    if not non_empty_trials:
        logging.warning(f"No hay ensayos con datos de velocidad válidos para {stimulus_key} en {body_part} día {dia_experimental}")
        print(f"No hay ensayos con datos de velocidad válidos para {stimulus_key} en {body_part} día {dia_experimental}")
        return

    num_trials = len(non_empty_trials)
    num_rows = (num_trials - 1) // max_trials_per_row + 1
    num_cols = min(num_trials, max_trials_per_row)

    # Calcular límites máximos para el escalado de ejes
    max_time = 0
    max_disp = 0
    max_vel = 0
    min_amp = float('inf')
    max_amp = float('-inf')

    for trial in non_empty_trials:
        pos = trial['positions']
        if 'x' in pos and 'y' in pos and len(pos['x']) > 0 and len(pos['y']) > 0:
            displacement = np.sqrt((pos['x'] - pos['x'][0])**2 + (pos['y'] - pos['y'][0])**2)
            t_disp = len(displacement) / 100
            if t_disp > max_time:
                max_time = t_disp
            current_disp = displacement.max() if len(displacement) > 0 else 0
            if current_disp > max_disp:
                max_disp = current_disp

        vel_cruda = trial['velocity_cruda']
        t_vel = len(vel_cruda) / 100
        if t_vel > max_time:
            max_time = t_vel
        current_vel = vel_cruda.max() if len(vel_cruda) > 0 else 0
        if current_vel > max_vel:
            max_vel = current_vel

    if 'amplitude_list' in data and len(data['amplitude_list']) > 0:
        min_amp = min(min_amp, min(data['amplitude_list']))
        max_amp = max(max_amp, max(data['amplitude_list']))
    else:
        # Valores por defecto
        min_amp = -160
        max_amp = 160

    if max_time == 0:
        max_time = 10
    if max_disp == 0:
        max_disp = 100
    if max_vel == 0:
        max_vel = 100
    if min_amp == float('inf'):
        min_amp = -160
    if max_amp == float('-inf'):
        max_amp = 160

    print(f"Variables calculadas - max_time: {max_time}, max_disp: {max_disp}, max_vel: {max_vel}, min_amp: {min_amp}, max_amp: {max_amp}")

    # Definir tamaño de la figura
    fig_height = 5 * 4  # 5 subplots por ensayo (filas)
    fig_width = 4 * max_trials_per_row
    fig, axes = plt.subplots(5, max_trials_per_row, figsize=(fig_width, fig_height), sharey=False)
    axes = axes.flatten()

    # Ventanas de promedios móviles solicitadas
    window_sizes = [5, 10, 15]
    window_colors = {
        5: 'red',
        10: 'orange',
        15: 'green'
    }

    for idx, trial_data in enumerate(non_empty_trials):
        if idx >= max_trials_per_row * 5:
            logging.warning(f"Se excedió el máximo de subplots disponibles ({max_trials_per_row * 5}).")
            print(f"Se excedió el máximo de subplots disponibles ({max_trials_per_row * 5}).")
            break

        col = idx % max_trials_per_row
        trial_index = trial_data['trial_index']
        vel_cruda = trial_data.get('velocity_cruda', [])
        pos = trial_data['positions']
        submovements = trial_data['submovements']
        movement_ranges = trial_data['movement_ranges']

        frames_vel = np.arange(len(vel_cruda))
        t_vel = frames_vel / 100

        # Asignación de axes
        ax_disp = axes[col + 0 * max_trials_per_row]
        ax_vel = axes[col + 1 * max_trials_per_row]
        ax_mov = axes[col + 2 * max_trials_per_row]
        ax_submov = axes[col + 3 * max_trials_per_row]
        ax_stim = axes[col + 4 * max_trials_per_row]

        # 1. Desplazamiento
        if 'x' in pos and 'y' in pos and len(pos['x']) > 0 and len(pos['y']) > 0:
            displacement = np.sqrt((pos['x'] - pos['x'][0])**2 + (pos['y'] - pos['y'][0])**2)
            t_disp = np.arange(len(displacement)) / 100
            ax_disp.plot(t_disp, displacement, color=body_parts_specific_colors.get(body_part, 'blue'))
            ax_disp.set_title(f'Ensayo {trial_index + 1} - Desplazamiento')
            ax_disp.set_xlabel('Tiempo (s)')
            ax_disp.set_ylabel('Desplazamiento (px)')
            ax_disp.set_xlim(0, max_time)
            ax_disp.set_ylim(0, max_disp + 10)
            ax_disp.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)
        else:
            ax_disp.text(0.5, 0.5, 'Sin datos de posición', ha='center', va='center', fontsize=8)
            ax_disp.set_title(f'Ensayo {trial_index + 1} - Desplazamiento')
            ax_disp.set_xlabel('Tiempo (s)')
            ax_disp.set_ylabel('Desplazamiento (px)')
            ax_disp.set_xlim(0, max_time)
            ax_disp.set_ylim(0, max_disp + 10)
            ax_disp.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)

        # 2. Velocidad cruda y promedios móviles (5,10,15)
        ax_vel.plot(t_vel, vel_cruda, color='blue', alpha=0.6, label='Velocidad Cruda')

        # Promedios móviles
        for window_size in window_sizes:
            color_ma = window_colors.get(window_size, 'black')
            label_ma = f'Prom. Móvil {window_size}'
            if len(vel_cruda) >= window_size:
                ma = suavizar_datos(vel_cruda, window_size=window_size)
                ax_vel.plot(t_vel, ma, color=color_ma, alpha=0.8, label=label_ma)
            else:
                logging.warning(f"Ensayo {trial_index + 1}: Insuficientes datos para ventana {window_size}.")
                print(f"Ensayo {trial_index + 1}: Insuficientes datos para ventana {window_size}.")

        ax_vel.axhline(data['threshold'], color='k', linestyle='--', label=f'Umbral ({data["threshold"]:.2f})')
        ax_vel.axhline(data['mean_vel_pre'], color='lightcoral', linestyle='-', linewidth=1, label=f'Media Pre-estímulo ({data["mean_vel_pre"]:.2f})')
        ax_vel.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)
        ax_vel.set_xlabel('Tiempo (s)')
        ax_vel.set_ylabel('Vel. (pix/s)')
        ax_vel.set_xlim(0, max_time)
        ax_vel.set_ylim(0, max_vel + 5)

        handles, labels = ax_vel.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_vel.legend(by_label.values(), by_label.keys(), fontsize=8)

        # 3. Duración del movimiento (representación achatada)
        ax_mov.set_title(f'Ensayo {trial_index + 1} - Duración Mov.')
        ax_mov.set_xlabel('Tiempo (s)')
        ax_mov.set_ylabel('Movimiento')
        ax_mov.set_xlim(0, max_time)
        ax_mov.set_ylim(0.95, 1.05)
        for movement in movement_ranges:
            movement_start = movement['Inicio Movimiento (Frame)'] / 100
            movement_end = movement['Fin Movimiento (Frame)'] / 100
            periodo = movement['Periodo']
            if periodo == 'Pre-Estímulo':
                color_mov = 'orange'
            elif periodo == 'Durante Estímulo':
                color_mov = 'red'
            elif periodo == 'Post-Estímulo':
                color_mov = 'gray'
            else:
                color_mov = 'blue'
            ax_mov.hlines(y=1, xmin=movement_start, xmax=movement_end, color=color_mov, linewidth=4)
        ax_mov.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)
        ax_mov.get_yaxis().set_visible(False)

        # 4. Submovimientos (Mínimo Jerk + Gaussianas)
        if submovements:
            for i, submovement in enumerate(submovements, 1):
                try:
                    movement_start = submovement['movement']['Inicio Movimiento (Frame)'] / 100
                    movement_end = submovement['movement']['Fin Movimiento (Frame)'] / 100
                    t_segment = np.arange(movement_start, movement_end + 0.01, 0.01)
                    v_sm = submovement['v_sm']

                    if len(t_segment) != len(v_sm):
                        min_length = min(len(t_segment), len(v_sm))
                        t_segment = t_segment[:min_length]
                        v_sm = v_sm[:min_length]

                    A = submovement['A']
                    t0 = submovement['t0']
                    T = submovement['T']
                    sigma = T / 4
                    gaussian = A * np.exp(-0.5 * ((t_segment - t0) / sigma) ** 2)

                    # Sin información extra del window_size del submovimiento, usar un color genérico
                    # Podrías asociar los submovimientos con una ventana si fuese necesario
                    color_sm = 'purple'
                    ax_submov.plot(t_segment, gaussian, linestyle='--', linewidth=2, color=color_sm, label=f'Gaussiana {i}')
                    ax_submov.plot(t_segment, v_sm, linestyle='-', linewidth=2, color=color_sm, label=f'Submov {i}')
                except KeyError as e:
                    logging.error(f"Clave faltante en submovement: {e}")
                    print(f"Clave faltante en submovement: {e}")
                    continue
        else:
            ax_submov.text(0.5, 0.5, 'No hay submovimientos', ha='center', va='center', fontsize=8)

        ax_submov.set_title(f'Ensayo {trial_index + 1} - Submovimientos')
        ax_submov.set_xlabel('Tiempo (s)')
        ax_submov.set_ylabel('Vel. Ajustada (pix/s)')
        ax_submov.set_xlim(0, max_time)
        ax_submov.set_ylim(0, max_vel + 5)

        ax_submov.axhline(data['threshold'], color='k', linestyle='--', label=f'Umbral ({data["threshold"]:.2f})')
        ax_submov.axhline(data['mean_vel_pre'], color='lightcoral', linestyle='-', linewidth=1, label=f'Media Pre-estímulo ({data["mean_vel_pre"]:.2f})')
        ax_submov.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)

        if submovements:
            handles, labels = ax_submov.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_submov.legend(by_label.values(), by_label.keys(), fontsize=8)
        else:
            ax_submov.legend(fontsize=8)

        # 5. Estímulo
        x_vals = [data['start_frame'] / 100]
        y_vals = [0]
        current_time_stimulus = data['start_frame'] / 100
        for amp, dur in zip(data['amplitude_list'], data['duration_list']):
            next_time = current_time_stimulus + dur / 100
            x_vals.extend([current_time_stimulus, next_time])
            y_vals.extend([amp / 1000, amp / 1000])
            current_time_stimulus = next_time

        ax_stim.step(x_vals, y_vals, color='purple', where='post', linewidth=1)
        ax_stim.set_title(f'Ensayo {trial_index + 1} - Estímulo')
        ax_stim.set_xlabel('Tiempo (s)')
        ax_stim.set_ylabel('Amplitud (μA)')
        ax_stim.set_xlim(0, max_time)
        ax_stim.set_ylim(-160, 160)
        ax_stim.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax_stim.axvspan(data['start_frame'] / 100, data['current_frame'] / 100, color='green', alpha=0.1)

        if col == 0:
            estimulo_params_text = f"Forma: {data['form']}\nDuración: {data['duration_ms']} ms\nFrecuencia: {data['frequency']} Hz"
            ax_stim.text(
                0.95, 0.95, estimulo_params_text,
                transform=ax_stim.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    main_title = f'{stimulus_key} - {body_part} - Día {dia_experimental}'
    fig.suptitle(main_title, fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.95)

    output_path = os.path.join(output_dir, 'trials_side_by_side.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico guardado en {output_path}")
    print(f"Gráfico guardado en {output_path}")





# ... (código de importaciones y configuración inicial permanece igual)

def collect_velocity_threshold_data():
    logging.info("Iniciando la recopilación de datos de umbral de velocidad.")
    print("Iniciando la recopilación de datos de umbral de velocidad.")
    
    total_trials = 0
    all_movement_data = []
    thresholds_data = []
    processed_combinations = set()
    movement_ranges_all = []

    # Definir output_dir_trials al inicio
    output_dir_trials = os.path.join(output_comparisons_dir, 'trials_side_by_side')
    os.makedirs(output_dir_trials, exist_ok=True)
    logging.info(f"Directorio para ensayos individuales: {output_dir_trials}")
    print(f"Directorio para ensayos individuales: {output_dir_trials}")

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
            logging.info(f'Procesando: Día {dia_experimental}, Articulación {part}')
            print(f'Procesando: Día {dia_experimental}, Articulación {part}')
            processed_combinations.add((dia_experimental, 'All_Stimuli', part))  # 'All_Stimuli' indica que no hay grupos específicos

            # Inicializar diccionarios para acumular datos de todos los estímulos
            all_stimuli_data = {}  # Clave: stimulus_key, Valor: datos para graficar

            # Velocidades pre-estímulo para la articulación específica en todos los ensayos del día
            pre_stim_velocities = []

            # Inicializar el contador de ensayos por grupo
            trial_counter_group = 0  # <--- Inicialización añadida

            # Inicializar lista para almacenar datos de ensayos
            trials_data = []  # <--- Inicialización añadida

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
                                velocidades_cruda, velocidades_suavizada, posiciones = calcular_velocidades(csv_path)

                                vel_cruda = velocidades_cruda.get(part, [])
                                if len(vel_cruda) > 0:
                                    vel_pre_stim = vel_cruda[:100]  # Primeros 100 frames para pre-estímulo
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
                                        velocidades_cruda, velocidades_suavizada, posiciones = calcular_velocidades(csv_path)

                                        vel_cruda = velocidades_cruda.get(part, [])
                                        vel_suavizada = velocidades_suavizada.get(part, [])
                                        pos = posiciones.get(part, {})

                                        # **Inicializar 'submovements' al inicio de cada ensayo**
                                        submovements = []

                                        if len(vel_cruda) > 0:
                                            vel_pre_stim = vel_cruda[:start_frame]
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
                                            frames_vel = np.arange(len(vel_cruda))
                                            above_threshold = (vel_cruda > threshold)
                                            indices_above = frames_vel[above_threshold]

                                            if len(indices_above) > 0:
                                                segments = np.split(indices_above, np.where(np.diff(indices_above) != 1)[0] + 1)

                                                for segment in segments:
                                                    movement_start = segment[0]
                                                    movement_end = segment[-1]

                                                    if start_frame <= movement_start <= current_frame:
                                                        # Crear movement_data correctamente
                                                        movement_data = {
                                                            'Ensayo': trial_counter_group + 1,  # Asumiendo que trial_counter_group comienza en 0
                                                            'Inicio Movimiento (Frame)': movement_start,
                                                            'Fin Movimiento (Frame)': movement_end,
                                                            'Periodo': 'Durante Estímulo',
                                                            'Forma del Pulso': forma_pulso.capitalize(),
                                                            'Duración (ms)': duracion_ms,
                                                            'Dia experimental': dia_experimental  # Añadir el día experimental
                                                        }

                                                        # Ajustar submovimientos para este segmento
                                                        t_segment = frames_vel[movement_start:movement_end+1] / 100
                                                        vel_segment = vel_cruda[movement_start:movement_end+1]

                                                        # Guardar diagnósticos para los primeros 5 ensayos
                                                        save_diagnostics = trial_counter_group < 5
                                                        save_path = os.path.join(output_dir_trials, f'diagnostic_trial_{trial_counter_group + 1}.png') if save_diagnostics else None

                                                        result = fit_velocity_profile(
                                                            t_segment,
                                                            vel_segment,
                                                            save_diagnostics=save_diagnostics,
                                                            save_path=save_path
                                                        )
                                                        if result is None:
                                                            logging.warning(f"Omitiendo movimiento en el ensayo {trial_counter_group + 1} debido a datos insuficientes para el ajuste.")
                                                            continue

                                                        fitted_params = result.x
                                                        n_submovements = len(fitted_params) // 3
                                                        submovements = []

                                                        for i in range(n_submovements):
                                                            A = fitted_params[3*i]
                                                            t0 = fitted_params[3*i + 1]
                                                            T = fitted_params[3*i + 2]
                                                            v_sm = minimum_jerk_velocity(t_segment, A, t0, T)
                                                            submovement = {
                                                                'A': A,
                                                                't0': t0,
                                                                'T': T,
                                                                'v_sm': v_sm,
                                                                'movement': movement_data  # Asociar con el movimiento
                                                            }
                                                            submovements.append(submovement)

                                                        movement_data['submovements'] = submovements
                                                        movement_ranges_all.append(movement_data)

                                                        movement_trials += 1  # Incrementar el conteo
                                                        break  # Solo necesitamos contar una vez por trial

                                            # Colectar máximos de velocidad
                                            max_vel = np.max(vel_cruda)
                                            max_velocities.append(max_vel)

                                            # Almacenar datos por ensayo
                                            trial_data = {
                                                'velocity_cruda': vel_cruda,
                                                'velocity_suavizada': vel_suavizada,
                                                'positions': pos,
                                                'trial_index': trial_counter_group,  # Ahora está definido
                                                'movement_ranges': [md for md in movement_ranges_all if md['Ensayo'] == trial_counter_group + 1],
                                                'submovements': submovements  # Siempre asignado
                                            }
                                            trials_data.append(trial_data)

                                            # **Incrementar el contador aquí**
                                            trial_counter_group += 1  # <--- Incremento añadido
                                        else:
                                            # Si no hay movimientos que excedan el umbral, almacenar el ensayo sin submovimientos
                                            submovements = []  # <--- Inicialización añadida
                                            trial_data = {
                                                'velocity_cruda': vel_cruda,
                                                'velocity_suavizada': vel_suavizada,
                                                'positions': pos,
                                                'trial_index': trial_counter_group,  # Ahora está definido
                                                'movement_ranges': [],
                                                'submovements': submovements  # Asignar lista vacía
                                            }
                                            trials_data.append(trial_data)

                                            # **Incrementar el contador aquí también**
                                            trial_counter_group += 1  # <--- Incremento añadido

                # Después de procesar todos los estímulos y ensayos
                # Almacenar los datos de todos los estímulos
                if trials_data:
                    # Crear una clave única para este estímulo con el número de orden
                    if duracion_ms is not None:
                        stimulus_key = f"{order}. {forma_pulso.capitalize()}_{duracion_ms}ms"
                    else:
                        stimulus_key = f"{order}. {forma_pulso.capitalize()}"

                    # Obtener 'form' del primer ensayo
                    form = 'Unknown'
                    if trials_data and trials_data[0]['movement_ranges']:
                        form = trials_data[0]['movement_ranges'][0].get('Forma del Pulso', forma_pulso.capitalize())

                    # Almacenar los datos para graficar
                    all_stimuli_data[stimulus_key] = {
                        'velocities': [trial['velocity_cruda'] for trial in trials_data],
                        'positions': [trial['positions'] for trial in trials_data],
                        'threshold': threshold,
                        'amplitude_list': amplitude_list,
                        'duration_list': duration_list,
                        'start_frame': start_frame,
                        'current_frame': current_frame,
                        'mean_vel_pre': mean_vel_pre,
                        'std_vel_pre': std_vel_pre,
                        'movement_ranges': movement_ranges_all,
                        'form': form.capitalize(),  # <--- Corrección aquí
                        'duration_ms': duracion_ms,   # <--- Añadido
                        'frequency': row['Frecuencia (Hz)'] if 'Frecuencia (Hz)' in row else 'Unknown',  # <--- Corregido
                        'trials_data': trials_data  # Almacenar datos por ensayo
                    }

            if not all_stimuli_data:
                logging.debug(f"No hay estímulos para graficar para {part} en el día {dia_experimental}.")
                print(f"No hay estímulos para graficar para {part} en el día {dia_experimental}.")
                continue  # No hay datos para graficar para este grupo

            # Almacenar los datos de todos los estímulos
            all_movement_data.extend(movement_ranges_all)

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

            # --- Llamar a la función para graficar por ensayo ---
            for stimulus_key, data in all_stimuli_data.items():
                plot_trials_side_by_side(
                    stimulus_key=stimulus_key,
                    data=data,
                    body_part=part,
                    dia_experimental=dia_experimental,
                    output_dir=output_dir_trials,
                    max_trials_per_row=15
                )

    # Crear el DataFrame counts_df a partir de all_movement_data
    counts_df = pd.DataFrame(all_movement_data)

    # Retornar counts_df y movement_ranges_all
    return counts_df, movement_ranges_all


def plot_summary_movement_data(movement_ranges_df):
    logging.info("Generando gráficos comparativos simplificados de movimientos durante el estímulo por articulación y día, incluyendo Latencia al Pico y Valor Pico.")
    print("Generando gráficos comparativos simplificados de movimientos durante el estímulo por articulación y día, incluyendo Latencia al Pico y Valor Pico.")

    # Filtrar movimientos durante el estímulo y hacer una copia
    df_durante_estimulo = movement_ranges_df[movement_ranges_df['Periodo'] == 'Durante Estímulo'].copy()

    if df_durante_estimulo.empty:
        logging.info("No hay movimientos durante el estímulo para graficar.")
        print("No hay movimientos durante el estímulo para graficar.")
        return

    # Convertir tiempos a milisegundos si es necesario
    # Asumiendo que ya están en milisegundos, ajustar según corresponda
    # Si están en frames o segundos, conviértelos apropiadamente

    # Crear métricas adicionales si no existen
    # Por ejemplo, si deseas calcular la latencia al pico y el valor pico
    # Asegúrate de que estos cálculos estén reflejados en tu DataFrame

    # Ejemplo de cálculo de Latencia al Pico y Valor Pico
    # Esto dependerá de cómo estén almacenados los datos en movement_ranges_df
    # A continuación, un ejemplo hipotético:
    df_durante_estimulo['Latencia al Inicio (ms)'] = df_durante_estimulo['Inicio Movimiento (Frame)'] / 100 * 10  # Ejemplo de conversión
    df_durante_estimulo['Duración Total (ms)'] = (df_durante_estimulo['Fin Movimiento (Frame)'] - df_durante_estimulo['Inicio Movimiento (Frame)']) / 100 * 10  # Ejemplo de conversión
    df_durante_estimulo['Latencia al Pico (ms)'] = df_durante_estimulo['Duración Total (ms)'] / 2  # Ejemplo simplificado
    df_durante_estimulo['Valor Pico (velocidad)'] = df_durante_estimulo['submovements'].apply(
        lambda subs: max([sub['v_sm'].max() for sub in subs]) if subs else 0
    )

    # Asegurarse de que 'Forma del Pulso' y 'Duración (ms)' estén bien definidas
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

# Bloque principal del script
# Bloque principal del script
if __name__ == "__main__":
    logging.info("Ejecutando el bloque principal del script.")
    print("Ejecutando el bloque principal del script.")

    # Llamar a collect_velocity_threshold_data y capturar ambos retornos
    counts_df, movement_ranges_all = collect_velocity_threshold_data()
    logging.info(f"Counts DataFrame after collection: {counts_df.shape}")
    print("Counts DataFrame after collection:", counts_df.shape)
    print(counts_df.head())

    # Guardar 'movement_ranges_summary.csv' si hay datos
    if not movement_ranges_all:
        logging.warning("No se encontraron rangos de movimiento para guardar.")
        print("No se encontraron rangos de movimiento para guardar.")
    else:
        movement_ranges_df = pd.DataFrame(movement_ranges_all)
        movement_ranges_df.to_csv(os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv'), index=False)
        logging.info(f"'movement_ranges_summary.csv' guardado con {len(movement_ranges_df)} filas.")
        print(f"'movement_ranges_summary.csv' guardado con {len(movement_ranges_df)} filas.")

    # Verificar si counts_df no está vacío antes de continuar
    if counts_df.empty:
        logging.error("counts_df está vacío. No se puede proceder con el análisis.")
        print("counts_df está vacío. No se puede proceder con el análisis.")
        sys.exit("counts_df está vacío. No se puede proceder con el análisis.")

    # Analizar los mejores bodyparts y estímulos
    analyze_best_bodyparts_and_stimuli(counts_df)

    # Generar heatmap
    plot_heatmap(counts_df)

    # Generar gráficos cronológicos de efectividad
    plot_effectiveness_over_time(counts_df)

    # Generar gráficos comparativos de movement_ranges si hay datos
    if not movement_ranges_all:
        logging.warning("No hay datos de movement_ranges para graficar.")
        print("No hay datos de movement_ranges para graficar.")
    else:
        movement_ranges_df_path = os.path.join(output_comparisons_dir, 'movement_ranges_summary.csv')
        plot_summary_movement_data(movement_ranges_df)
