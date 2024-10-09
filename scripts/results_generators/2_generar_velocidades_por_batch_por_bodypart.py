import os
import sys
import pandas as pd
import numpy as np
from math import sqrt
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

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
output_pdf_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\pdfs'
font_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\fonts\Arial-Unicode-Regular.ttf'

# Asegurarse de que el directorio de salida existe
if not os.path.exists(output_pdf_dir):
    os.makedirs(output_pdf_dir)

# Cargar archivos CSV
stimuli_info = pd.read_csv(stimuli_info_path)
segmented_info = pd.read_csv(segmented_info_path)

# Filtrar entradas donde 'Descartar' es 'No'
stimuli_info = stimuli_info[stimuli_info['Descartar'] == 'No']

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
    'DedoMedio': 'pink',      # Usamos 'pink' en minúsculas
    'Braquiradial': 'Greys',
    'Bicep': 'YlOrBr'
}

body_parts = list(body_parts_colors.keys())

# Función para calcular la distancia entre dos puntos
def calcular_distancia(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Función para encontrar el archivo CSV correspondiente basado en la cámara y el nombre del segmento
def encontrar_csv(camara_lateral, nombre_segmento):
    try:
        for file_name in os.listdir(csv_folder):
            if camara_lateral in file_name and nombre_segmento in file_name and file_name.endswith('.csv'):
                return os.path.join(csv_folder, file_name)
        logging.warning(f'Archivo CSV no encontrado para la cámara: {camara_lateral}, segmento: {nombre_segmento}')
        return None
    except Exception as e:
        logging.error(f'Error al acceder a los archivos CSV: {e}')
        return None

# Función de suavizado usando media móvil
def moving_average(data, window_size=11):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Función para calcular velocidades y posiciones para cada articulación con suavizado
def calcular_velocidades(csv_path):
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        velocidades = {}
        posiciones = {}

        for part in body_parts:
            x_col = (df.columns[1][0], part, 'x')
            y_col = (df.columns[1][0], part, 'y')
            likelihood_col = (df.columns[1][0], part, 'likelihood')
            df_filtered = df[df[likelihood_col] > 0.1]

            if df_filtered.empty:
                velocidades[part] = []
                posiciones[part] = {'x': [], 'y': []}
                continue

            velocidad_part = []
            for i in range(1, len(df_filtered)):
                x1, y1 = df_filtered.iloc[i - 1][x_col], df_filtered.iloc[i - 1][y_col]
                x2, y2 = df_filtered.iloc[i][x_col], df_filtered.iloc[i][y_col]
                distancia = calcular_distancia(x1, y1, x2, y2)
                delta_t = 1 / 100  # 100 fps
                velocidad = distancia / delta_t
                velocidad_part.append(velocidad)

            # Aplicar suavizado con una ventana más grande
            velocidad_part = moving_average(velocidad_part, window_size=21)
            velocidades[part] = velocidad_part
            posiciones[part] = {
                'x': df_filtered[x_col].values,
                'y': df_filtered[y_col].values
            }

        return velocidades, posiciones
    except Exception as e:
        logging.error(f'Error al calcular velocidades para CSV: {csv_path}, Error: {e}')
        return {}, {}

def calcular_aceleraciones(velocidades):
    aceleraciones = {}
    for part, vel in velocidades.items():
        # La aceleración es la diferencia entre velocidades consecutivas
        aceleracion_part = np.diff(vel, prepend=np.nan)  # Agregamos np.nan al inicio
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

# Función para plotear los gráficos incluyendo la leyenda del estímulo y ajustes de estilo
# ... (resto de las importaciones y configuraciones)

# Función para plotear los gráficos incluyendo la leyenda del estímulo y ajustes de estilo
def plot_graphs(velocidades_per_bodypart, aceleraciones_per_bodypart, positions_per_bodypart,
                amplitude_list, duration_list, segmento, body_part, estimulo_params_text,
                start_frame, current_frame):
    # Verificaciones iniciales (sin cambios)
    positions_list = positions_per_bodypart[body_part]
    if not any(len(pos['x']) > 0 for pos in positions_list):
        logging.warning(f"No hay datos de posiciones para la articulación {body_part}.")
        return '', np.array([])

    velocities_list = velocidades_per_bodypart[body_part]
    if not any(len(vel) > 0 for vel in velocities_list):
        logging.warning(f"No hay datos de velocidades para la articulación {body_part}.")
        return '', np.array([])

    aceleraciones_list = aceleraciones_per_bodypart[body_part]
    if not any(len(acc) > 0 for acc in aceleraciones_list):
        logging.warning(f"No hay datos de aceleraciones para la articulación {body_part}.")
        return '', np.array([])

    # Configuración de la figura y subplots
    fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69),
                             gridspec_kw={'height_ratios': [4, 4, 4, 2]})
    ax1, ax2, ax3, ax4 = axes

    # Gráfico de trayectorias en ax1
    num_ensayos = len(positions_per_bodypart[body_part])
    base_cmap_name = body_parts_colors[body_part]
    base_cmap = plt.get_cmap(base_cmap_name)
    colors = [base_cmap(1 - (i / max(num_ensayos - 1, 1)))
              for i in range(num_ensayos)]

    print(f"Plotting trajectories for body part: {body_part}, number of ensayos: {num_ensayos}")

    for i, positions in enumerate(positions_per_bodypart[body_part]):
        if len(positions['x']) > 0:
            frames = np.arange(len(positions['x']))
            ax1.plot(frames, positions['x'], color=colors[i],
                     linestyle='-', label=f'Ens. {i+1} - x')
            ax1.plot(frames, positions['y'], color=colors[i],
                     linestyle='--', label=f'Ens. {i+1} - y')

    ax1.set_title(f'Trayectorias - {body_part}', fontsize=12)
    ax1.set_xlabel('Frames', fontsize=10)
    ax1.set_ylabel('Coordenadas (px)', fontsize=10)
    ax1.set_xlim(0, 400)
    ax1.legend(loc='upper right', fontsize='x-small', ncol=3)
    ax1.tick_params(axis='both', labelsize=8)

    # Configuración de ticks y gridlines
    ax1.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))
    ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', which='major', length=7)
    ax1.tick_params(axis='x', which='minor', length=4)

    # Sombrear la región del estímulo
    ax1.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

    # Gráfico de velocidades en ax2
    max_length = 400
    velocities_aligned = []
    for vel in velocities_list:
        vel_padded = np.pad(vel, (0, max_length - len(vel)),
                            'constant', constant_values=np.nan)
        velocities_aligned.append(vel_padded[:max_length])

    velocities_array = np.vstack(velocities_aligned)
    mean_velocity = np.nanmean(velocities_array, axis=0)

    for i, vel in enumerate(velocities_aligned):
        ax2.plot(vel, color=colors[i], label=f'Ens. {i+1}')

    ax2.plot(mean_velocity, color='#00BFFF', label='Media de ensayos', linewidth=1)
    ax2.set_title(f'Velocidades - {body_part}', fontsize=12)
    ax2.set_xlabel('Frames', fontsize=10)
    ax2.set_ylabel('Velocidad (unidades/segundo)', fontsize=10)
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 2000)  # Fijar el límite superior en 2000
    ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
    ax2.tick_params(axis='both', labelsize=8)

    # Configuración de ticks y gridlines
    ax2.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))
    ax2.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='x', which='major', length=7)
    ax2.tick_params(axis='x', which='minor', length=4)

    ax2.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

    # Gráfico de aceleraciones en ax3
    aceleraciones_aligned = []
    for acc in aceleraciones_list:
        acc_padded = np.pad(acc, (0, max_length - len(acc)),
                            'constant', constant_values=np.nan)
        aceleraciones_aligned.append(acc_padded[:max_length])

    aceleraciones_array = np.vstack(aceleraciones_aligned)
    mean_aceleracion = np.nanmean(aceleraciones_array, axis=0)

    # Definir un umbral para considerar cambios significativos en la aceleración
    acc_threshold = 1e-2  # Puedes ajustar este valor según tus datos

    for i, acc in enumerate(aceleraciones_aligned):
        frames = np.arange(len(acc))
        # Crear máscara donde la aceleración es significativa
        significant_acc_mask = (np.abs(acc) > acc_threshold) & (~np.isnan(acc))
        # Reemplazar valores no significativos por NaN
        acc_filtered = np.where(significant_acc_mask, acc, np.nan)
        # Graficar solo los valores significativos
        ax3.plot(frames, acc_filtered, color=colors[i], label=f'Ens. {i+1}')

    # Aplicar el mismo filtrado a la media de aceleraciones
    frames_mean = np.arange(len(mean_aceleracion))
    mean_significant_acc_mask = (np.abs(mean_aceleracion) > acc_threshold) & (~np.isnan(mean_aceleracion))
    mean_acc_filtered = np.where(mean_significant_acc_mask, mean_aceleracion, np.nan)
    ax3.plot(frames_mean, mean_acc_filtered, color='#FF69B4',
             label='Media de ensayos', linewidth=1)

    ax3.set_title(f'Aceleraciones - {body_part}', fontsize=12)
    ax3.set_xlabel('Frames', fontsize=10)
    ax3.set_ylabel('Aceleración (unidades/segundo²)', fontsize=10)
    ax3.set_xlim(0, 400)
    ax3.legend(loc='upper right', fontsize='x-small', ncol=3)
    ax3.tick_params(axis='both', labelsize=8)

    # Configuración de ticks y gridlines
    ax3.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))
    ax3.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    ax3.tick_params(axis='x', which='major', length=7)
    ax3.tick_params(axis='x', which='minor', length=4)

    ax3.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

    # Gráfico del estímulo en ax4
    x_vals = [start_frame]
    y_vals = [0]
    current_frame_stimulus = start_frame

    for amp, dur in zip(amplitude_list, duration_list):
        frames_to_add = dur  # El tiempo ya está en frames
        next_frame = current_frame_stimulus + frames_to_add
        x_vals.extend([current_frame_stimulus, next_frame])
        y_vals.extend([amp / 1000, amp / 1000])  # Dividir por 1000 para obtener μA
        current_frame_stimulus = next_frame

    ax4.step(x_vals, y_vals, color='blue', where='post', linewidth=1)
    ax4.set_xlabel('Frames', fontsize=10)
    ax4.set_ylabel('Amplitud (μA)', fontsize=10)
    ax4.set_xlim(0, 400)
    ax4.set_ylim(-160, 160)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.tick_params(axis='both', labelsize=8)

    # Configuración de ticks y gridlines
    ax4.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax4.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))
    ax4.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    ax4.tick_params(axis='x', which='major', length=7)
    ax4.tick_params(axis='x', which='minor', length=4)

    # Añadir el texto de los parámetros del estímulo
    ax4.text(0.95, 0.95, estimulo_params_text, transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax4.axvspan(start_frame, current_frame, color='blue', alpha=0.3)

    # Ajustar espacios entre subplots
    plt.subplots_adjust(hspace=0.4)

    # Guardar la imagen
    graph_image_path = f"temp_graph_{segmento}_{body_part}.png"
    plt.savefig(graph_image_path, dpi=300)
    plt.close()

    return graph_image_path, mean_velocity

def generar_pdf_por_fila(index, row, global_max_velocity):
    try:
        print(f"\nGenerando PDF para la fila {index}, Cámara Lateral: {row['Cámara Lateral']}")
        logging.info(f"Generando PDF para la fila {index}, Cámara Lateral: {row['Cámara Lateral']}")
        camara_lateral = row['Cámara Lateral']

        if pd.notna(camara_lateral):
            matching_segment = segmented_info[segmented_info['CarpetaPertenece'].str.contains(camara_lateral, na=False)]
            if not matching_segment.empty:
                matching_segment_sorted = matching_segment.sort_values(by='NumeroOrdinal')

                # Crear diccionarios para almacenar velocidades, posiciones y aceleraciones por articulación
                velocidades_per_bodypart = {part: [] for part in body_parts}
                positions_per_bodypart = {part: [] for part in body_parts}
                aceleraciones_per_bodypart = {part: [] for part in body_parts}
                mean_velocities_per_bodypart = {}  # Para almacenar las medias de velocidades

                for _, segment_row in matching_segment_sorted.iterrows():
                    nombre_segmento = segment_row['NombreArchivo'].replace('.mp4', '').replace('lateral_', '')
                    csv_path = encontrar_csv(camara_lateral, nombre_segmento)
                    if csv_path:
                        velocidades, posiciones = calcular_velocidades(csv_path)
                        # Calcular aceleraciones
                        aceleraciones = calcular_aceleraciones(velocidades)
                        for part in body_parts:
                            if len(velocidades.get(part, [])) > 0:
                                velocidades_per_bodypart[part].append(velocidades[part])
                                positions_per_bodypart[part].append(posiciones[part])
                            if len(aceleraciones.get(part, [])) > 0:
                                aceleraciones_per_bodypart[part].append(aceleraciones[part])

                # Generar estímulo
                amplitude_list, duration_list = generar_estimulo_desde_parametros(
                    row['Forma del Pulso'],
                    row['Amplitud (μA)'] * 1000,      # Convertir a μA
                    row['Duración (ms)'] * 1000,      # Convertir a μs
                    row['Frecuencia (Hz)'],
                    200, compensar=True)              # Duración del pulso en μs

                # Información del estímulo para el gráfico y el PDF
                estimulo_params_text = f"Forma: {row['Forma del Pulso']}, Amplitud: {row['Amplitud (μA)']} μA, Duración: {row['Duración (ms)']} ms, Frecuencia: {row['Frecuencia (Hz)']} Hz"

                # Calcular start_frame y current_frame para sombrear correctamente
                start_frame = 100  # Asignamos el valor de inicio
                current_frame = start_frame + sum(duration_list)  # Calculamos el frame final del estímulo

                # Crear PDF
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_font("ArialUnicode", "", font_path, uni=True)
                pdf.set_font("ArialUnicode", size=10)

                # Información del estímulo para el PDF (imprimir al inicio)
                info_estimulo = f"Información del estímulo:\n{estimulo_params_text}\n"

                # Añadir página inicial con la información del estímulo
                pdf.add_page()
                pdf.multi_cell(0, 10, info_estimulo)

                # Generar y añadir el gráfico de trayectorias x vs y
                trajectory_graph_path = plot_overall_trajectories(positions_per_bodypart, start_frame, current_frame)
                pdf.image(trajectory_graph_path, x=10, y=30, w=190)
                os.remove(trajectory_graph_path)

                # Por cada articulación, generar el gráfico y añadir al PDF
                # En generar_pdf_por_fila...
                for body_part in body_parts:
                    if len(velocidades_per_bodypart[body_part]) > 0:
                        # Generar gráfico
                        graph_image_path, mean_velocity = plot_graphs(
                            velocidades_per_bodypart, aceleraciones_per_bodypart, positions_per_bodypart,
                            amplitude_list, duration_list, camara_lateral, body_part, estimulo_params_text,
                            start_frame, current_frame)

                        if graph_image_path and os.path.exists(graph_image_path):
                            # Guardar la media de velocidad
                            mean_velocities_per_bodypart[body_part] = mean_velocity

                            # Añadir página al PDF
                            pdf.add_page()
                            pdf.multi_cell(0, 10, f"Articulación: {body_part}")

                            # Añadir imagen al PDF
                            pdf.image(graph_image_path, x=10, y=40, w=190)

                            # Eliminar archivo temporal
                            os.remove(graph_image_path)
                        else:
                            logging.warning(f"No se generó gráfico para la articulación {body_part}.")
                    else:
                        logging.warning(f"No hay datos suficientes para la articulación {body_part}.")

                # Añadir página final con las medias de velocidades
                pdf.add_page()
                pdf.multi_cell(0, 10, "Media de velocidades por articulación")

                # Generar gráfico de medias de velocidades
                plt.figure(figsize=(11.69, 8.27))
                ax = plt.gca()
                max_length = 400

                for idx, body_part in enumerate(body_parts):
                    if body_part in mean_velocities_per_bodypart:
                        mean_velocity = mean_velocities_per_bodypart[body_part]
                        if len(mean_velocity) > 0:
                            frames = np.arange(len(mean_velocity))
                            ax.plot(frames, mean_velocity, label=body_part)
                            # Añadir línea punteada horizontal de la velocidad promedio
                            avg_velocity = np.nanmean(mean_velocity)
                            line_color = ax.get_lines()[-1].get_color()
                            ax.axhline(avg_velocity, linestyle='--', color=line_color)
                            # Añadir etiqueta con el valor fuera del área del gráfico
                            # Usar coordenadas relativas para colocar el texto fuera del área de datos
                            ax.annotate(f"{avg_velocity:.2f}",
                                        xy=(1.01, avg_velocity),
                                        xycoords=('axes fraction', 'data'),
                                        xytext=(5, 0),
                                        textcoords='offset points',
                                        va='center', ha='left',
                                        fontsize=8, color=line_color)
                    else:
                        logging.warning(f"No hay datos de velocidad promedio para la articulación {body_part}.")

                # Sombrear la región del estímulo
                ax.axvspan(start_frame, current_frame, color='blue', alpha=0.1)

                ax.set_title('Medias de Velocidades por Articulación')
                ax.set_xlabel('Frames')
                ax.set_ylabel('Velocidad (unidades/segundo)')
                ax.set_xlim(0, max_length)
                ax.set_ylim(0, 2000)  # Fijar el límite en y

                # Configuración de ticks y gridlines
                ax.xaxis.set_major_locator(plt.MultipleLocator(50))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 50 == 0 else ''))
                ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
                ax.tick_params(axis='x', which='major', length=7)
                ax.tick_params(axis='x', which='minor', length=4)

                # Colocar la leyenda de las partes del cuerpo dentro del gráfico (arriba a la derecha)
                ax.legend(loc='upper right', fontsize='small')

                # Colocar el texto de los parámetros del estímulo abajo a la derecha, pegado al eje x
                ax.text(0.99, 0.02, estimulo_params_text, transform=ax.transAxes, fontsize=8,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                # Asegurarse de que no haya elementos superpuestos
                plt.tight_layout()
                summary_graph_path = f"temp_summary_{camara_lateral}.png"
                plt.savefig(summary_graph_path, bbox_inches='tight')
                plt.close()




                # Añadir gráfico al PDF
                pdf.image(summary_graph_path, x=10, y=40, w=190)
                os.remove(summary_graph_path)

                # Asegurarse de que el directorio de salida existe
                if not os.path.exists(output_pdf_dir):
                    os.makedirs(output_pdf_dir)

                # Guardar PDF
                safe_filename = camara_lateral.replace("/", "-").replace("\\", "-").replace(":", "-")
                output_pdf_path = os.path.join(output_pdf_dir, f'{safe_filename}_stimuli.pdf')
                pdf.output(output_pdf_path, 'F')
                print(f'PDF generado: {output_pdf_path}')
                logging.info(f'PDF generado: {output_pdf_path}')
            else:
                logging.warning(f"No se encontró segmento coincidente para la cámara: {camara_lateral}")
        else:
            logging.warning(f"Información de cámara lateral faltante para la fila {index}")
    except Exception as e:
        print(f"Error al procesar la fila {index}: {e}")
        logging.error(f"Error al procesar la fila {index}: {e}")

def plot_overall_trajectories(positions_per_bodypart, start_frame, current_frame):
    plt.figure(figsize=(8.27, 11.69))  # Tamaño A4 en orientación vertical
    ax = plt.gca()
    
    for body_part in body_parts:
        ensayos_positions = positions_per_bodypart[body_part]
        num_ensayos = len(ensayos_positions)
        base_cmap_name = body_parts_colors[body_part]
        base_cmap = plt.get_cmap(base_cmap_name)
        colors = [base_cmap(1 - (i / max(num_ensayos - 1, 1))) for i in range(num_ensayos)]
        
        for i, positions in enumerate(ensayos_positions):
            x = positions['x'][int(start_frame):int(current_frame)]
            y = positions['y'][int(start_frame):int(current_frame)]
            if len(x) > 0 and len(y) > 0:
                ax.plot(x, y, color=colors[i], alpha=0.7)
    
    ax.set_title('Trayectorias x vs y durante el estímulo', fontsize=12)
    ax.set_xlabel('Posición x (px)', fontsize=10)
    ax.set_ylabel('Posición y (px)', fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')

    # Añadir líneas de cuadrícula en ambos ejes
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Crear una leyenda personalizada para las articulaciones
    handles = [plt.Line2D([0], [0], color=plt.get_cmap(body_parts_colors[part])(0.5), lw=2) for part in body_parts]
    labels = body_parts
    ax.legend(handles, labels, title='Articulaciones', loc='upper right', fontsize='small', ncol=1)
    
    plt.tight_layout()
    
    # Guardar la imagen
    trajectory_graph_path = "temp_overall_trajectory.png"
    plt.savefig(trajectory_graph_path, dpi=300)
    plt.close()
    
    return trajectory_graph_path


# Cálculo de global_max_velocity
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
                    velocidades, _ = calcular_velocidades(csv_path)
                    for vel in velocidades.values():
                        if len(vel) > 0:
                            max_vel = np.nanmax(vel)  # Usar np.nanmax para manejar NaNs
                            if max_vel > global_max_velocity:
                                global_max_velocity = max_vel

logging.info(f"global_max_velocity calculado: {global_max_velocity}")

# Generar PDF para cada fila en stimuli_info
for index, row in stimuli_info.iterrows():
    generar_pdf_por_fila(index, row, global_max_velocity)
