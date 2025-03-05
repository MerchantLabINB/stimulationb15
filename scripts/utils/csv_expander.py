import os
import pandas as pd
import re
import glob
import logging

# Rutas definidas (como en tu código original)
stimuli_info_path = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\tablas\Stimuli_information.csv'
csv_folder = r'C:\Users\samae\Documents\GitHub\stimulationb15\DeepLabCut\xv_lat-Br-2024-10-02\videos'

# Función para encontrar el CSV asociado usando la cámara lateral y el start frame (del ensayo)
def encontrar_csv(camara_lateral, start_frame):
    try:
        # Construir el patrón de búsqueda:
        # Se espera que el archivo comience con:
        # <camara_lateral>_<start_frame>_
        pattern = f"{camara_lateral}_{start_frame}_*filtered.csv"
        search_pattern = os.path.join(csv_folder, pattern)
        logging.debug(f"Searching for CSV with pattern: {search_pattern}")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            csv_path = matching_files[0]
            logging.debug(f"Found CSV: {csv_path}")
            return csv_path
        else:
            logging.warning(f"CSV not found for camera={camara_lateral}, start_frame={start_frame}")
            return None
    except Exception as e:
        logging.error(f"Error in encontrar_csv: {e}")
        return None

# Cargar el CSV original de estímulos
stimuli_info = pd.read_csv(stimuli_info_path, encoding='utf-8')

# Filtrar las filas que se deben descartar (no se procesarán ni se incluirán en el CSV expandido)
stimuli_info = stimuli_info[stimuli_info['Descartar'].str.strip() != "Sí"]

def expandir_ensayos(row, col_name="Start frame (lateral)", delim=","):
    """
    Para una fila del CSV:
      - Separa los valores de la columna 'Start frame (lateral)' (usando coma como delimitador).
      - Para cada valor (ensayo) se crea una nueva fila con:
           • "trial_number": número de ensayo (orden en esa fila)
           • "csv_path": path completo del CSV encontrado usando la función 'encontrar_csv'
           • "csv_filename": nombre del archivo (basename) del CSV encontrado
           
    Se usa la columna "Camara Lateral" para formar el nombre base y el valor individual (ensayo)
    se utiliza como start frame para buscar el CSV.
    """
    valor = str(row[col_name]).strip()
    ensayos = [x.strip() for x in valor.split(delim) if x.strip() != '']
    nuevas_filas = []
    
    camara_lateral = str(row["Camara Lateral"]).strip()
    for idx, ensayo in enumerate(ensayos, start=1):
        csv_path_real = encontrar_csv(camara_lateral, ensayo)
        csv_filename = os.path.basename(csv_path_real) if csv_path_real is not None else None
        nueva_fila = row.copy()
        nueva_fila[col_name] = ensayo  # Se asigna el start frame individual
        nueva_fila["trial_number"] = idx
        nueva_fila["csv_path"] = csv_path_real
        nueva_fila["csv_filename"] = csv_filename
        nuevas_filas.append(nueva_fila)
    return nuevas_filas

# Procesar todas las filas para expandir los ensayos
filas_expandida = []
for _, row in stimuli_info.iterrows():
    filas_expandida.extend(expandir_ensayos(row, col_name="Start frame (lateral)"))

# Crear el DataFrame extendido
stimuli_info_expandido = pd.DataFrame(filas_expandida)

# Guardar el nuevo CSV en el mismo directorio que el original (o donde prefieras)
nuevo_csv_path = os.path.join(os.path.dirname(stimuli_info_path), 'Stimuli_information_expanded.csv')
stimuli_info_expandido.to_csv(nuevo_csv_path, index=False, encoding='utf-8')

print(f"CSV expandido guardado en: {nuevo_csv_path}")
