import os
from datetime import datetime

def find_repo_root(startpath):
    """
    Busca la carpeta raíz del repositorio, identificada por la presencia de un archivo característico,
    como '.gitignore' o '.git'.
    
    Parámetros:
    - startpath: Ruta inicial desde la cual comenzar a buscar
    
    Retorna:
    - root_path: Ruta de la carpeta raíz del repositorio
    """
    current_path = startpath
    while current_path != os.path.dirname(current_path):
        if os.path.exists(os.path.join(current_path, '.gitignore')) or os.path.exists(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    return None

def save_directory_structure_to_txt(startpath, output_file, level=0):
    """
    Recorre recursivamente la estructura de directorios a partir de 'startpath' e imprime solo las carpetas no vacías,
    y guarda la salida en un archivo .txt.
    
    Parámetros:
    - startpath: Ruta al directorio de inicio (directorio raíz del repositorio)
    - output_file: Ruta al archivo donde se guardará la estructura
    - level: Nivel actual de profundidad en el directorio (usado para formatear la salida)
    """
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(startpath):
            # Nivel de indentación basado en la profundidad
            indent = ' ' * 4 * (root.count(os.sep) - level)
            
            # Revisar si la carpeta actual tiene subcarpetas o archivos
            if dirs or files:
                f.write(f"{indent}{os.path.basename(root)}/\n")
            
            # Recorrer únicamente las subcarpetas que no están vacías
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                if os.listdir(dir_path):  # Imprimir solo si la carpeta no está vacía
                    f.write(f"{indent}    {directory}/\n")

# Obtener el directorio donde está el script
script_path = os.path.dirname(os.path.abspath(__file__))

# Buscar la raíz del repositorio
repo_root = find_repo_root(script_path)

if repo_root:
    print(f"Raíz del repositorio encontrada en: {repo_root}")
    
    # Crear la carpeta de logs si no existe
    log_dir = r'C:\Users\samae\Documents\GitHub\stimulationb15\data\logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Crear un archivo con timestamp en el nombre
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_txt = os.path.join(log_dir, f'estructura_repositorio_{timestamp}.txt')
    
    # Guardar la estructura en el archivo
    save_directory_structure_to_txt(repo_root, output_txt)
    print(f"Estructura del repositorio guardada en: {output_txt}")
else:
    print("No se pudo encontrar la raíz del repositorio. Asegúrate de que hay un archivo .gitignore o .git en la carpeta raíz.")
