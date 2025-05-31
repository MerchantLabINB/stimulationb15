# stimulationb15

Repositorio oficial asociado a la tesis **"Microestimulación eléctrica en la corteza motora del primate y la evaluación cinemática de las conductas evocadas"**. Este conjunto de scripts y herramientas fue desarrollado para ejecutar tareas de estimulación intracortical prolongada (ICMS), adquisición de datos conductuales en primates, análisis cinemático automatizado con DeepLabCut y análisis estadístico riguroso de los movimientos evocados.

## 🧠 Descripción general

Este repositorio contiene:

- GUI para configuración y ejecución de sesiones de estimulación eléctrica intracortical.
- Scripts para detección de LED TTL y sincronización con video.
- Scripts de preprocesamiento y segmentación automática de videos.
- Código para inferencia de posturas con DeepLabCut y análisis cinemático de velocidad.
- Herramientas para ajuste de submovimientos (modelos gaussianos y minimum jerk).
- Scripts de análisis estadístico y visualización de resultados.

La infraestructura fue desarrollada en Python 3.7.11, compatible con sistemas Linux y Windows.

---

## ⚙️ Ambiente de desarrollo y reproducibilidad

Para asegurar la reproducibilidad de los análisis, se utilizó un entorno de Conda definido explícitamente mediante el archivo `environment.yml`, incluido en este repositorio.

### 🔧 Dependencias clave:

- `numpy`, `pandas`, `scipy`: para análisis numérico y manipulación de datos.
- `matplotlib`, `seaborn`: para visualización de resultados.
- `deeplabcut`: estimación de postura sin marcadores.
- `statsmodels`, `scikit-learn`: para análisis estadístico y modelado.
- `opencv-python-headless`: manipulación de video.
- `ffmpeg`, `imageio`: codificación y decodificación de videos.

### 💻 Instrucciones de instalación:

```bash
conda env create -f environment.yml
conda activate stimulationb15

## 🗂️ Estructura del repositorio: `scripts/`

La carpeta `scripts` contiene los módulos principales del proyecto, organizados según su función:

### `data_cleaning/`
Contiene scripts para depurar y transformar tablas generadas por DeepLabCut. Incluye validaciones, eliminación de valores nulos y filtrado cinemático.

### `dlc_scripts/`
Funciones para manejar los resultados de DeepLabCut: segmentación por ensayo, alineación con TTL, detección de artefactos y etiquetado de partes del cuerpo.

### `GUI_pattern_generator/`
Contiene interfaces gráficas (Tkinter) para diseñar patrones de estimulación (rectangular, rampa, rombo) exportables a archivos de configuración del generador de estímulo.

### `results_generators/`
Scripts para calcular métricas cinemáticas (velocidad, latencia, duración, submovimientos). También incluye generación de gráficos y tablas para análisis estadístico (ANOVA y post-hoc).

### `utils/`
Funciones auxiliares para manejo de rutas, logs, operaciones matemáticas especializadas (ajuste de gaussianas, minimum jerk) y reutilizables en todo el pipeline.

### `video_preprocessing/`
Herramientas para recodificar y recortar videos experimentales. Incluye scripts para detectar LEDs TTL, convertir videos a 100 fps y preparar clips para análisis con DeepLabCut.

---

## 📦 Archivos clave adicionales

- `environment.yml`: archivo con las dependencias exactas necesarias para recrear el ambiente.
- `README.md`: este archivo.
- `estructura_repositorio_*.txt`: árbol de directorios con organización del código.

---

## 📄 Cita sugerida

Si este código resulta útil en tu investigación, por favor cita la tesis original:

**Bustos Alarcon, B. (2025).**  
*Microestimulación eléctrica en la corteza motora del primate y la evaluación cinemática de las conductas evocadas.*  
Tesis de maestría, Universidad Nacional Autónoma de México.  
Repositorio: [https://github.com/MerchantLabINB/stimulationb15](https://github.com/MerchantLabINB/stimulationb15)

---

## 📬 Contacto

Para dudas, reproducibilidad o colaboración:

**Bruno Bustos Alarcón**  
Correo: brunobustos@neurotechx.com  
Sitio web del laboratorio: [merchantlab.inb.unam.mx](http://merchantlab.inb.unam.mx)
