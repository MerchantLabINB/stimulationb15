# Patrones de Estimulación para Experimentos

Este archivo `Stimulation.py` contiene el código utilizado para generar patrones de estimulación durante los experimentos. Los patrones definidos son variaciones de formas de ondas y parámetros que se aplican para estudiar diferentes respuestas a estímulos.

## Patrones Disponibles

A continuación se detallan los patrones de estimulación que están definidos y utilizados en los experimentos:

| Patrón             | Duración (ms) |
|--------------------|---------------|
| Rectangular        | 500           |
| Rectangular        | 1000          |
| Rombos             | 500           |
| Rombos             | 750           |
| Rombos             | 1000          |
| Rampa Ascendente   | 1000          |
| Rombos Triple      | 700           |

Estos patrones están predefinidos en el código y se pueden ajustar según las necesidades del experimento.

## Uso

Para ejecutar los patrones de estimulación, el script `Stimulation.py` debe ser utilizado dentro del entorno del experimento con los parámetros definidos por el investigador. Asegúrate de que los estímulos estén configurados correctamente antes de iniciar el experimento.

### Ejemplo

```python
patterns = [
    ("rectangular", 500), ("rectangular", 1000), ("rombo", 500),
    ("rombo", 750), ("rombo", 1000), ("rampa ascendente", 1000), ("rombo triple", 700)
]
