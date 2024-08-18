# generate_audio.py Documentation 🇺🇸

## Overview

This script generates synthetic speech using the ParlerTTS model from Hugging Face. The input is a text prompt and a voice description, and the output is a WAV file containing the generated audio.

## Script Details

### What the Script Does

The script takes a text prompt and a description of the desired voice characteristics, then uses the ParlerTTS model to generate an audio file that matches these specifications.

### How the Script Works

1. **Device Configuration**: The script automatically detects whether a GPU is available and uses it if possible. Otherwise, it defaults to CPU.

2. **Model and Tokenizer Loading**: The ParlerTTS model and its corresponding tokenizer are loaded from Hugging Face's model hub.

3. **Input Processing**: 
   - The script tokenizes both the voice description and the text prompt.
   - The tokens are then converted to tensors and sent to the appropriate device (GPU or CPU).

4. **Audio Generation**:
   - The model generates audio based on the provided inputs.
   - The generated audio data is then converted into a NumPy array.

5. **Audio Saving**: The resulting audio is saved as a WAV file named `parler_tts_output.wav`.

6. **Output**: The script prints a confirmation message indicating that the audio has been saved.

## Requirements

The script depends on the following Python packages:

- `torch`
- `transformers`
- `soundfile`
- `parler-tts` (via GitHub)

These dependencies are listed in the `requirements.txt` file.

## How to Run the Script

1. **Install Dependencies**:  
   First, install the required packages using pip:

   ```bash
   pip install -r requirements.txt

## Run the Script

Execute the script using Python:

```bash
python generate_audio.py
```
### Output

The generated audio will be saved as `parler_tts_output.wav` in the same directory.

### Estimated Time and Disk Space

- **Download Time**: The first time you run the script, it will download the ParlerTTS model (approximately 1.5 GB). This might take a few minutes depending on your internet speed.
- **Execution Time**: Generating the audio should take only a few seconds, assuming the model is already downloaded.
- **Disk Space**: The downloaded model and tokenizer will occupy around 1.5 GB on your disk. The generated WAV file will be around 100 KB, depending on the length of the audio.

---

# Documentación de `generate_audio.py` 🇨🇴

## Descripción General

Este script genera habla sintética utilizando el modelo ParlerTTS de Hugging Face. La entrada es un texto y una descripción de la voz deseada, y la salida es un archivo WAV que contiene el audio generado.

## Detalles del Script

### ¿Qué Hace el Script?

El script toma un texto y una descripción de las características de la voz deseada, luego usa el modelo ParlerTTS para generar un archivo de audio que coincide con estas especificaciones.

### Cómo Funciona el Script

- **Configuración del Dispositivo**: El script detecta automáticamente si una GPU está disponible y la utiliza si es posible. De lo contrario, utiliza la CPU.
  
- **Carga del Modelo y el Tokenizador**: Se cargan el modelo ParlerTTS y su correspondiente tokenizador desde el repositorio de modelos de Hugging Face.

- **Procesamiento de Entradas**:
  - El script tokeniza tanto la descripción de la voz como el texto.
  - Luego, los tokens se convierten en tensores y se envían al dispositivo adecuado (GPU o CPU).

- **Generación de Audio**:
  - El modelo genera audio basado en las entradas proporcionadas.
  - Los datos de audio generados se convierten en un arreglo de NumPy.

- **Guardado del Audio**: El audio resultante se guarda como un archivo WAV llamado `parler_tts_output.wav`.

- **Salida**: El script imprime un mensaje de confirmación indicando que el audio ha sido guardado.

## Requisitos

El script depende de los siguientes paquetes de Python:

- `torch`
- `transformers`
- `soundfile`
- `parler-tts` (vía GitHub)

Estas dependencias están enumeradas en el archivo `requirements.txt`.

## Cómo Ejecutar el Script

1. **Instalar Dependencias**:  
   Primero, instala los paquetes necesarios usando pip:

   ```bash
   pip install -r requirements.txt

   ### Ejecutar el Script

Ejecuta el script utilizando Python:

```bash
python generate_audio.py```
Salida
El audio generado se guardará como parler_tts_output.wav en el mismo directorio.

Tiempo Estimado y Espacio en Disco
Tiempo de Descarga: La primera vez que ejecutes el script, se descargará el modelo ParlerTTS (aproximadamente 1.5 GB). Esto podría tardar unos minutos dependiendo de la velocidad de tu internet.
Tiempo de Ejecución: La generación de audio debería tomar solo unos segundos, suponiendo que el modelo ya esté descargado.
Espacio en Disco: El modelo y el tokenizador descargados ocuparán alrededor de 1.5 GB en tu disco. El archivo WAV generado ocupará aproximadamente 100 KB, dependiendo de la longitud del audio.