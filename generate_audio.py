import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Configuración del dispositivo (GPU si está disponible)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Cargar el modelo y el tokenizador desde Hugging Face
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Texto de ejemplo y descripción de la voz
prompt = "Asi como digo una cosa digo otra"
description = "una voz de mujer que es dulce y simpatica"

# Tokenizar la descripción y el prompt
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generar el audio
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

# Guardar el audio generado como un archivo WAV
output_filename = "parler_tts_output.wav"
sf.write(output_filename, audio_arr, model.config.sampling_rate)

print(f"Audio generado guardado en {output_filename}")
