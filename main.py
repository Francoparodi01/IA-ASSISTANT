import os
import requests
import time
import speech_recognition as sr
import pyttsx3
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
from pymongo import MongoClient
import pywhatkit
import tinytuya 


# Cargar variables de entorno
load_dotenv()
OPENWEATHER_API = os.getenv("OpenWeather_API")
MongoDB_URI = os.getenv("MongoDB_URI")

# Configuración de tinytuya
DEVICE_ID = os.getenv("DEVICE_ID")  # ID del dispositivo
IP = os.getenv("IP")  # IP 
LOCAL_KEY = os.getenv("LOCAL_KEY")  # Clave secreta del dispositivo
VERSION = 3.3  # Versión del protocolo

device = tinytuya.BulbDevice(DEVICE_ID, IP, LOCAL_KEY)
device.set_version(VERSION)




# MongoDB variables
client = MongoClient(MongoDB_URI) 
db = client["asistente"]
recordatorios_col = db["recordatorios"]

# Verificar si el modelo entrenado existe
model_path = "modelo_chatbot"
if not os.path.exists(model_path):
    print("\u274c No se encontró el modelo. Ejecuta 'train.py' primero.")
    exit()

# Cargar modelo y tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Cargar dataset y crear mapeo de categorías
df = pd.read_csv("dataset_chatbot.csv")
categories = sorted(df["category"].unique())
id_to_category = {idx: category for idx, category in enumerate(categories)}


def clasificar_intencion(texto):
    pred = classifier(texto)
    label_str = pred[0]['label']
    try:
        label_id = int(label_str.split('_')[-1])
    except ValueError:
        return "desconocido"
    return id_to_category.get(label_id, "desconocido")

def agregar_recordatorio(descripcion, fecha_hora):
    recordatorio = {
        "descripcion": descripcion,
        "fecha_hora": fecha_hora
    }
    recordatorios_col.insert_one(recordatorio)
    return "Recordatorio agregado."

def eliminar_recordatorio(descripcion):
    resultado = recordatorios_col.delete_one({"descripcion": descripcion})
    if resultado.deleted_count > 0:
        return "Recordatorio eliminado."
    return "No encontré un recordatorio con esa descripción."


def obtener_clima(ciudad):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={OPENWEATHER_API}&units=metric&lang=es"
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        datos = respuesta.json()
        return f"El clima en {ciudad} es {datos['weather'][0]['description']} con {datos['main']['temp']}°C."
    return "No pude obtener el clima."


def contar_chiste():
    try:
        respuesta = requests.get("https://v2.jokeapi.dev/joke/Any?lang=es&format=txt")
        if respuesta.status_code == 200:
            return respuesta.text.strip()
    except requests.RequestException:
        pass
    return "No pude generar un chiste en este momento."


def buscar_en_youtube(artista):
    try:
        pywhatkit.playonyt(artista)
        return f"Reproduciendo en YouTube Music:"
    except Exception as e:
        return f"No pude reproducir {artista}. Error: {e}"

def fecha_actual():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def controlar_luces(estado=None, brillo=None, color=None):
    try:
        if estado is not None:
            device.set_status(estado, '20')
            return f"Luces {'encendidas' if estado else 'apagadas'}."
        if brillo is not None:
            device.set_status(brillo, '22')
            return f"Brillo ajustado a {brillo}."
        if color is not None:
            device.set_colour(color)
            return f"Color cambiado a {color}."
    except Exception as e:
        return f"Error al controlar las luces: {e}"



def procesar_comando(comando):
    intencion = clasificar_intencion(comando)
    print(f"Comando: {comando} -> Intención detectada: {intencion}")

    if "recordatorio" in comando.lower():
        if "elimina" in comando.lower() or "borrar" in comando.lower():
            descripcion = comando.replace("elimina", "").replace("borrar", "").strip()
            return eliminar_recordatorio(descripcion)
        else:
            partes = comando.split("para")
            if len(partes) > 1:
                descripcion = partes[0].replace("recordatorio", "").strip()
                fecha_hora = partes[1].strip()
                return agregar_recordatorio(descripcion, fecha_hora)
            return "Por favor, especifica la fecha y hora del recordatorio."

    elif intencion == "clima":
        return obtener_clima("Buenos Aires")
    elif intencion == "chiste":
        return "Aquí tienes un chiste: " + contar_chiste()
    elif intencion == "hora":
        return f"La hora actual es {datetime.now().strftime('%H:%M')}"
    elif intencion == "fecha":
        return f"Hoy es {datetime.now().strftime('%d de %B de %Y')}"
    if intencion == "youtube":
        return buscar_en_youtube(comando.replace("youtube", "").strip())
    # Controlar luces
    if intencion == "encender_luces":
        return controlar_luces(estado=True)
    elif intencion == "apagar_luces":
        return controlar_luces(estado=False)
    elif intencion == "ajustar_brillo":
        if "subir" in comando.lower():
            return controlar_luces(brillo=100)
        elif "bajar" in comando.lower():
            return controlar_luces(brillo=10)
    elif intencion == "cambiar_color":
        colores = {"rojo": "ff0000", "azul": "0000ff", "verde": "00ff00", "amarillo": "ffff00", "blanco": "ffffff"}
        for color, hex_code in colores.items():
            if color in comando.lower():
                return controlar_luces(color=hex_code)
    
    # Responder para otras intenciones
    return "No estoy seguro de cómo responder a eso."


def asistente_virtual():
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    print("Asistente virtual activado. Di 'salir' para detenerlo.")
    while True:
        with sr.Microphone() as source:
            print("Esperando comando...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                comando = recognizer.recognize_google(audio, language="es-ES")
                print("Comando recibido:", comando)

                if comando.lower() == "salir":
                    print("Apagando el asistente.")
                    break

                respuesta = procesar_comando(comando)
                print("Respuesta:", respuesta)
                engine.say(respuesta)
                engine.runAndWait()
                time.sleep(2)

            except sr.UnknownValueError:
                print("No pude entender el audio.")
            except sr.WaitTimeoutError:
                print("Tiempo de espera agotado.")


if __name__ == "__main__":
    asistente_virtual()
