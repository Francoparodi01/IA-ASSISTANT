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
import redis

# Cargar variables de entorno
load_dotenv()
OPENWEATHER_API = os.getenv("OpenWeather_API")
MongoDB_URI = os.getenv("MongoDB_URI")

# Configuración de dispositivos inteligentes
DEVICE_ID = os.getenv("DEVICE_ID")
IP = os.getenv("IP")
LOCAL_KEY = os.getenv("LOCAL_KEY")
VERSION = 3.3

device = tinytuya.BulbDevice(DEVICE_ID, IP, LOCAL_KEY)
device.set_version(VERSION)

# Conectar a MongoDB
client = MongoClient(MongoDB_URI)
db = client["asistente"]
recordatorios_col = db["recordatorios"]

# Conectar a Redis para almacenar contexto y posibles correcciones
try:
    redis_host = 'redis-15500.c17.us-east-1-4.ec2.redns.redis-cloud.com'
    redis_port = 15500
    redis_db = 0
    redis_password = '7ohozyocmkIHm6lSObxGOpx1uZvizs1R'
    try:
        redis_client = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        print("Conexión exitosa a Redis remoto.")
    except redis.ConnectionError:
        print("Error conectando a Redis remoto. Verifica la configuración.")
    except Exception as e:
        print(f"Error conectando a Redis remoto. {e}")
        redis_client = None
except Exception as e:
    print(f"Error cargando las variables de entorno: {e}")
    redis_client = None

# Cargar modelo y tokenizer
MODEL_PATH = "modelo_chatbot"
if not os.path.exists(MODEL_PATH):
    print("❌ No se encontró el modelo. Ejecuta 'train.py' primero.")
    exit()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Cargar categorías del dataset
df = pd.read_csv("dataset_chatbot.csv")
categories = sorted(df["category"].unique())
id_to_category = {idx: category for idx, category in enumerate(categories)}

def clasificar_intencion(texto):
    """Clasifica la intención del usuario con el modelo entrenado."""
    pred = classifier(texto)
    label_str = pred[0]['label']
    try:
        label_id = int(label_str.split('_')[-1])
    except ValueError:
        return "desconocido"
    return id_to_category.get(label_id, "desconocido")

def obtener_clima(comando):
    """Obtiene el clima de la ciudad mencionada en el comando."""
    palabras = comando.split()
    ciudad = " ".join(palabras[1:]) if len(palabras) > 1 else "Buenos Aires"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={OPENWEATHER_API}&units=metric&lang=es"
    respuesta = requests.get(url)
    
    if respuesta.status_code == 200:
        datos = respuesta.json()
        return f"El clima en {ciudad} es {datos['weather'][0]['description']} con {datos['main']['temp']}°C."
    return "No pude obtener el clima."

def agregar_recordatorio(comando):
    """Agrega un recordatorio en la base de datos."""
    partes = comando.split("para")
    if len(partes) > 1:
        descripcion = partes[0].replace("recordatorio", "").strip()
        fecha_hora_str = partes[1].strip()
        try:
            fecha_hora = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
        except ValueError:
            return "Formato de fecha incorrecto. Usa: YYYY-MM-DD HH:MM"
        
        recordatorios_col.insert_one({"descripcion": descripcion, "fecha_hora": fecha_hora})
        return f"Recordatorio agregado: {descripcion} para el {fecha_hora}."
    return "Por favor, especifica la fecha y la hora del recordatorio."


def hex_a_rgb(hex_code):
    """Convierte un código hexadecimal (por ejemplo, 'ff0000') a valores RGB."""
    hex_code = hex_code.lstrip('#')  # Elimina el símbolo '#' si existe
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def interpretar_comando_luces(comando):
    """
    Interpreta y ejecuta los comandos relacionados con las luces.
    Aquí se buscan órdenes como "Ajusta la luz a nivel medio", "apagá la luz",
    o incluso "pone color cálido".
    Además, se guarda en Redis el contexto del comando y el resultado.
    """
    comando_lwr = comando.lower()
    resultado = "No entendí el comando de ajuste de luces."
    
    # Ajuste de brillo según niveles indicados
    if "nivel bajo" in comando_lwr:
        device.set_status(10, '22')  # Ajusta al 10%
        resultado = "Luz ajustada a nivel bajo."
    elif "nivel medio" in comando_lwr:
        device.set_status(50, '22')  # Ajusta al 50%
        resultado = "Luz ajustada a nivel medio."
    elif "nivel alto" in comando_lwr:
        device.set_status(100, '22')  # Ajusta al 100%
        resultado = "Luz ajustada a nivel alto."
    # Acciones de encender/apagar
    elif any(word in comando_lwr for word in ["encender", "prender", "prende"]):
        device.set_status(True, '20')
        resultado = "Luces encendidas."
    elif any(word in comando_lwr for word in ["apagar", "apaga"]):
        device.set_status(False, '20')
        resultado = "Luces apagadas."
    else:
        # Manejo de otros comandos de luces (como brillo o colores)
        colores = {"rojo": "ff0000", "azul": "0000ff", "verde": "00ff00", "fria": "00ffff",
                  "calida": "ffff00", "blanco": "ffffff", "amarillo": "ffff00", "naranja": "ffa500",
                  "violeta": "800080", "rosa": "ff69b4", "morado": "800080"}
        # Comprobar si el comando contiene un color
        if any(color in comando_lwr for color in colores.keys()):
            # Extraer el color del comando
            for color in colores.keys():
                if color in comando_lwr:
                    r, g, b = hex_a_rgb(colores[color])
                    device.set_colour(r, g, b)
                    resultado = f"Color cambiado a {color}."


        for color, hex_code in colores.items():
            if color in comando_lwr:
                r, g, b = hex_a_rgb(hex_code)  # Convierte hexadecimal a RGB
                device.set_colour(r, g, b)
                resultado = f"Color cambiado a {color}."
                break

    if redis_client:
        redis_client.hset("context:luces", comando_lwr, resultado)

    return resultado

def contar_chiste():
    """Devuelve un chiste obtenido de una API."""
    try:
        respuesta = requests.get("https://v2.jokeapi.dev/joke/Any?lang=es&format=txt")
        return respuesta.text.strip() if respuesta.status_code == 200 else "No pude encontrar un chiste."
    except Exception as e:
        return "No pude obtener un chiste."

def procesar_comando(comando):
    """Detecta la intención del comando y ejecuta la función correspondiente."""
    intencion = clasificar_intencion(comando)
    print(f"Comando: {comando} -> Intención detectada: {intencion}")

    funciones = {
        "clima": obtener_clima,
        "chiste": lambda _: "Aquí tienes un chiste: " + contar_chiste(),
        "hora": lambda _: f"La hora actual es {datetime.now().strftime('%H:%M')}",
        "fecha": lambda _: f"Hoy es {datetime.now().strftime('%d de %B de %Y')}",
        "recordatorio": agregar_recordatorio,
        "luces": interpretar_comando_luces
    }

    # Si la intención detectada no se encuentra, se da una respuesta por defecto
    funcion = funciones.get(intencion, lambda _: "No estoy seguro de cómo responder a eso.")
    return funcion(comando)

def asistente_virtual():
    """Función principal que activa el asistente con reconocimiento de voz."""
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    print("Asistente activado. Di 'salir' para detenerlo.")
    while True:
        with sr.Microphone() as source:
            print("Esperando comando...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                comando = recognizer.recognize_google(audio, language="es-ES")
                print("Comando recibido:", comando)

                if "salir" in comando.lower():
                    print("Apagando el asistente.")
                    break

                respuesta = procesar_comando(comando)
                print("Respuesta:", respuesta)
                engine.say(respuesta)
                engine.runAndWait()

            except sr.UnknownValueError:
                print("No pude entender el audio.")
            except Exception as e:
                print(f"Error procesando el comando: {e}")

if __name__ == "__main__":
    asistente_virtual()
    # Aquí podrías agregar un loop que revise recordatorios o contexto almacenado en Redis.