import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate
from sklearn.preprocessing import MultiLabelBinarizer

# Definir ruta del modelo entrenado
MODEL_PATH = "modelo_chatbot"
MODEL_NAME = "bertin-project/bertin-roberta-base-spanish"

def train_model(dataset_csv="dataset_chatbot.csv"):
    """Entrena un modelo de clasificación de intención con múltiples etiquetas."""
    # Verificar si el modelo ya está entrenado
    if os.path.exists(MODEL_PATH):
        print("✅ Modelo ya entrenado encontrado. No es necesario entrenarlo de nuevo.")
        return

    print("📂 Cargando dataset para entrenamiento...")
    df = pd.read_csv(dataset_csv)
    
    # Normalizar texto
    df["text"] = df["text"].str.lower().str.strip()
    df["labels"] = df["category"].apply(lambda x: x.split(";"))
    
    # MultiLabel binarizer para entrenamiento multi-etiqueta
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(df["labels"])
    labels_list = mlb.classes_
    df_encoded = pd.DataFrame(y_encoded, columns=labels_list)
    df = pd.concat([df["text"], df_encoded], axis=1)
    
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    df_train, df_eval = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔹 Dispositivo para entrenamiento:", device)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_list))
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (logits > 0).astype(int)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    print("🚀 Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Modelo guardado en:", MODEL_PATH)

if __name__ == "__main__":
    train_model()
