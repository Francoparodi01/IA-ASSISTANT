import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

# Configuración
MODEL_PATH = "modelo_chatbot"
MODEL_NAME = "bertin-project/bertin-roberta-base-spanish"
DATASET_PATH = "dataset_chatbot.csv"


def preprocess_text(text):
    """Preprocesa el texto eliminando espacios y convirtiéndolo a minúsculas."""
    return text.lower().strip()


def train_model():
    """Entrena o reentrena un modelo de clasificación de intención."""
    if not os.path.exists(DATASET_PATH):
        print("❌ No se encontró el dataset:", DATASET_PATH)
        return

    print("📂 Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)
    if "text" not in df.columns or "category" not in df.columns:
        print("❌ El dataset debe contener las columnas 'text' y 'category'")
        return

    df["text"] = df["text"].apply(preprocess_text)
    df["label_text"] = df["category"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "unknown")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label_text"])
    label_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Etiquetas:", label_dict)

    dataset = Dataset.from_pandas(df[["text", "label"]])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        encoding = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )
        encoding["labels"] = examples["label"]
        return encoding

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔹 Dispositivo:", device)
    num_labels = len(le.classes_)

    class_weights = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    if os.path.exists(MODEL_PATH):
        print("🔄 Cargando modelo existente...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
    else:
        print("🚀 Entrenando modelo desde cero...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    model.to(device)
    model.loss_fn = loss_fn

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=10,
    )

    metric = load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("🚀 Iniciando entrenamiento...")
    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Modelo guardado en:", MODEL_PATH)


if __name__ == "__main__":
    train_model()
    print("🔄 Entrenamiento completado.")
