import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate

def train_model(dataset_csv="dataset_chatbot.csv", model_name="dccuchile/bert-base-spanish-wwm-cased"):
    model_path = "modelo_chatbot"

    # Verificar si ya existe el modelo entrenado
    if os.path.exists(model_path):
        print("✅ Modelo ya entrenado encontrado. No es necesario entrenarlo de nuevo.")
        return

    print("📂 Cargando dataset para entrenamiento...")
    df = pd.read_csv(dataset_csv)

    categories = sorted(df["category"].unique())
    category_to_id = {category: idx for idx, category in enumerate(categories)}
    df["label"] = df["category"].map(category_to_id)

    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(category_to_id))
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
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
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("✅ Modelo guardado en:", model_path)

if __name__ == "__main__":
    train_model()
