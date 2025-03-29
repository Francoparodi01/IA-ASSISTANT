import pandas as pd 

df = pd.read_csv("dataset_chatbot.csv")
print(df["category"].value_counts())

print(df["text"].value_counts())
print(df["category"].value_counts(normalize=True))
print(df["text"].value_counts(normalize=True))
print(df["text"].isnull().sum())
print(df["category"].isnull().sum())
print(df["text"].isna().sum())
    