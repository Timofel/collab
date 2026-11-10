import pandas as pd

path = r"C:\Users\user\Desktop\detects_ns\archive\dataset_2\main_dataset_2\metadata.csv"
data = pd.read_csv(path)

print(data.head())
data.info()
print(data.isna().sum())

from pathlib import Path

print("Папка скрипта:", Path(__file__).parent)
print("Рабочая папка:", Path.cwd())
print("CSV найден:", Path("путь_к_metadata.csv").exists())