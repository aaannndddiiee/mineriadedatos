from kaggle import api 
import pandas as pd
import re

api.dataset_download_files("tunguz/online-retail", path='data', unzip=True)
df = pd.read_csv('data/Online_Retail.csv', encoding = "latin-1")

#Limpieza
df['Country'] = df['Country'].replace('EIRE', 'Ireland')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df_clean = df.dropna(subset=["CustomerID"])
df_clean = df.drop_duplicates(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"], keep="first")



