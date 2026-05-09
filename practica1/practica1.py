from kaggle import api 
import pandas as pd
import os
from tabulate import tabulate

#El dataset es demasiado pesado y github no lo acepta 

api.dataset_download_files("tunguz/online-retail", path='data', unzip=True)
df = pd.read_csv('data/Online_Retail.csv', encoding = "latin-1")

def categorizarHora(fila):
    if fila.hour >= 0 and fila.hour < 6:
        return 'EarlyMorn'
    if fila.hour >= 6 and fila.hour < 12:
        return 'Morning'
    if fila.hour >= 12 and fila.hour < 18:
        return 'Afternoon'
    if fila.hour >= 18 and fila.hour < 24:
        return 'Evening'
    return "Unknown"

def cambiar_Fecha_formato(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df["TimesDay"] = df["InvoiceDate"].apply(categorizarHora)
    df['Day'] = df['InvoiceDate'].dt.day_name()
    df['Month'] = df['InvoiceDate'].dt.month_name()
    df['Year'] = df['InvoiceDate'].dt.year
    return df 

def fechas_Especiales(fila):
    rebajas_nom = ["Invierno", "Verano", "CyberMonday", "CyberWeekend", "BoxingDay"]
    rebajas_fechas = {
        0: [pd.Timestamp('2011-01-12'), pd.Timestamp('2011-02-08')],
        1: [pd.Timestamp('2011-06-30'), pd.Timestamp('2011-08-08')],
        2: pd.Timestamp('2011-11-28'),
        3: [pd.Timestamp('2011-11-23'), pd.Timestamp('2011-11-27')],
        4: [pd.Timestamp('2010-12-26')]
    }
    if fila >= rebajas_fechas[0][0] and fila < rebajas_fechas[0][1]:
        return rebajas_nom[0]
    if fila >= rebajas_fechas[1][0] and fila < rebajas_fechas[1][1]:
        return rebajas_nom[1]
    if fila == rebajas_fechas[2]:
        return rebajas_nom[2]
    if fila >= rebajas_fechas[3][0] and fila < rebajas_fechas[3][1]:
        return rebajas_nom[3]
    if fila == rebajas_fechas[4]:
        return rebajas_nom[4]
    return "NoSpecial"

def crear_Columnas(df):
    df['Total'] = df['UnitPrice'] * df['Quantity']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = cambiar_Fecha_formato(df)
    df['SpecialDate'] = df['InvoiceDate'].apply(fechas_Especiales)
    return df

#1
def Limpieza_datos(df):
    #Limpieza
    df = df.drop(columns = "Description")
    df['Country'] = df['Country'].replace('EIRE', 'Ireland')

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df_clean = df.dropna(subset=["CustomerID"])
    df_clean = df_clean.drop_duplicates(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"], keep="first")

    codigos_especiales  = ['POST', 'DOT', 'M', 'D', 'C2', 'BANK CHARGES','CRUK', 'PADS', 'ADJUST', 'TEST']
    df_clean = df_clean[~df_clean['StockCode'].isin(codigos_especiales)]

    df_clean = crear_Columnas(df_clean)

    df_clean.to_csv("data/Online_Retail_Clean.csv")
    return "data/Online_Retail_Clean.csv"

print("Limpieza Datos...\n")
ruta = Limpieza_datos(df)
df = pd.read_csv(ruta, encoding = "latin-1")