from kaggle import api 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

api.dataset_download_files("tunguz/online-retail", path='data', unzip=True)
df = pd.read_csv('data/Online_Retail.csv', encoding = "latin-1")
#1
def Limpieza_datos(df):
    #Limpieza
    df['Country'] = df['Country'].replace('EIRE', 'Ireland')

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df_clean = df.dropna(subset=["CustomerID"])
    df_clean = df_clean.drop_duplicates(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"], keep="first")

    codigos_especiales  = ['POST', 'DOT', 'M', 'D', 'C2', 'BANK CHARGES','CRUK', 'PADS', 'ADJUST', 'TEST']
    df_clean = df_clean[~df_clean['StockCode'].isin(codigos_especiales)]

    df_clean.to_csv("data/Online_Retail_Clean.csv")
    return "data/Online_Retail_Clean.csv"
#2 + 3
def Analisis_Graficas(df):
    df_by_client = df.groupby(["InvoiceNo","CustomerID"])[["InvoiceNo","CustomerID","Quantity", "UnitPrice", "Country"]]
    df_by_country = df.groupby(["Country"])["CustomerID"].nunique()
    df_copy = df.copy()
    df_copy["Total"] = df_copy["Quantity"] * df_copy["UnitPrice"]
    df_by_invoice = df_copy.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])[["Total"]].sum()
    rebajas_nom = ["Invierno", "Verano", "CyberMonday", "CyberWeekend", "BoxingDay"]
    rebajas = {
        0: [pd.Timestamp('2011-01-12'), pd.Timestamp('2011-02-08')],
        1: [pd.Timestamp('2011-06-30'), pd.Timestamp('2011-08-08')],
        2: pd.Timestamp('2011-11-28'),
        3: [pd.Timestamp('2011-11-23'), pd.Timestamp('2011-11-27')],
        4: [pd.Timestamp('2010-12-26')]
    }
    #Graficar
    

#4
#def Estadisticas_datos(df):

df = pd.read_csv(Limpieza_datos(df), encoding = "latin-1")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
