from kaggle import api 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
    df_fechas = df_by_invoice.sort_values(by='InvoiceDate')
    rebajas_nom = ["Invierno", "Verano", "CyberMonday", "CyberWeekend", "BoxingDay"]
    rebajas = {
        0: [pd.Timestamp('2011-01-12'), pd.Timestamp('2011-02-08')],
        1: [pd.Timestamp('2011-06-30'), pd.Timestamp('2011-08-08')],
        2: pd.Timestamp('2011-11-28'),
        3: [pd.Timestamp('2011-11-23'), pd.Timestamp('2011-11-27')],
        4: [pd.Timestamp('2010-12-26')]
    }

    df_country_10 = df_by_country.sort_values(ascending=False).head(5)
    df_country_11 = df_by_country.sort_values().head(len(df_by_country)-5)
    df_country = df_country_10.copy()
    suma_c = pd.Series({'Others': df_country_11.sum()})
    df_country = pd.concat([df_country_10,suma_c])
    df_country
    #Graficar

    carpeta = "graficos"

    if not os.path.exists(carpeta):
        os.mkdir(carpeta)

    color_palette = sns.color_palette('rocket')

    plt.pie(df_country, labels = df_country.keys(),autopct = "%0.2f%%", colors = color_palette,rotatelabels=90, labeldistance= 1, shadow=True, radius = 1.2)
    plt.legend(df_country.keys(), loc = 'lower left')
    plt.tight_layout()
    plt.title("Distribucion de los paises en donde se encuentran los clientes")
    plt.savefig("graficos/distribucion_paises.png", bbox_inches='tight', dpi=300)

    sns.histplot(data = df_fechas, x = 'InvoiceDate', bins = 150, color = 'purple')
    plt.title("Numero de ventas a lo largo de un año 2010-12 a 2011-12")
    plt.tight_layout()
    plt.savefig("graficos/ventas_años.png", bbox_inches='tight', dpi=300)


#4
#def Estadisticas_datos(df):

df = pd.read_csv(Limpieza_datos(df), encoding = "latin-1")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
