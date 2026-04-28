from kaggle import api 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#api.dataset_download_files("tunguz/online-retail", path='data', unzip=True)
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

#2 + 3
def Analisis_Graficas(df):
    df_by_country = df.groupby(["Country"])["CustomerID"].nunique()
    df_by_invoice = df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate", "Year", "Month", "Day", "TimesDay", "SpecialDate"])[["Total"]].sum()
    df_by_client = df_by_invoice.copy()
    df_fechas = df_by_invoice.sort_values(by='InvoiceDate')
    df_by_client = df_by_client.sort_values(by = "CustomerID")
    df_by_client = df_by_client[~(df_by_client['Total'] < 0)]
    df_fechas = df_fechas[~(df_fechas['Total'] < 0)]
    df_country_10 = df_by_country.sort_values(ascending=False).head(5)
    df_country_11 = df_by_country.sort_values().head(len(df_by_country)-5)
    df_country = df_country_10.copy()
    suma_c = pd.Series({'Others': df_country_11.sum()})
    df_country = pd.concat([df_country_10,suma_c])

    df_by_invoice = df_by_invoice.reset_index()

    carpeta = "graficos"

    if not os.path.exists(carpeta):
        os.mkdir(carpeta)

    color_palette = sns.color_palette('rocket')

    plt.pie(df_country, labels = df_country.keys(),autopct = "%0.2f%%", colors = color_palette,rotatelabels=90, labeldistance= 1, shadow=True, radius = 1.2)
    plt.legend(df_country.keys(), loc = 'lower left')
    plt.tight_layout()
    plt.title("Distribucion de los paises en donde se encuentran los clientes")
    plt.savefig("graficos/distribucion_paises.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    sns.lineplot(data = df_fechas, x = 'InvoiceDate', y = "Total", color = 'purple')
    plt.title("Numero de ventas a lo largo de un año 2010-12 a 2011-12")
    plt.tight_layout()
    plt.savefig("graficos/ventas_años.png", bbox_inches='tight', dpi=300)
    plt.close()

    df_plot = df_by_client[df_by_client['Total'] < 1209.334]
    sns.histplot(data=df_plot, x= "Total", bins = 180, color='purple')
    plt.title("Total de ventas realizadas por clientes")
    plt.tight_layout()
    plt.savefig("graficos/ventas_clientes.png", bbox_inches='tight', dpi=300)
    plt.close()

    df_heatmap = df_by_invoice.groupby(['Day', 'TimesDay'])['InvoiceNo'].nunique()
    df_heatmap = df_heatmap.unstack()
    sns.heatmap(df_heatmap, annot=True, fmt="0.0f",linewidth=.5, cmap = color_palette)
    plt.title("Relacion entre el dia y hora de la compra y el total de compras realizadas")
    plt.tight_layout()
    plt.savefig("graficos/dia_hora_ncompras.png", bbox_inches='tight', dpi=300)
    plt.close()

    df_bar = df_by_invoice.groupby(['Year', 'SpecialDate'])['InvoiceNo'].nunique()
    df_bar = df_bar.reset_index()
    sns.barplot(data = df_bar,x = "SpecialDate", y = "InvoiceNo", palette = ['darkslateblue', 'indigo', 'darkviolet', 'purple'])
    plt.title("Numero de compras realizadas en fechas tipicamente altas")
    plt.tight_layout()
    plt.savefig("graficos/fechas_especiales_ncompras.png", bbox_inches='tight', dpi=300)
    plt.close()

#4
#def Estadisticas_datos(df):
ruta = Limpieza_datos(df)
df = pd.read_csv(ruta, encoding = "latin-1")
Analisis_Graficas(df)
print("Listo")