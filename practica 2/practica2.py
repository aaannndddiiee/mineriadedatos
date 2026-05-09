from kaggle import api 
import pandas as pd
import os
from tabulate import tabulate


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

def Descriptive_Statistics(df, df_country):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    with open('DescriptiveStatistics.txt', 'a') as file:
        df_no_c = df.copy()
        df_no_c = df_no_c[df_no_c['Total'] >= 0]
        file.write("==== DATASET INFO ====\n")
        tabla = []
        filas, columnas = df.shape
        file.writelines([f"Numero de registros: {filas}\n", f"Numero de columnas: {columnas}\n"])
        nombres_columnas = df.columns.values
        tipos_datos = df.dtypes
        for i in range(len(nombres_columnas)):
            tabla.append([nombres_columnas[i], tipos_datos[i]])
        file.write(tabulate(tabla, headers=["Nombre Columna", "Tipo de dato"], tablefmt="grid"))

        file.write("\n==== DESCRIPTIVE STATISTICS ====\n")
        columnas = ["Quantity", "UnitPrice", "Total"]
        tabla = []
        for i in range(3):
            tabla.append([columnas[i], df[columnas[i]].mean().round(2), df[columnas[i]].std().round(2),df[columnas[i]].min().round(2), df[columnas[i]].quantile(0.25).round(2),df[columnas[i]].quantile(0.5).round(2), df[columnas[i]].quantile(0.75).round(2)])
        file.write(tabulate(tabla, headers=["Columnas", "Media", "Desviacion Estandar", "Valor Minimo", "Valor Maximo", "Percentil 25%", "Percentil 50%", "Percentil 75%"], tablefmt="grid"))

        file.write("\n==== CUSTOMER ANALYSIS ====\n")
        df_country.reset_index()
        num_por_pais = df_country.to_list()
        paises = df_country.keys().to_list()
        tabla = []
        for i in range(len(paises)):
            tabla.append([paises[i], num_por_pais[i]])
        file.write(tabulate(tabla, headers=["Pais", "Clientes por pais"], tablefmt="grid"))
        file.write("\nTop 10 clientes que mas gastan\n")
        df_by_client = df_no_c.groupby(["CustomerID"])["Total"].sum()
        df_by_client = df_by_client.reset_index()
        df_by_client = df_by_client.sort_values(by="Total", ascending=False)
        valores = df_by_client["CustomerID"].values
        total = df_by_client["Total"].values
        tabla = []
        for i in range(10):
            tabla.append([valores[i], total[i]])
        file.write(tabulate(tabla, headers=["CustomerID", "Total de compras"], tablefmt="grid"))
        file.write("\nResumen de clientes\n")
        file.write(f"Total de clientes: {df_by_client['CustomerID'].count()}\n")
        file.write(f"Promedio de total de compras: {df_by_client['Total'].mean().round(2)}\n")
        file.write(f"Percentil de compras total 25%: {df_by_client['Total'].quantile(0.25).round(2)}\n")
        file.write(f"Percentil de compras total 75%: {df_by_client['Total'].quantile(0.75).round(2)}\n")
    
        file.write("\n==== COUNTRY ANALYSIS ====\n")
        df_paises = df_no_c[df_no_c['Country'] != 'United Kingdom']
        df_compras= df_paises.groupby(["Country"])["InvoiceNo"].nunique()
        df_compras = df_compras.reset_index()
        df_totales= df_paises.groupby(["Country"])["Total"].sum()
        df_totales = df_totales.reset_index()
        df_promedio= df_paises.groupby(["Country"])["Total"].mean()
        df_promedio = df_promedio.reset_index()
        df_compras = df_compras.sort_values(by = 'InvoiceNo', ascending=False).head(10)
        df_totales = df_totales.reset_index()
        df_totales = df_totales.sort_values(by = 'Total', ascending=False).head(10)
        df_promedio = df_promedio.reset_index()
        df_promedio = df_promedio.sort_values(by = 'Total', ascending=False).head(10)
        file.write("Top 10 paises\n")
        file.write("\nTotal de facturas por pais (sin UK)\n")
        tabla = []
        for i in range(10):
            tabla.append([df_compras['Country'].values[i], df_compras['InvoiceNo'].values[i]])
        file.write(tabulate(tabla, headers=["Pais", "Total de facturas"], tablefmt="grid"))
        file.write("\nTotal de compra por pais (sin UK)\n")
        tabla = []
        for i in range(10):
            tabla.append([df_totales['Country'].values[i], df_totales['Total'].values[i]])
        file.write(tabulate(tabla, headers=["Pais", "Total de compra"], tablefmt="grid"))
        file.write("\nPromedio de compra por pais (sin UK)\n")
        tabla = []
        for i in range(10):
            tabla.append([df_promedio['Country'].values[i], df_promedio['Total'].values[i]])
        file.write(tabulate(tabla, headers=["Pais", "Promedio de compra"], tablefmt="grid"))

        file.write("\n==== TIME ANALYSIS ====\n")
        file.write("Promedio de compra por dia\n")
        df_dia = df_no_c.groupby(["Day"])["Total"].mean()
        df_dia = df_dia.reset_index()
        tabla = []
        for i in range(len(df_dia['Day'].values)):
            tabla.append([df_dia['Day'].values[i], df_dia['Total'].values[i].round(2)])
        file.write(tabulate(tabla, headers=["Dia", "Promedio de compra"], tablefmt="grid"))
        df_com = df_no_c.groupby(df['InvoiceDate'].dt.date)['InvoiceNo'].nunique()
        file.write(f"\nPromedio de facturas por dia: {df_com.mean().round(2)}\n")
        file.write(f"Numero minimo facturas realizadas: {df_com.min().round(2)}\n")
        file.write(f"Numero maximo facturas realizadas: {df_com.max().round(2)}\n")
        file.write("Promedio de facturas por dia de la semana\n")
        df_com = df_com.reset_index()
        df_com['InvoiceDate'] = pd.to_datetime(df_com["InvoiceDate"])
        df_com['Day'] = df_com['InvoiceDate'].dt.day_name()
        df_com['Month'] = df_com['InvoiceDate'].dt.month_name()
        df_dia = df_com.groupby(["Day"])["InvoiceNo"].mean()
        df_dia = df_dia.reset_index()
        tabla = []
        for i in range(len(df_dia['Day'].values)):
            tabla.append([df_dia['Day'].values[i], df_dia['InvoiceNo'].values[i].round(0)])
        file.write(tabulate(tabla, headers=["Dia", "Promedio de factuars"], tablefmt="grid" ))
        df_mes = df_com.groupby(["Month"])["InvoiceNo"].mean()
        df_mes = df_mes.reset_index()
        file.write("\nPromedio de facturas por mes\n")
        tabla = []
        for i in range(len(df_mes['Month'].values)):
            tabla.append([df_mes['Month'].values[i], df_mes['InvoiceNo'].values[i].round(0)])
        file.write(tabulate(tabla, headers=["Mes", "Promedio de factuars"], tablefmt="grid" ))

        file.write("\n==== SPECIAL DATES ANALYSIS ====\n")
        df_especial = df_no_c.groupby(["SpecialDate", "Year"])["InvoiceNo"].nunique()
        df_especial = df_especial.reset_index()
        df_especial = df_especial[df_especial['Year'] != 2010] 
        file.write("Numero de facturas en las fechas especiales en 2010\n")
        tabla = []
        for i in range(len(df_especial['SpecialDate'].values)):
            tabla.append([df_especial['SpecialDate'].values[i], df_especial['InvoiceNo'].values[i]])
        file.write(tabulate(tabla, headers=["Fecha Especial", "Total de facturas"], tablefmt="grid" ))

        file.write("\n==== RETURN  ANALYSIS ====\n")
        df_cancel = df[df['Total'] < 0]
        df_cancel = df_cancel.groupby(["CustomerID", "Country", "Total"])['InvoiceNo'].nunique()
        df_cancel = df_cancel.reset_index()
        file.write(f"Total de cancelaciones: {df_cancel['InvoiceNo'].count()}\n")
        file.write(f"Monto total de cancelado: {df_cancel['Total'].sum().round(2)}\n")
        file.write(f"Promedio de cancelado: {df_cancel['Total'].mean().round(2)}\n")
        file.write(f"Monto minimo cancelado: {df_cancel['Total'].min().round(2)}\n")
        file.write(f"Monto maximo cancelado: {df_cancel['Total'].max().round(2)}\n")
        file.write(f"Percentil de cancelado total 25%: {df_cancel['Total'].quantile(0.25).round(2)}\n")
        file.write(f"Percentil de cancelado total 75%: {df_cancel['Total'].quantile(0.75).round(2)}\n")
        file.write(f"Proporcion de cancelaciones: {(df_cancel['InvoiceNo'].count() / df['InvoiceNo'].nunique()):.2f}%\n")
        df_canc_cliente = df_cancel.groupby(df['CustomerID'])['Total'].sum()
        df_canc_cliente = df_canc_cliente.reset_index()
        df_canc_cliente = df_canc_cliente.sort_values(by="Total", ascending=True).head(10)
        file.write("Top 10 clientes con mayor monto cancealdo\n")
        tabla = []
        for i in range(10):
            tabla.append([df_canc_cliente['CustomerID'].values[i], df_canc_cliente['Total'].values[i].round(2)])
        file.write(tabulate(tabla, headers=["CustomerID", "Total cancelado"], tablefmt="grid"))
        file.write("\nTop 10 clientes con mayor numero de facturas canceladas\n")
        df_canc_cliente = df_cancel.groupby(df['CustomerID'])['InvoiceNo'].count()
        df_canc_cliente = df_canc_cliente.reset_index()
        df_canc_cliente = df_canc_cliente.sort_values(by = 'InvoiceNo', ascending=False).head(10)
        tabla = []
        for i in range(10):
            tabla.append([df_canc_cliente['CustomerID'].values[i], df_canc_cliente['InvoiceNo'].values[i]])
        file.write(tabulate(tabla, headers=["CustomerID", "Total de facturas"], tablefmt="grid"))
        file.write("\nTop 10 paises con mas monto total cancelado\n")
        df_canc_paises = df_cancel.groupby(df['Country'])['Total'].sum()
        df_canc_paises = df_canc_paises.reset_index()
        df_canc_paises = df_canc_paises.sort_values(by = 'Total').head(10)
        tabla = []
        for i in range(10):
            tabla.append([df_canc_paises['Country'].values[i], df_canc_paises['Total'].values[i].round(2)])
        file.write(tabulate(tabla, headers=["Pais", "Total de cancelado"], tablefmt="grid"))
        file.write("\nTop 10 paises con mas facturas canceladas\n")
        df_canc_paises = df_cancel.groupby(df['Country'])['InvoiceNo'].count()
        df_canc_paises = df_canc_paises.reset_index()
        df_canc_paises = df_canc_paises.sort_values(by = 'InvoiceNo', ascending=False).head(10)
        tabla = []
        for i in range(10):
            tabla.append([df_canc_paises['Country'].values[i], df_canc_paises['InvoiceNo'].values[i]])
        file.write(tabulate(tabla, headers=["Pais", "Total de facturas"], tablefmt="grid"))

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
    Descriptive_Statistics(df, df_country)

print("Limpieza Datos...\n")
ruta = Limpieza_datos(df)
df = pd.read_csv(ruta, encoding = "latin-1")
print("Analisis")
Analisis_Graficas(df)