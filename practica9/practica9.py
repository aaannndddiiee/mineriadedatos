import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from kaggle import api

api.dataset_download_files("tunguz/online-retail", path='data', unzip=True)
df = pd.read_csv('data/Online_Retail.csv', encoding = "latin-1")

def TextAnalysis():
    df = pd.read_csv('data/Online_Retail.csv', encoding = "latin-1", usecols=['Description'])
    df = df.dropna()

    text = ' '.join(df['Description'])

    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False, stopwords=set(['SET', 'BOX', 'PACK', 'LARGE', 'SMALL', 'OF'])).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("WordCloud.png", bbox_inches='tight', dpi=300)
    plt.close()

TextAnalysis()