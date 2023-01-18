import numpy as np
import pandas as pd
import string
import nltk
import requests
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



#remove as pontuações 
def RemovePontuacao (frase):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    frase = frase.translate(table)
    return frase

#divide as frases em palavras
def Tokenize (frase):
    frase = frase.lower()
    frase = nltk.word_tokenize(frase, 'portuguese')
    return frase

#retorna apenas o radical da palavra
def Stemming (frase):
    stemmer = RSLPStemmer()
    sentence = []
    for palavra in frase:
        sentence.append(stemmer.stem(palavra.lower()))
    return sentence

#remove as stopwords(o, a, e...)
def RemoveStopWords (frase):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    sentence = []
    for palavra in frase:
        if palavra not in stopwords:
            sentence.append(palavra)
    return sentence


#chama todas as funções acima
def tratamentoTexto (frases):
    lista = []
    for frase in frases:
        frase = RemovePontuacao(frase)
        frase = Tokenize(frase)
        frase = RemoveStopWords(frase)
        frase = Stemming(frase)
        frase = " ".join(frase) # formar a frase novamente
        lista.append (frase)
    return lista
    
def tratamentoFraseUnica (frase):
    frase = RemovePontuacao(frase)
    frase = Tokenize(frase)
    frase = Stemming(frase)
    frase = RemoveStopWords(frase)
    frase = " ".join(frase) # formar a frase novamente
    return frase


#main
file = "https://raw.githubusercontent.com/GabrielOliveiraBR/TCC/main/DadosJuntos.csv"
data = pd.read_csv(file, delimiter=';', encoding="ISO-8859-1") # ler o arquivo csv e armazena na variável data
data = data.dropna() # remove linhas vazias
classificacao = data.racismo # armazena na variável categoria apenas os itens da coluna categoria do dataset
frases = data.text # armazena na variável frases apenas os itens da coluna frases do dataset
textos = pd.Series(tratamentoTexto(frases)) #transformando a lista em uma Series
df = pd.DataFrame(data=dict(text=textos, classificacao=classificacao)) #criar DF com o texto já tratado
#print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.classificacao) #criando os DFs de treinamento e teste

#print(f'{len(X_train)} registros para treinamento')
#print(f'{len(X_test)} registros para teste' )


# Criação do modelo
modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Treinamento
modelo.fit(X_train, y_train)
# Predição das categorias dos textos de teste
y_pred = modelo.predict(X_test)

# Parâmetros para medir os resultados da aplicação do algoritmo
#cm = confusion_matrix(y_test, y_pred)
#cr = classification_report(y_test, y_pred)
#accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)

# Teste de uma frase isolada
fraseusuario = input(str('frase teste: '))
#frase = tratamentoFraseUnica(fraseusuario)
frase = tratamentoFraseUnica('NÃO SOU PRECONCEITUOSO até tenho uma empregada negra')
predicao = modelo.predict([frase])
print(predicao)