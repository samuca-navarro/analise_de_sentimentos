# analise_de_sentimentos
#USO DE ANÁLISE DE SENTIMENTOS COM REGRESSÃO LOGÍSTICA PARA CLASSIFICAÇÃO DE COMENTÁRIOS

#PASSO-A-PASSO

#IMPORTANDO A FERRAMENTA PANDAS
import pandas as pd
import csv

#IMPORTANDO OS DADOS 
dados = pd.read_csv("comentarios_desmatamento_amazonia_v10.csv", sep=";")

#FAZENDO O TRATAMENTO DE DADOS NULOS
dados2 = pd.DataFrame(dados)
enulo = dados2.isnull().sum(0)
print(enulo)

#SEPARANDO AS CLASSES DE TREINO E TESTE
from sklearn.model_selection import train_test_split
treino, teste, classe_treino, classe_teste = train_test_split(dados.comentario, dados.classificacao, test_size = 0.30, random_state = 42)

#VETORIZANDO
from sklearn.feature_extraction.text import CountVectorizer
vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(dados.comentario)
print(bag_of_words.shape)

#IMPORTANDO O NLTK
import nltk
nltk.download("all")

#TOKENIZANDO O NOSSO CORPUS
from nltk import tokenize
token_espaco = tokenize.WhitespaceTokenizer()
todas_palavras = ' '.join([texto for texto in dados.comentario])
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), "Frequência": list(frequencia.values())})

#RETIRADA DE PALAVRAS IRRELEVANTE
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
frase_processada = list()
for opiniao in dados.comentario:
  nova_frase = list()
  palavras_texto = token_espaco.tokenize(opiniao)
  for palavra in palavras_texto:
    if palavra not in palavras_irrelevantes:
      nova_frase.append(palavra)
  frase_processada.append(' '.join(nova_frase))
dados["tratamento_1"] = frase_processada 

#REMOVENDO AS STOPWORDS
from string import punctuation
token_pontuacao = tokenize.WordPunctTokenizer()
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)
pontuacao_stopwords = pontuacao + palavras_irrelevantes
frase_processada = list()
for line in dados["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(word)
    frase_processada.append(' '.join(nova_frase))

dados["tratamento_2"] = frase_processada 


#APLICANDO UNIDECODE
pip install unidecode
import unidecode
sem_acentos = [unidecode.unidecode(texto) for texto in dados["tratamento_2"]]

dados["tratamento_3"] = sem_acentos

frase_processada = list()
for line in dados["tratamento_3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(word)
    frase_processada.append(' '.join(nova_frase))

dados["tratamento_4"] = frase_processada  

#STEMMETIZAÇÃO
stemmer = nltk.RSLPStemmer()
frase_processada = list()
for line in dados["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(stemmer.stem(word))
    frase_processada.append(' '.join(nova_frase))
dados["tratamento_5"] = frase_processada  

#NUVEM DE PALAVRAS DE SENTIMENTOS NEGATIVOS
def nuvem_palavras_pos(texto, coluna_texto):
  texto_negativo = texto.query("classificacao == 'p'")
  todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])
  nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)
  plt.figure(figsize=(10,7))
  plt.imshow(nuvem_palavras, interpolation='bilinear')
  plt.axis("off")
  plt.show
  
#NUVEM DE PALAVRA SENTIMENTOS POSITIVOS
from nltk import tokenize
token_espaco = tokenize.WhitespaceTokenizer()
todas_palavras = ' '.join([texto for texto in dados.comentario])
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), "Frequência": list(frequencia.values())})


nuvem_palavras_pos(dados, "comentario")
nuvem_palavras_neg(dados, "comentario")

#COLOCAR DADOS EM LETRAS MINUSCULAS
frase_processada = list()
for line in dados["tratamento_5"]:
    nova_frase = list()
    line = line.lower()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(word)
    frase_processada.append(' '.join(nova_frase))

dados["tratamento_6"] = frase_processada 

#APLICANDO TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_tratados = tfidf.fit_transform(dados["tratamento_6"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados, dados["classificacao"], random_state=42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratados = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_tratados)

#APLICANDO NGRAMS
tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
vetor_tfidf = tfidf.fit_transform(dados["tratamento_6"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, dados["classificacao"], random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams)

#para verificar os pesos positivos e negativos que o algoritimo está considerando
pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names()
)

pesos.nlargest(10, 0)
pesos.nsmallest(10,0)
