# Распарсить файл с текстами документов. Нам понадобятся 3 поля:
# ID (.I), заголовок (.T) и аннотация (.W). Первое предложение анно-
# тации совпадает с заголовком. Тексты содержат знаки препинания
# и стоп-слова, слова не нормализованы (за исключением приведе-
# ния к нижнему регистру), поэтому необходимо реализовать неко-
# торый нормализатор текста, который будет использоваться перед
# индексацией. Нормализатор осуществляет стемминг и(или) лемма-
# тизацию, а также исключает стоп-слова.

# .I <N>
# .T
# title text
# .A
# authors
# .B
# some info
# .W
# full abstra

import typing
from article import Article
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import LancasterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


NUMBER_OF_ABSTRACTS = 1400
# NUMBER_OF_ABSTRACTS = 1400
INDEX_PREFIX = '.I'
TITLE_PREFIX = '.T'
AUTHORS_PREFIX = '.A'
INFO_PREFIX = '.B'
ABSTRACT_PREFIX = '.W'


lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
# st = PorterStemmer()
st = EnglishStemmer()
# st = LancasterStemmer()


def normalize(text):
    # tokens = [st.stem(s) for s in word_tokenize(text)]
    tokens = [lemmatizer.lemmatize(s) for s in word_tokenize(text)]
    return [t for t in tokens if t not in stop]


def sum_lines(x, y):
    return x + y.replace('\n', ' ')


def parse_data(f, prefix, handler=sum_lines):
    s = f.readline()
    title = ''
    while prefix not in s and not s == '':
        title = handler(title, s)
        s = f.readline()
    return title, s


def get_index(s):
    return int(s.replace(INDEX_PREFIX, ''))


def get_title(f):
    title, s = parse_data(f, AUTHORS_PREFIX)
    return title


def get_authors(f):
    authors, s = parse_data(f, INFO_PREFIX)
    return authors


def get_info(f):
    info, s = parse_data(f, ABSTRACT_PREFIX)
    return info


def get_abstract(f):
    abstract, s = parse_data(f, INDEX_PREFIX)
    return abstract, s


def parse(verbose=0):
    articles: typing.List[Article] = []
    with open('./data/cran.all.1400') as f:
        s = f.readline()
        for i in range(0, NUMBER_OF_ABSTRACTS, 1):
            index = get_index(s)
            f.readline()
            title = get_title(f)
            authors = get_authors(f)
            info = get_info(f)
            abstract, s = get_abstract(f)
            articles.append(Article(index, normalize(title), authors, info, normalize(abstract)))

    if verbose > 0:
        for article in articles:
            print(article.index)
            if verbose > 1:
                print(article.title)
            if verbose > 2:
                print(article.abstract)
    return articles
