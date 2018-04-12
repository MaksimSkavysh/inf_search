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
from normalize import normalize


def sum_lines(x, y):
    return x + y.replace('\n', ' ')


def parse_data(f, prefix, handler=sum_lines):
    s = f.readline()
    title = ''
    while prefix not in s and not s == '':
        title = handler(title, s)
        s = f.readline()
    return title, s


NUMBER_OF_ABSTRACTS = 1400
# NUMBER_OF_ABSTRACTS = 700
INDEX_PREFIX = '.I'
TITLE_PREFIX = '.T'
AUTHORS_PREFIX = '.A'
INFO_PREFIX = '.B'
ABSTRACT_PREFIX = '.W'


def get_index(s):
    return int(s.replace(INDEX_PREFIX, ''))


def get_title(f):
    title, s = parse_data(f, AUTHORS_PREFIX)
    return title


def get_authors(f):
    authors, s = parse_data(f, INFO_PREFIX)
    return authors


def get_info(f):
    info = parse_data(f, ABSTRACT_PREFIX)
    return info


def get_abstract(f):
    abstract, s = parse_data(f, INDEX_PREFIX)
    return abstract, s


def parse_articles(doc_file, parse_abstract=False, verbose=False):
    documents = {}
    with open(doc_file) as f:
        s = f.readline()
        while s:
            index = get_index(s)
            f.readline()
            title = get_title(f)
            authors = get_authors(f)
            info = get_info(f)
            abstract, s = get_abstract(f)
            # articles.append(Article(index, normalize(title), authors, info, normalize(abstract)))

            if parse_abstract:
                documents[index] = normalize(abstract)
            else:
                documents[index] = normalize(title)
    if verbose:
        for article in documents:
            print(article)

    return documents


def get_question(f):
    info, s = parse_data(f, INDEX_PREFIX)
    return info, s


def parse_requests(verbose=0):
    requests = []
    with open('./data/cran.qry') as f:
        s = f.readline()
        i = 0
        while s:
            i = i + 1
            index = get_index(s)
            f.readline()
            question, s = get_question(f)
            normalized = normalize(question)
            # print(index, normalize(normalized))
            requests.append({'index': i, 'tokens': normalized})
    return requests
