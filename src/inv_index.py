import typing
from article import Article
from collections import defaultdict


def inv_index(articles: typing.List[Article], field):
    index = defaultdict(list)
    N = 0
    total_words = 0
    for key in articles:
        N = N + 1
        article = articles[key]
        tokens_list = article.__getattribute__(field)
        total_words = total_words + len(tokens_list)
        for token in tokens_list:
            if not index[token]:
                index[token] = []
            if article.index not in index[token]:
                index[token].append(article.index)
            # index[token].append(article.index)
    # print(index['.'])
    # print(N)
    # print(total_words)
    L = total_words / N
    return index, N, L


def title_inv_index(articles: typing.List[Article]):
    return inv_index(articles, 'title')


def abstract_inv_index(articles: typing.List[Article]):
    return inv_index(articles, 'abstract')

