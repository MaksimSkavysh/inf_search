import typing
from article import Article
from collections import defaultdict


def inv_index(articles: typing.List[Article], field):
    index = defaultdict(list)
    for article in articles:
        for token in article.__getattribute__(field):
            if not index[token]:
                index[token] = []
            if article.index not in index[token]:
                index[token].append(article.index)
    print(index['transverse'])
    return index


def title_inv_index(articles: typing.List[Article]):
    return inv_index(articles, 'title')


def abstract_inv_index(articles: typing.List[Article]):
    return inv_index(articles, 'abstract')

