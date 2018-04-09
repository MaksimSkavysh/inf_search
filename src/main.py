from parse import parse
from inv_index import title_inv_index
import typing
from article import Article

articles: typing.List[Article] = parse()
index = title_inv_index(articles)
