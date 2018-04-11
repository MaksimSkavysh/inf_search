import math
from parse import parse_articles, parse_requests
from inv_index import title_inv_index
import typing
from article import Article

documents: typing.List[Article] = parse_articles()
index, N, L = title_inv_index(documents)
requests = parse_requests()


def get_ftd():
    return 1


def rsv(q, d):
    b = 0.75
    k1 = 1.2
    Ld = len(d)

    rsv_sum = 0
    for token in q:
        N_t = len(index[token])
        ftd = get_ftd()
        idf = math.log(1 + (N - N_t + 0.5)/(N_t + 0.5))
        tf_td = (ftd*(k1 + 1)) / (k1*((1-b) + b * (Ld/L)) + ftd)
        rsv_sum = rsv_sum + idf*tf_td
    return rsv_sum


def get_related_documents(q):
    for token in q:
        print(token, index[token])


def main():
    request = requests[0]

    q_index = request['index']
    q = request['question']
    get_related_documents(q)


main()
