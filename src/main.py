import math
from parse import parse_articles, parse_requests
from inv_index import title_inv_index
import typing
from article import Article

# documents = parse_articles()
# index, N, L = title_inv_index(documents)


def get_ftd(token, document):
    return 1


def get_related_documents_list(inv_index, q):
    d_list = []
    for token in q:
        cur_documents = inv_index[token]
        for d in cur_documents:
            if d not in d_list:
                d_list.append(d)
    return d_list


class InvIndex:
    def __init__(self):
        self.documents = parse_articles()
        self.inv_index, self.N, self.L = title_inv_index(self.documents)
        self.questions = parse_requests()

    def check(self):
        q = self.questions[0]
        related_documents = get_related_documents_list(self.inv_index, q['tokens'])
        print(related_documents)


# def rsv(q, d):
#     b = 0.75
#     k1 = 1.2
#     Ld = len(d)
#
#     rsv_sum = 0
#     for token in q:
#         N_t = len(index[token])
#         ftd = get_ftd(token, d)
#         idf = math.log(1 + (N - N_t + 0.5)/(N_t + 0.5))
#         tf_td = (ftd*(k1 + 1)) / (k1*((1-b) + b * (Ld/L)) + ftd)
#         rsv_sum = rsv_sum + idf*tf_td
#     return rsv_sum


def main():
    inv = InvIndex()
    inv.check()


main()

