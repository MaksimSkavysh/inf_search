import math
from parse import parse_articles, parse_requests
from inv_index import title_inv_index

# documents = parse_articles()
# index, N, L = title_inv_index(documents)


def get_ftd(token, document):
    return document.count(token) / len(document)


def get_related_documents_list(inv_index, q):
    d_list = []
    for token in q:
        cur_documents = inv_index[token]
        for d in cur_documents:
            if d not in d_list:
                d_list.append(d)
    return d_list


def rsv(q, d, N, L, inv_index):
    b = 0.75
    k1 = 1.2
    L_d = len(d)
    rsv_sum = 0
    for token in q:
        N_t = len(inv_index[token])
        ftd = get_ftd(token, d)
        idf = math.log(1 + (N - N_t + 0.5)/(N_t + 0.5))
        tf_td = (ftd*(k1 + 1)) / (k1*((1-b) + b * (L_d/L)) + ftd)
        rsv_sum = rsv_sum + idf*tf_td
    return rsv_sum


class InvIndex:
    def __init__(self, field):
        self.field = field
        self.documents = parse_articles()
        self.inv_index, self.N, self.L = title_inv_index(self.documents)
        self.questions = parse_requests()

    def rsv(self, q, d):
        return rsv(q, d, self.N, self.L, self.inv_index)

    def check(self):
        q = self.questions[0]
        related_documents = get_related_documents_list(self.inv_index, q['tokens'])
        document = self.documents[related_documents[0]].__getattribute__(self.field)
        print(q['tokens'], document)
        print(self.rsv(q['tokens'], document))


def main():
    inv = InvIndex('title')
    inv.check()


main()
