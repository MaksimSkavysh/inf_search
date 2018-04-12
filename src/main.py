import math
import numpy as np
from parse import parse_articles, parse_requests
from inv_index import get_inv_index
from eval_fuction import check_eval

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


def calculate_simple_idf(N, N_t):
    return math.log((N - N_t) / N_t)


def calculate_idf(N, N_t ):
    return math.log(1 + (N - N_t + 0.5) / (N_t + 0.5))


def calculate_rsv(q, d, N, L, b, k1, inv_index):
    L_d = len(d)
    rsv_sum = 0
    for token in q:
        N_t = len(inv_index[token])
        ftd = get_ftd(token, d)
        idf = calculate_idf(N, N_t )
        tf_td = (ftd*(k1 + 1)) / (k1*((1-b) + b * (L_d/L)) + ftd)
        rsv_sum = rsv_sum + idf*tf_td
    return rsv_sum


class InvIndex:
    def __init__(self, use_abstracts=False, b=0.75, k1=1.2):
        self.documents = parse_articles(use_abstracts)
        self.inv_index, self.N, self.L = get_inv_index(self.documents)
        self.questions = parse_requests()
        self.relevance = {}
        self.b = b
        self.k1 = k1

    def calculate_rsv(self, q, d):
        return calculate_rsv(
            q=q,
            d=d,
            N=self.N,
            L=self.L,
            b=self.b,
            k1=self.k1,
            inv_index=self.inv_index
        )

    def search_relevance(self, q):
        related_documents_indexes = get_related_documents_list(self.inv_index, q['tokens'])

        rsv_list = []
        for d_index in related_documents_indexes:
            document = self.documents[d_index]
            rsv = self.calculate_rsv(q['tokens'], document)
            rsv_list.append((d_index, rsv))

        sorted_rsv_list = sorted(rsv_list, key=lambda tup: -tup[1])
        relevant_documents = [t[0] for t in sorted_rsv_list[:10]]
        # print(sorted_rsv_list)
        # print(q['index'], relevant_documents)
        return relevant_documents

    def search(self):
        for question in self.questions:
            self.relevance[question['index']] = self.search_relevance(question)
        # print(self.relevance)

    def print(self):
        with open('./data/answer', 'w') as f:
            for q_index in self.relevance:
                for d_index in self.relevance[q_index]:
                    f.write(str(q_index) + ' ' + str(d_index) + '\n')


def main():
    use_abstracts = False

    print('\n default params: ')
    inv = InvIndex(use_abstracts=use_abstracts, b=0.75, k1=1.2)
    inv.search()
    inv.print()
    check_eval()

    # for b in np.arange(0, 1.25, 0.25):
    #     for k1 in np.arange(1.2, 2.1, 0.1):
    #         print('\n params: ', 'k1=', k1, ' b=', b)
    #         inv = InvIndex(use_abstracts=False, b=b, k1=k1)
    #         inv.search()
    #         inv.print()
    #         check_eval()

    print('\n best params: k1=1.2 b=0.0')
    inv = InvIndex(use_abstracts=use_abstracts, b=0.0, k1=1.2)
    inv.search()
    inv.print()
    check_eval()


main()
