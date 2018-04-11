import math
from parse import parse_articles, parse_requests
from inv_index import get_inv_index

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


def calculate_rsv(q, d, N, L, inv_index):
    # 1.0788933600108224
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
        self.inv_index, self.N, self.L = get_inv_index(self.documents)
        self.questions = parse_requests()
        self.relevance = {}

    def calculate_rsv(self, q, d):
        return calculate_rsv(q, d, self.N, self.L, self.inv_index)

    def search_relevance(self, q):
        # q = self.questions[0]
        related_documents_indexes = get_related_documents_list(self.inv_index, q['tokens'])

        # document = self.documents[related_documents_indexes[0]]
        # print(q['tokens'], document)
        # print(self.rsv(q['tokens'], document))

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
        print(self.relevance)

    def print(self):
        # f = open('workfile', 'w')
        with open('./data/answer', 'w') as f:
            for q_index in self.relevance:
                for d_index in self.relevance[q_index]:
                    print(q_index, d_index)
                    f.write(str(q_index) + ' ' + str(d_index) + '\n')


def main():
    inv = InvIndex('title')
    inv.search()
    inv.print()


main()
