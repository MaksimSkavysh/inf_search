import math
import sys
import numpy as np
from parse import parse_articles, parse_requests
from inv_index import get_inv_index
from eval_fuction import check_eval

INV_INDEX_FILE = './inv_index_save.txt'
STR_DIVIDER = '-$-'
DOCUMENT_DIVIDER = 'DOCUMENT_DIVIDER\n'


def get_ftd(token, document):
    return document.count(token) / len(document)


def get_related_documents_list(inv_index, q):
    d_list = []
    for token in q:
        if token not in inv_index:
            continue
        cur_documents = inv_index[token]
        for d in cur_documents:
            if d not in d_list:
                d_list.append(d)
    return d_list


def calculate_simple_idf(N, N_t):
    return math.log((N - N_t) / N_t)


def calculate_idf(N, N_t):
    return math.log(1 + (N - N_t + 0.5) / (N_t + 0.5))


def calculate_rsv(q, d, N, L, b, k1, inv_index, k2=0):
    L_d = len(d)
    rsv_sum = 0
    for token in q:
        if token not in inv_index:
            continue
        N_t = len(inv_index[token])
        if N_t == 0:
            continue
        ftd = get_ftd(token, d)
        if ftd == 0:
            continue
        idf = calculate_idf(N, N_t)
        tf_td = (ftd * (k1 + 1)) / (k1 * ((1 - b) + b * (L_d / L)) + ftd)
        tf_tq = ((k2 + 1) * ftd) / (k2 + ftd)
        if idf > 0:
            rsv_sum = rsv_sum + idf * tf_td * tf_tq
    return rsv_sum

#  default params:
# mean precision: 0.18222222222222234
# mean recall: 0.25804024457131114
# mean F-measure: 0.21360288616472492
# MAP@10: 0.16733339702136005
#
#  best params: k1=1.2 b=0.0
# mean precision: 0.22400000000000014
# mean recall: 0.3219516630145968
# mean F-measure: 0.2641888555373502
# MAP@10: 0.23864681769827273


class InvIndex:
    def __init__(self, use_abstracts=False, b=0.75, k1=1.2, k2=0):
        self.documents = None
        self.questions = None
        self.relevance = {}
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.use_abstracts = use_abstracts
        self.inv_index = None
        self.N = None
        self.L = None

    def load_documents(self, doc_file):
        self.documents = parse_articles(doc_file, self.use_abstracts)

    def load_questions(self, qry_file):
        self.questions = parse_requests(qry_file)

    def build_inv_index(self):
        self.inv_index, self.N, self.L = get_inv_index(self.documents)

    def print_inv_index(self):
        with open(INV_INDEX_FILE, 'w') as f:
            f.write(str(self.N) + '\n')
            f.write(str(self.L) + '\n')
            for token in self.inv_index:
                inv_str = ','.join([str(x) for x in self.inv_index[token]])
                f.write(token + STR_DIVIDER + inv_str + '\n')
            f.write(DOCUMENT_DIVIDER)
            for d_id in self.documents:
                d_str = ','.join(self.documents[d_id])
                f.write(str(d_id) + STR_DIVIDER + d_str + '\n')
            print('saved in ' + INV_INDEX_FILE + ' file')

    def load_inv_index(self):
        with open(INV_INDEX_FILE, 'r') as f:
            self.N = int(f.readline())
            self.L = float(f.readline())
            self.inv_index = {}
            for line in f:
                if line == DOCUMENT_DIVIDER:
                    break
                token, d_list = line.split(STR_DIVIDER)
                d_list = d_list.replace('\n', '')
                self.inv_index[token] = [int(x) for x in d_list.split(',')]
            self.documents = {}
            for line in f:
                d_index, d_tokens = line.split(STR_DIVIDER)
                d_tokens = d_tokens.replace('\n', '')
                self.documents[int(d_index)] = [x for x in d_tokens.split(',')]

    def calculate_rsv(self, q, d):
        return calculate_rsv(
            q=q,
            d=d,
            N=self.N,
            L=self.L,
            b=self.b,
            k1=self.k1,
            k2=self.k2,
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
        return [t[0] for t in sorted_rsv_list[:10]]

    def search(self):
        for question in self.questions:
            self.relevance[question['index']] = self.search_relevance(question)

    def print(self):
        with open('./data/answer', 'w') as f:
            for q_index in self.relevance:
                for d_index in self.relevance[q_index]:
                    f.write(str(q_index) + ' ' + str(d_index) + '\n')


def run(file_path, use_abstracts=False, b=0.75, k1=1.2, k2=0):
    print('\nParams: ',
          'using',
          'Annotations, ' if use_abstracts else 'Titles',
          '| b=' + str(b),
          '| k1=' + str(k1),
          '| k2=' + str(k2),
          )
    inv = InvIndex(use_abstracts=use_abstracts, b=b, k1=k1, k2=k2)
    inv.load_documents(file_path)
    inv.build_inv_index()
    inv.print_inv_index()

    inv.search()
    inv.print()
    check_eval()


def index_mode(file_path, use_abstracts=False, b=0.75, k1=1.2, k2=0):
    print('\nBuilding index using',
          'Annotations' if use_abstracts else 'Titles', )
    inv = InvIndex(use_abstracts=use_abstracts, b=b, k1=k1, k2=k2)
    inv.load_documents(file_path)
    inv.build_inv_index()
    inv.print_inv_index()


def search_mode(qry_path, use_abstracts=False, b=0.75, k1=1.2, k2=0):
    print('\nParams: ',
          'using',
          'Annotations' if use_abstracts else 'Titles',
          '| b=' + str(b),
          '| k1=' + str(k1),
          '| k2=' + str(k2),
          )
    inv = InvIndex(use_abstracts=use_abstracts, b=b, k1=k1, k2=k2)
    inv.load_questions(qry_path)
    inv.load_inv_index()
    inv.search()
    inv.print()
    check_eval()


def main():
    mode = None
    file_path = None
    use_abstracts = False
    try:
        mode = sys.argv[1]
        file_path = sys.argv[2]
        if len(sys.argv) > 3:
            use_abstracts = True if sys.argv[3] == 'abstract' else False
        print('Running', mode, 'mode')
    except Exception as e:
        print('Wrong arguments')
        print('\nTemplate:')
        print('<script> index <articles file path> <abstract (optional to use abstracts or titles)>')
        print('<script> search <queries file path>')
        print('\nExamples:')
        print('python3 ./src/main.py index ./data/cran.all.1400')
        print('python3 ./src/main.py index ./data/cran.all.1400 abstract')
        print('python3 ./src/main.py search ./data/cran.qry')
        exit(0)

    if mode == 'index':
        index_mode(use_abstracts=use_abstracts, b=0.75, k1=1.2, k2=0, file_path=file_path)

    if mode == 'search':
        search_mode(use_abstracts=use_abstracts, b=0.75, k1=1.2, k2=0, qry_path=file_path)
        search_mode(use_abstracts=use_abstracts, b=0.0, k1=1.2, k2=0, qry_path=file_path)

        # index_mode(use_abstracts=False, b=0.75, k1=1.2, k2=0)
        # run(use_abstracts=False, b=0.75, k1=1.2, k2=0)
        # run(use_abstracts=True, b=0.75, k1=1.2, k2=0)

        # print('\nBest params:')
        # run(use_abstracts=False, b=0.0, k1=1.2, k2=0)
        # run(use_abstracts=True, b=0.0, k1=1.2, k2=0)

        # for b in np.arange(0, 1.25, 0.25):
        #     for k1 in np.arange(1.2, 2.1, 0.1):
        #         print('\n params: ', 'k1=', k1, ' b=', b)
        #         inv = InvIndex(use_abstracts=False, b=b, k1=k1)
        #         inv.search()
        #         inv.print()
        #         check_eval()

        # for k2 in [1, 10, 100, 1000]:
        #     print('\n params: k1=1.2 b=0.0, k2=', k2)
        #     inv = InvIndex(use_abstracts=use_abstracts, b=0.0, k1=1.0, k2=k2)
        #     inv.search()
        #     inv.print()
        #     check_eval()

main()
