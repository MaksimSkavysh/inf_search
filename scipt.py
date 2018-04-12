import math
import sys
from collections import defaultdict
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import LancasterStemmer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


INV_INDEX_FILE = './inv_index_save.txt'
STR_DIVIDER = '-$-'
DOCUMENT_DIVIDER = 'DOCUMENT_DIVIDER\n'

NUMBER_OF_ABSTRACTS = 1400
INDEX_PREFIX = '.I'
TITLE_PREFIX = '.T'
AUTHORS_PREFIX = '.A'
INFO_PREFIX = '.B'
ABSTRACT_PREFIX = '.W'


def check_eval():
    groundtruth_file = './data/qrel_clean'
    answer_file = './inv_index_answer.txt'

    q2reld = {}
    for line in open(groundtruth_file):
        qid, did = [int(x) for x in line.split()]
        if qid not in q2reld.keys():
            q2reld[qid] = set()
        q2reld[qid].add(did)

    q2retrd = {}
    for line in open(answer_file):
        qid, did = [int(x) for x in line.split()]
        if qid not in q2retrd.keys():
            q2retrd[qid] = []
        q2retrd[qid].append(did)

    N = len(q2retrd.keys())
    precision = sum([len(q2reld[q].intersection(q2retrd[q])) * 1.0 / len(q2retrd[q]) for q in q2retrd.keys()]) / N
    recall = sum([len(q2reld[q].intersection(q2retrd[q])) * 1.0 / len(q2reld[q]) for q in q2retrd.keys()]) / N
    print("mean precision: {}\nmean recall: {}\nmean F-measure: {}" \
          .format(precision, recall, 2 * precision * recall / (precision + recall)))

    MAP = 0.0
    for q in q2retrd.keys():
        n_results = min(10, len(q2retrd[q]))
        avep = np.zeros(n_results)
        for i in range(n_results):
            avep[i:] += q2retrd[q][i] in q2reld[q]
            avep[i] *= (q2retrd[q][i] in q2reld[q]) / (i + 1.0)
        MAP += sum(avep) / min(n_results, len(q2reld[q]))
    print("MAP@10: {}".format(MAP / N))


english_stop_words = stopwords.words('english')
english_stop_words.append('.')
english_stop_words.append(',')
stop = set(english_stop_words)

lemmatizer = WordNetLemmatizer()
st = EnglishStemmer()
# st = PorterStemmer()
# st = LancasterStemmer()


def normalize(text):
    tokens = word_tokenize(text)
    tokens = [st.stem(t) for t in tokens if t not in stop]
    # tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop]
    return tokens


def get_inv_index(documents):
    inv_index = defaultdict(list)
    N = 0
    total_words = 0
    for index in documents:
        N = N + 1
        document = documents[index]
        total_words = total_words + len(document)
        for token in document:
            if not inv_index[token]:
                inv_index[token] = []
            if index not in inv_index[token]:
                inv_index[token].append(index)
    L = total_words / N
    return inv_index, N, L


def sum_lines(x, y):
    return x + y.replace('\n', ' ')


def parse_data(f, prefix, handler=sum_lines):
    s = f.readline()
    title = ''
    while prefix not in s and not s == '':
        title = handler(title, s)
        s = f.readline()
    return title, s


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


def parse_requests(qry_file):
    requests = []
    with open(qry_file) as f:
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
    idf_sum = 0
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
        with open('./inv_index_answer.txt', 'w') as f:
            for q_index in self.relevance:
                for d_index in self.relevance[q_index]:
                    f.write(str(q_index) + ' ' + str(d_index) + '\n')
            print('Top 10 RSV saved into ./inv_index_answer.txt')


def index_mode(file_path, use_abstracts=False, b=0.75, k1=1.2, k2=0):
    print('\nBuilding index using',
          'Annotations' if use_abstracts else 'Titles', )
    inv = InvIndex(use_abstracts=use_abstracts, b=b, k1=k1, k2=k2)
    inv.load_documents(file_path)
    inv.build_inv_index()
    inv.print_inv_index()


def search_mode(qry_path, use_abstracts=False, b=0.75, k1=1.2, k2=0):
    print('\nParams: ',
          'b=' + str(b),
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
        # search_mode(use_abstracts=use_abstracts, b=0.75, k1=1.2, k2=0, qry_path=file_path)
        search_mode(use_abstracts=use_abstracts, b=0.0, k1=1.2, k2=0, qry_path=file_path)

        # index_mode(use_abstracts=False, b=0.75, k1=1.2, k2=0)
        # run(use_abstracts=False, b=0.75, k1=1.2, k2=0)
        # run(use_abstracts=True, b=0.75, k1=1.2, k2=0)

        # print('\nBest params:')
        # run(use_abstracts=False, b=0.0, k1=1.2, k2=0)
        # run(use_abstracts=True, b=0.0, k1=1.2, k2=0)

        # for b in np.arange(0, 1.25, 0.25):
        #     for k1 in np.arange(1.2, 2.1, 0.1):
        #         search_mode(use_abstracts=use_abstracts, b=b, k1=k1, k2=0, qry_path=file_path)

        # for k2 in [1, 2, 3, 4, 5,  10, 50, 100, 200, 500, 700, 1000]:
        #     search_mode(use_abstracts=use_abstracts, b=0.75, k1=1.2, k2=k2, qry_path=file_path)

main()
