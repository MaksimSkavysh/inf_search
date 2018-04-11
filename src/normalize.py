from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import LancasterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

english_stop_words = stopwords.words('english')
english_stop_words.append('.')
english_stop_words.append(',')
stop = set(english_stop_words)

lemmatizer = WordNetLemmatizer()
# st = EnglishStemmer()
# st = PorterStemmer()
st = LancasterStemmer()


def normalize(text):
    # tokens = [st.stem(s) for s in word_tokenize(text)]
    tokens = [lemmatizer.lemmatize(s) for s in word_tokenize(text)]
    return [t for t in tokens if t not in stop]
