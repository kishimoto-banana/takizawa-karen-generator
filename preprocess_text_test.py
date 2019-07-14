import pickle
import re
import io
import collections
import mojimoji
import MeCab

FILE_PATH = 'data/karen_insta_test.txt'
WORD_TO_ID_DICT_PATH = 'data/corpus.pkl'
CORPUS_PATH = 'data/corpus_test.pkl'

DELIMITER = '+++$+++'


class Tokenizer:
    def __init__(self,
                 sentences,
                 eos='<eos>',
                 unk='<unk>',
                 digit='N',
                 dict_path=None):

        if dict_path is not None:
            self.tagger = MeCab.Tagger(f'-d {dict_path}')
        else:
            self.tagger = MeCab.Tagger()

        self.sentences = sentences
        self.eos = eos
        self.unk = unk
        self.digit = digit
        self.tokenized_sentences = None
        self.vocab = None
        self.regex = re.compile(r'\d')

    def tokenize(self):

        tokenized_sentences = []
        for sentence in self.sentences:
            for chunk in self.tagger.parse(sentence).splitlines()[:-1]:

                (surface, feature) = chunk.split('\t')
                if self.regex.search(surface):
                    tokenized_sentences.append(self.digit)
                else:
                    word = self.__word_normiraze(surface)
                    tokenized_sentences.append(word)
            tokenized_sentences.append(self.eos)

        self.tokenized_sentences = tokenized_sentences
        return tokenized_sentences


with io.open(FILE_PATH, encoding='utf-8') as f:
    text = f.read().lower()
sentences = text.split(DELIMITER)
# 先頭と末尾の改行を削除（最初の行だけ末尾のみ）
sentences = [
    sentence[1:-1] if idx != 0 else sentence[:-1]
    for idx, sentence in enumerate(sentences)
]

tokenizer = Tokenizer(
    sentences, dict_path='/usr/local/lib/mecab/dic/mecab-ipadic-neologd/')
tokenized_sentences = tokenizer.tokenize()

with open(WORD_TO_ID_DICT_PATH, 'rb') as f:
    _ = pickle.load(f)
    word_to_id = pickle.load(f)
    id_to_word = pickle.load(f)

corpus = []
for word in tokenized_sentences:
    try:
        corpus.append(word_to_id[word])
    except KeyError:
        corpus.append(0)

with open(CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)
    pickle.dump(word_to_id, f)
    pickle.dump(id_to_word, f)
