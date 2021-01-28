import pickle
import re
import io
import collections
import mojimoji
import MeCab

FILE_PATH = "data/karen_insta.txt"
CORPUS_PATH = "data/corpus.pkl"

DELIMITER = "+++$+++"


def preprocess(words):

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = [word_to_id[word] for word in words]

    return corpus, word_to_id, id_to_word


class Tokenizer:
    def __init__(self, sentences, eos="<eos>", unk="<unk>", digit="N", dict_path=None):

        if dict_path is not None:
            self.tagger = MeCab.Tagger(f"-d {dict_path}")
        else:
            self.tagger = MeCab.Tagger()

        self.sentences = sentences
        self.eos = eos
        self.unk = unk
        self.digit = digit
        self.tokenized_sentences = None
        self.vocab = None
        self.regex = re.compile(r"\d")

    def tokenize(self):

        tokenized_sentences = []
        for sentence in self.sentences:
            for chunk in self.tagger.parse(sentence).splitlines()[:-1]:

                (surface, feature) = chunk.split("\t")
                if self.regex.search(surface):
                    tokenized_sentences.append(self.digit)
                else:
                    word = self.__word_normiraze(surface)
                    tokenized_sentences.append(word)
            tokenized_sentences.append(self.eos)

        self.tokenized_sentences = tokenized_sentences
        return tokenized_sentences

    def normilize(self, freq=3):

        word_freqs = self.__count_word_freq()
        print(word_freqs[0])

    def __count_word_freq(self):
        word_freqs = collections.Counter()

        for word in self.tokenized_sentences:
            word_freqs[word] += 1

        return word_freqs

    def __word_normiraze(self, word):
        """単語の正規化"""
        word = word.lower()
        word = mojimoji.han_to_zen(word, ascii=False, digit=False)
        word = mojimoji.zen_to_han(word, kana=False)

        return word


with io.open(FILE_PATH, encoding="utf-8") as f:
    text = f.read().lower()
sentences = text.split(DELIMITER)
# 先頭と末尾の改行を削除（最初の行だけ末尾のみ）
sentences = [
    sentence[1:-1] if idx != 0 else sentence[:-1]
    for idx, sentence in enumerate(sentences)
]

tokenizer = Tokenizer(
    sentences, dict_path="/usr/local/lib/mecab/dic/mecab-ipadic-neologd/"
)
tokenized_sentences = tokenizer.tokenize()

corpus, word_to_id, id_to_word = preprocess(tokenized_sentences)

with open(CORPUS_PATH, "wb") as f:
    pickle.dump(corpus, f)
    pickle.dump(word_to_id, f)
    pickle.dump(id_to_word, f)
