import io
import sys
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
MAX_EPOCHS = 60
FILE_PATH = 'data/karen_insta.txt'
MODEL_CHECKPOINT_PATH = 'model/word-wise-karen_{epoch:02d}.h5'
MODEL_FILE_PATH = 'model/char-wise-karen.h5'
GENERATED_TEXT_PATH = 'generated_text/char-wise-karen_diversity_{}.txt'

DELIMITER = '+++$+++'
SEQ_MAX_LEN = 8
STEP = 1
GENERATE_CHARS = 300

diversities = [0.5, 1.0]


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, start_index=2374):

    for diversity in diversities:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index:start_index + SEQ_MAX_LEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(GENERATE_CHARS):
            x_pred = np.zeros((1, SEQ_MAX_LEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        generated = generated.replace('N:N', '9:00')
        generated = generated.replace('N/N', '8/10')
        generated = generated.replace('N', '5')
        print(generated)


# load dataset
with io.open(FILE_PATH, encoding='utf-8') as f:
    text = f.read().lower()
text = text.replace(DELIMITER, '\n\n')
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_chars = []
for i in range(0, len(text) - SEQ_MAX_LEN, STEP):
    sentences.append(text[i:i + SEQ_MAX_LEN])
    next_chars.append(text[i + SEQ_MAX_LEN])
print('nb sequences:', len(sentences))

model = tf.keras.models.load_model(MODEL_FILE_PATH)

for i in range(3):
    generate(model)
