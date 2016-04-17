import csv  # put the import inside for use in IPython.parallel
import re

from tensorflow.python.platform import gfile

FIELDNAMES = ('polarity', 'id', 'date', 'query', 'author', 'text')

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def read_sentiment_csv(csv_file, fieldnames=FIELDNAMES, max_count=None,
                       n_partitions=1, partition_id=0):
    def file_opener(csv_file):
        try:
            open(csv_file, 'r', encoding="latin1").close()
            return open(csv_file, 'r', encoding="latin1")
        except TypeError:
            # Python 2 does not have encoding arg
            return open(csv_file, 'rb')

    texts = []
    targets = []
    with file_opener(csv_file) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames,
                                delimiter=',', quotechar='"')
        pos_count, neg_count, neutral_count = 0, 0, 0
        for i, d in enumerate(reader):
            if i % n_partitions != partition_id:
                # Skip entry if not in the requested partition
                continue

            if d['polarity'] == '4':
                if max_count and pos_count >= max_count / 2:
                    continue
                pos_count += 1
                texts.append(d['text'])
                targets.append("4")

            elif d['polarity'] == '0':
                if max_count and neg_count >= max_count / 2:
                    continue
                neg_count += 1
                texts.append(d['text'])
                targets.append("0")

            elif d['polarity'] == '2':
                if max_count and neutral_count >= max_count / 2:
                    continue
                neutral_count += 1
                texts.append(d['text'])
                targets.append("2")
    print("neg=%i pos=%i neu=%i" % (neg_count, pos_count, neutral_count))
    return texts, targets


_buckets = [(10, 3), (15, 3), (25, 3), (50, 3)]

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_DIGIT_RE = re.compile(r"\d")
_TWITTER_RE = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z|_]+[A-Za-z0-9|_]+)")
_URL_RE = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)")

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

_PUNC_DEL_SPLIT = re.compile(".,!?\"'")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    sentence = sentence.lower()
    sentence = re.sub(_URL_RE, '@url', sentence)  # illuminate url
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    return [re.sub(_TWITTER_RE, '@name', w) for w in words if w]


def create_vocabulary(vocab_path, text, max_size, is_norm_digits=False):
    if not gfile.Exists(vocab_path):
        print("Creating vocabulary %s from data" % vocab_path)
        vocab = {}
        counter = 0
        for raw_line in text:
            counter += 1
            if counter % 50000 == 0: print("  processing line %d" % counter)
            line = re.sub(_URL_RE, '@url', raw_line)  # illuminate url
            line = re.sub(_PUNC_DEL_SPLIT, '', line)  # illuminate some punctuation to decrease sentence length
            tokens = basic_tokenizer(line.lower())  # tokenize and lowercase
            for w in tokens:
                # normalize digits and replace twitter @names
                if is_norm_digits:
                    word_normalize_digits = re.sub(_DIGIT_RE, "0", w)
                else:
                    word_normalize_digits = w
                    # print(w)
                word = re.sub(_TWITTER_RE, '@name', word_normalize_digits)
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_size and max_size != 0:
            vocab_list = vocab_list[:max_size]
        with gfile.GFile(vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                # print(w)
                vocab_file.write(w + "\n")
        return len(vocab_list)


def get_text_ids_from_file(ids_path):
    ids = []
    with gfile.GFile(ids_path, mode="r") as f:
        for line in f:
            ids.append(line)
    return ids


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)
