# """Utilities for downloading data from SNLI, tokenizing, vocabularies."""
from __future__ import print_function
import os
import re
import sys

from tensorflow.models.rnn.translate.data_utils import data_to_token_ids
from tensorflow.python.platform import gfile
from data_utils.preprocess_data import create_vocabulary, basic_tokenizer, EOS_ID

SOURCE_FILE = "nlci_%s_source.txt"
TARGET_FILE = "nlci_%s_target.txt"
SOURCE_VOCAB_FILE = "source_vocab%d.en"
TARGET_VOCAB_FILE = "target_vocab%d.en"
SOURCE_ID_FILE = "%s_source_ids%d.en"
TARGET_ID_FILE = "%s_target_ids%d.en"

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

_DIGIT_RE = re.compile(r"\d")

# bucket range with fixed decoder output:
_buckets = [(10, 3), (15, 3), (25, 3), (50, 3)]


#
#
def prepare_snli_data(parent_dir, source_vocab_size, target_vocab_size, train_source, train_target,
                      dev_source, dev_target, reuse_files=True):
    print("Preparing SNLI data.")
    # train_source_path, train_target_path, dev_source_path, dev_target_path = get_data_paths()

    # Create vocabularies of the appropriate sizes and get path to created files.
    print("Preparing vocabularies of the appropriate sizes.")
    source_vocab_path, target_vocab_path = get_source_target_vocab_path(parent_dir, source_vocab_size,
                                                                        target_vocab_size)
    create_vocabulary(source_vocab_path, train_source, source_vocab_size, is_norm_digits=True)
    create_vocabulary(target_vocab_path, train_target, target_vocab_size)

    print("Create token ids for the training and dev data.")
    # Create token ids for the training data and get path to created files.
    source_train_ids_path, target_train_ids_path = get_source_target_ids_path(parent_dir, source_vocab_size,
                                                                              target_vocab_size, "train")
    data_to_token_ids(train_source, source_train_ids_path, source_vocab_path)
    data_to_token_ids(train_target, target_train_ids_path, target_vocab_path)

    # Create token ids for the dev data and get path to created files.
    source_dev_ids_path, target_dev_ids_path = get_source_target_ids_path(parent_dir, source_vocab_size,
                                                                          target_vocab_size, "dev")
    data_to_token_ids(dev_source, source_dev_ids_path, source_vocab_path)
    data_to_token_ids(dev_target, target_dev_ids_path, target_vocab_path)

    return (source_train_ids_path, target_train_ids_path,
            source_dev_ids_path, target_dev_ids_path,
            source_vocab_path, target_vocab_path)


def read_actual_data_filter_repeat_premise(source_path, target_path, max_size):
    source_list = list()
    target_list = list()
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (max_size == 0 or counter < max_size):
                if counter == 0 or (source != source_list[-1]):
                    if counter != 0: print("new_s=%s old_s=%s sourcelen=%i counter=%i max=%i" % (
                        source, source_list[-1], len(source_list), counter, max_size))
                    source_list.append(source)
                    target_list.append(target)
                    counter += 1
                source, target = source_file.readline(), target_file.readline()
    return source_list, target_list


def read_actual_data_of_the_size(source_path, target_path, max_size,
                                 source_size=(5, 10),
                                 target_size=(10, 15)):
    source_list = list()
    target_list = list()
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (max_size == 0 or counter < max_size):
                source_tok = basic_tokenizer(source)
                target_tok = basic_tokenizer(target)
                if source_size[0] < len(source_tok) < source_size[1] and target_size[0] < len(target_tok) < target_size[
                    1]:
                    source_list.append(source)
                    target_list.append(target)
                    counter += 1
                source, target = source_file.readline(), target_file.readline()
    print(source_list)
    print(target_list)
    return source_list, target_list


def read_actual_data(source_path, target_path, max_size, bucket_id=None, tokenizer=basic_tokenizer):
    source_list = list()
    target_list = list()
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (max_size == 0 or counter < max_size):
                if counter == 0 or (source != source_list[-1]):
                    source_tok = basic_tokenizer(source)
                    target_tok = basic_tokenizer(target)
                    for cur_bucket_id, (buck_source_size, buck_target_size) in enumerate(_buckets):
                        # Will filter out of bucket sentences
                        if len(source_tok) < buck_source_size and len(target_tok) < buck_target_size:
                            if cur_bucket_id == bucket_id or bucket_id is None:
                                source_list.append(source)
                                target_list.append(target)
                                counter += 1
                            break
                            # fol loop is passed and no buckets in range were found
                            # print("Out of defined bucket sentence with len (%i, %i):" % (len(source_tok), len(target_tok)))
                source, target = source_file.readline(), target_file.readline()
    return source_list, target_list


def read_all_data_ids(source_ids_path_train, target_ids_path_train, source_ids_path_dev, target_ids_path_dev,
                      max_train_data_size):
    train_set = read_ids(source_ids_path_train, target_ids_path_train, max_train_data_size)
    dev_set = read_ids(source_ids_path_dev, target_ids_path_dev, None)  # Todo None
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Read all training and development data.")
    sys.stdout.flush()
    return train_set, dev_set, train_buckets_scale


def read_ids(source_ids_path, target_ids_path, max_size):
    print("Reading source files in" + source_ids_path + " and target files in " + target_ids_path)
    data_set = [[] for _ in _buckets]
    with gfile.GFile(source_ids_path, mode="r") as source_file:
        with gfile.GFile(target_ids_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 50000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    # print("len(target_ids)=%i , target_size=%i" % (len(target_ids), target_size))
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    sys.stdout.flush()
    return data_set


def get_data_paths():
    train_path = os.path.dirname(os.path.dirname(__file__)) + "/data/train/"
    dev_path = os.path.dirname(os.path.dirname(__file__)) + "/data/dev/"
    train_source, train_target = get_source_and_target_data("train", train_path)
    dev_source, dev_target = get_source_and_target_data("dev", dev_path)
    return train_source, train_target, dev_source, dev_target


def get_source_and_target_data(data_file_type, parent_path):
    source = os.path.join(parent_path, SOURCE_FILE % data_file_type)
    target = os.path.join(parent_path, TARGET_FILE % data_file_type)
    return source, target


def get_source_target_vocab_path(parent_path, source_vocab_size, target_vocab_size):
    source_vocab_path = os.path.join(parent_path, SOURCE_VOCAB_FILE % source_vocab_size)
    target_vocab_path = os.path.join(parent_path, TARGET_VOCAB_FILE % target_vocab_size)
    return source_vocab_path, target_vocab_path


def get_source_target_ids_path(parent_path, source_ids_size, target_ids_size, type):
    source_ids_path = os.path.join(parent_path, SOURCE_ID_FILE % (type, source_ids_size))
    target_ids_path = os.path.join(parent_path, TARGET_ID_FILE % (type, target_ids_size))
    return source_ids_path, target_ids_path
