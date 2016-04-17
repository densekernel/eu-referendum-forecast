import math
import sys

import numpy as np
import tensorflow as tf
import word2vec
from nltk.stem.lancaster import LancasterStemmer
from numpy.ma import array
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import gfile

from data_utils.data_utils import get_source_target_vocab_path

SOURCE_EMBEDDING_KEY = "embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding"
TARGET_EMBEDDING_KEY = "embedding_attention_seq2seq/embedding_attention_decoder/embedding"


def get_pretrained_embeddings(session, word2vec_model, input_size, dict_dir, source_vocab_size, target_vocab_size,
                              source_vectors, target_vectors):
    source_vocab_path, target_vocab_path = get_source_target_vocab_path(dict_dir, source_vocab_size, target_vocab_size)
    print("source:")
    source_embed = get_pretrained_vector(session, word2vec_model,
                                         source_vocab_path, source_vocab_size, source_vectors)
    print("target:")
    target_embed = get_pretrained_vector(session, word2vec_model,
                                         target_vocab_path, target_vocab_size, target_vectors)
    return source_embed, target_embed


def get_pretrained_vector(session, word2vec_model, vocab_path, vocab_size, vectors):
    print(vectors)
    with gfile.GFile(vocab_path, mode="r") as vocab_file:
        st = LancasterStemmer()
        counter = 0
        counter_w2v = 0.0
        while counter < vocab_size:
            vocab_w = vocab_file.readline().replace("\n", "")

            # vocab_w = st.stem(vocab_w)
            # for each word in vocabulary check if w2v vector exist and inject.
            # otherwise dont change value initialise randomly.
            if word2vec_model and vocab_w and word2vec_model.__contains__(vocab_w) and counter > 3:
                w2w_word_vector = word2vec_model.get_vector(vocab_w)
                print("word:%s c:%i w2v size %i" % (vocab_w, counter, w2w_word_vector.size))
                vectors[counter] = w2w_word_vector
                counter_w2v += 1
            else:
                vocab_w_st = st.stem(vocab_w)
                if word2vec_model and vocab_w_st and word2vec_model.__contains__(vocab_w_st):
                    w2w_word_vector = word2vec_model.get_vector(vocab_w_st)
                    print("st_word:%s c:%i w2v size %i" % (vocab_w_st, counter, w2w_word_vector.size))
                    vectors[counter] = w2w_word_vector
                    counter_w2v += 1
                else:
                    if not vocab_w:
                        print("no more words.")
                        break

            counter += 1
        print("injected %f per cent" % (100 * counter_w2v / counter))
        print(vectors)
    return vectors


def inject_pretrained_word2vec(session, word2vec_model, input_size, dict_dir, source_vocab_size, target_vocab_size):
    source_vocab_path, target_vocab_path = get_source_target_vocab_path(dict_dir, source_vocab_size, target_vocab_size)
    session.run(tf.initialize_all_variables())
    assign_w2v_pretrained_vectors(session, word2vec_model, SOURCE_EMBEDDING_KEY, source_vocab_path, source_vocab_size,
                                  33)
    assign_w2v_pretrained_vectors(session, word2vec_model, TARGET_EMBEDDING_KEY, target_vocab_path, target_vocab_size,
                                  19)


def assign_w2v_pretrained_vectors(session, word2vec_model, embedding_key, vocab_path, vocab_size, id_to_check):
    embedding_variable = [v for v in tf.trainable_variables() if embedding_key in v.name]
    if len(embedding_variable) != 1:
        print("Word vector variable not found or too many. key: " + embedding_key)
        print("Existing embedding trainable variables:")
        print([v.name for v in tf.trainable_variables() if "embedding" in v.name])
        sys.exit(1)

    embedding_variable = embedding_variable[0]
    vectors = embedding_variable.eval()

    with gfile.GFile(vocab_path, mode="r") as vocab_file:
        counter = 0
        while counter < vocab_size:
            vocab_w = vocab_file.readline().replace("\n", "")
            # for each word in vocabulary check if w2v vector exist and inject.
            # otherwise dont change value initialise randomly.
            if vocab_w and word2vec_model.__contains__(vocab_w):
                w2w_word_vector = word2vec_model.get_vector(vocab_w)
                vectors[counter] = w2w_word_vector
            if counter == id_to_check:
                print(vectors[counter])
            counter += 1
    print("Reinitialising embeddings with pretrained")
    session.run(tf.assign(embedding_variable, vectors))
    # session.run([vectors_variable.initializer],
    #             {vectors_variable.initializer.inputs[1]: vectors})


def inject_word2vec_embeddings_old(session, word2vec_path, input_size, dict_dir, source_vocab_size, target_vocab_size):
    # (100000, 300)
    word2vec_model = word2vec.load(word2vec_path, encoding="latin-1")  # automatically detects format
    print("w2v model created!")

    source_vocab_path, target_vocab_path = get_source_target_vocab_path(dict_dir, source_vocab_size, target_vocab_size)
    w2v_vectors_source = get_w2v_pretrained_vectors(word2vec_model, source_vocab_path, source_vocab_size, input_size)
    w2v_vectors_target = get_w2v_pretrained_vectors(word2vec_model, target_vocab_path, target_vocab_size, input_size)

    print("pre-trained source shape " + str(w2v_vectors_source.shape))
    print(w2v_vectors_source)
    print(w2v_vectors_source.shape)  # (vocab_size, embedding_dim)
    with tf.variable_scope("embedding_attention_seq2seq"):
        with tf.variable_scope("RNN"):
            with tf.variable_scope("EmbeddingWrapper", reuse=True):
                # 1) getting Variable containing embeddings
                embedding = vs.get_variable("embedding", w2v_vectors_source.shape, trainable=False)

                # 2) using placeholder to assign embedding
                X = tf.placeholder(tf.float32, shape=w2v_vectors_source.shape)  # model.vectors.shape
                set_x = embedding.assign(X)
                session.run(tf.initialize_all_variables())
                session.run(set_x, feed_dict={X: w2v_vectors_source})

                v = session.run(embedding)
                print("After pre-trained")
                print(v)

    # embedding_attention_decoder  | embedding_attention_seq2seq/embedding_attention_decoder/embedding:0
    with tf.variable_scope("embedding_attention_seq2seq"):
        with tf.variable_scope("embedding_attention_decoder", reuse=True):
            decoder_embedding = vs.get_variable("embedding", w2v_vectors_target.shape, trainable=False)
            # 2) using placeholder to assign embedding
            X = tf.placeholder(tf.float32, shape=w2v_vectors_target.shape)  # model.vectors.shape
            set_x = decoder_embedding.assign(X)

            session.run(tf.initialize_all_variables())
            session.run(set_x, feed_dict={X: w2v_vectors_target})

            v = session.run(decoder_embedding)
            print("After pre-trained")
            print(v)


def get_w2v_pretrained_vectors(word2vec_model, vocab_path, vocab_size, input_size):
    print("Inject vectors for words from %s." % vocab_path)

    # iterating through words (word2id) and creating array of embeddings (vladsword2vec) either from w2v or default
    pretrained_embeddings_array = list()

    with gfile.GFile(vocab_path, mode="r") as vocab_file:
        vocab_w = vocab_file.readline().replace("\n", "")
        counter = 0
        # for each word in vocabulary check if w2v vector exist and inject. otherwise initialise randomly.
        while vocab_w or counter < vocab_size:
            counter += 1
            if counter % 100 == 0:
                print("reading line %i" % counter)
            if word2vec_model.__contains__(vocab_w):
                w2w_w = word2vec_model.get_vector(vocab_w)
                # w2w_w_projected = Projector(input_size, non_linearity=tf.nn.tanh)(w2w_w) # w2w_w[-1]
                pretrained_embeddings_array.append(w2w_w)
            else:
                # Uniform(-sqrt(3), sqrt(3)) has variance=1.
                a = -math.sqrt(3)
                b = math.sqrt(3)
                w2w_w = (b - a) * np.random.random_sample(input_size) + a  # random initialisation between a and b
                pretrained_embeddings_array.append(w2w_w)

            vocab_w = vocab_file.readline().replace("\n", "")
        return array(pretrained_embeddings_array)
