from __future__ import print_function

import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.translate.data_utils import sentence_to_token_ids
from tensorflow.python.platform import gfile

import my_wordvectors
import seq2seq_model
from data_utils import preprocess_data, data_utils
from data_utils.preprocess_data import basic_tokenizer, _buckets, initialize_vocabulary, EOS_ID
from my_hooks import SaveModelPerIterHook, AccuracyOnDataSetHook, GenerateModelSamplesHook
from tfrnn.hooks import SpeedHook, LossHook
from utils import embeddings_utils
# SNLI data related flags
# we can only create new files in tmp folder, as a requirement from Legion
from utils.embeddings_utils import SOURCE_EMBEDDING_KEY, TARGET_EMBEDDING_KEY

tf.app.flags.DEFINE_integer("source_vocab_size", 10000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 10000, "Target vocabulary size.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")

# Model related flags
# directory used for training process files e.g.: checkpoints, models
tf.app.flags.DEFINE_string("train_summary_dir", "data", "Training files directory.")
# directory for summary files, used by tensorboard
tf.app.flags.DEFINE_string("summary_dir", "data", "Summary files directory.")
# now all dict files and ids file will be create in tmp_dir folder
tf.app.flags.DEFINE_string("dict_dir", "data", "Dict files directory.")
tf.app.flags.DEFINE_string("gen_dir", "data/gen", "Generated entailments files directory.")
# Embeddings
tf.app.flags.DEFINE_string("word2vec_path", "data/word_embeddings/GoogleNews-vectors-negative300.bin",
                           "Pre-trained word2vec embeddings.")
tf.app.flags.DEFINE_boolean("has_word2vec_embed", False, "Creates model with word2vec word embedding injections.")

tf.app.flags.DEFINE_integer("size", 10, "Size of each model layer.")
tf.app.flags.DEFINE_integer("input_size", 300, "Dimension of the input tp the unit.")

tf.app.flags.DEFINE_boolean("reuse", False, "Reuse vocabulary and ids if they were already generated")
tf.app.flags.DEFINE_boolean("is_lstm", True, "Use lstm or gru")
tf.app.flags.DEFINE_boolean("is_adam", True, "Use adam or no opt")

tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("max_train_data_size", 100,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_steps", 30000,
                            "Limit on the number of training steps (0: no limit).")
tf.app.flags.DEFINE_integer("max_bleu_data_size", 100,
                            "Limit on the data set used to calculate bleu score (0: no limit).")
tf.app.flags.DEFINE_integer("bucket_id_only", None,
                            "Limit on the data set used to calculate bleu score (0: no limit).")

# Visualize loss 500
tf.app.flags.DEFINE_integer("steps_per_summary_update", 5,
                            "Training steps per training update of loss and perplexity.")
# Save model 2500
tf.app.flags.DEFINE_integer("steps_per_eval", 10,
                            "Training steps per accuracy loss and bleu evaluation.")
# Generate examples from the model 5000
tf.app.flags.DEFINE_integer("steps_per_generation", 500,
                            "Training steps per data generation from model.")
tf.app.flags.DEFINE_integer("steps_per_small_generation", 100,
                            "Training steps per data generation from model. (smaller generation size)")
# Generate examples from the model
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5000,
                            "Training steps per saving model checkpoint.")

tf.app.flags.DEFINE_integer("data_size", None, "Amount of data used from sentiment140.")

FLAGS = tf.app.flags.FLAGS


def train():

    data_utils_folder = os.path.join('data_utils', 'datasets')
    sentiment140_folder = os.path.join(data_utils_folder, 'sentiment140')
    training_csv_file = os.path.join(sentiment140_folder, 'training.1600000.processed.noemoticon.csv')
    texts, targets = preprocess_data.read_sentiment_csv(training_csv_file, max_count=FLAGS.data_size)

    half_size = int(len(texts)/2)
    train_part = int(0.8 * half_size)
    pos_train_ix = (0, train_part)
    neg_train_ix = (half_size, half_size + train_part)

    pos_dev_ind = (train_part, half_size)
    neg_dev_ind = (half_size + train_part, len(texts))

    train_source_data = texts[pos_train_ix[0]:pos_train_ix[1]] + texts[neg_train_ix[0]:neg_train_ix[1]]
    train_target_data = targets[pos_train_ix[0]:pos_train_ix[1]] + targets[neg_train_ix[0]:neg_train_ix[1]]
    print("Reading training data (length: %d, %d)." % (len(train_source_data), len(train_target_data)))
    print("e.g.: %s" % train_source_data[0])
    print("e.g.: %s" % train_target_data[0])

    print("e.g.: %s" % train_source_data[neg_train_ix[0]])
    print("e.g.: %s" % train_target_data[neg_train_ix[0]])

    dev_source_data = texts[pos_dev_ind[0]:pos_dev_ind[1]] + texts[neg_dev_ind[0]:neg_dev_ind[1]]
    dev_target_data = targets[pos_dev_ind[0]:pos_dev_ind[1]] + targets[neg_dev_ind[0]:neg_dev_ind[1]]
    print("Reading dev data (length: %d, %d)." % (len(dev_source_data), len(dev_target_data)))

    gen_source_data = texts[pos_dev_ind[0]:pos_dev_ind[0]+50] + texts[neg_dev_ind[0]:neg_dev_ind[0]+50]
    gen_target_data = targets[pos_dev_ind[0]:pos_dev_ind[0]+50] + targets[neg_dev_ind[0]:neg_dev_ind[0]+50]

    print("Prepare folders: train_summary_dir: %s summary_dir: %s" % (FLAGS.train_summary_dir, FLAGS.summary_dir))
    # if not exist, create files: ids for dev and train, vocabs
    source_train_ids_path, target_train_ids_path, source_dev_ids_path, target_dev_ids_path, _, _ = data_utils.prepare_snli_data(
        FLAGS.dict_dir,
        FLAGS.source_vocab_size,
        FLAGS.target_vocab_size,
        train_source_data, train_target_data,
        dev_source_data, dev_target_data, reuse_files=FLAGS.reuse)

    tf.Graph().as_default()
    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        word2vec_model = None
        if FLAGS.has_word2vec_embed:
            word2vec_model = my_wordvectors.load(FLAGS.word2vec_path, encoding="ISO-8859-1")
            print("w2v model created!")

        model = create_model(sess, False, is_word2vec=FLAGS.has_word2vec_embed,
                             source_vocab_size=FLAGS.source_vocab_size,
                             target_vocab_size=FLAGS.target_vocab_size,
                             input_size=FLAGS.input_size,
                             size=FLAGS.size, num_layers=FLAGS.num_layers,
                             max_gradient_norm=FLAGS.max_gradient_norm,
                             batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
                             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                             train_summary_dir=FLAGS.train_summary_dir,
                             word2vec_model=word2vec_model,
                             dict_dir=FLAGS.dict_dir, is_trainable_embed=True)

        train_ids_set, dev_ids_set, train_buckets_scale = data_utils.read_all_data_ids(source_train_ids_path,
                                                                                       target_train_ids_path,
                                                                                       source_dev_ids_path,
                                                                                       target_dev_ids_path,
                                                                                       FLAGS.max_train_data_size)

        print("train_buckets_scale=" + str(len(train_buckets_scale)))
        print("Prepare Hooks with train_dir:%s summary files dir:%s" % (FLAGS.train_summary_dir, FLAGS.summary_dir))
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)
        summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, graph_def=sess.graph_def)

        hooks = [
            SpeedHook(summary_writer, FLAGS.steps_per_summary_update, FLAGS.batch_size),
            LossHook(summary_writer, FLAGS.steps_per_summary_update),

            SaveModelPerIterHook(FLAGS.summary_dir, FLAGS.steps_per_checkpoint),
            AccuracyOnDataSetHook(summary_writer, train_ids_set, len(_buckets), FLAGS.steps_per_eval),
            AccuracyOnDataSetHook(summary_writer, dev_ids_set, len(_buckets), FLAGS.steps_per_eval),

            GenerateModelSamplesHook(sentiment_sentence, gen_source_data,
                                     os.path.join(FLAGS.summary_dir, FLAGS.gen_dir),
                                     FLAGS.steps_per_generation,
                                     "train_set", expected_generated_data=gen_target_data)
        ]

        start_training_loop(sess, model, train_buckets_scale, train_ids_set, len(train_source_data), hooks)


def start_training_loop(session, model, train_buckets_scale, train_ids_set, train_data_length, hooks):
    print("Start training loop.")
    current_step = 0
    # print(train_ids_set)
    while not FLAGS.max_train_steps or current_step < FLAGS.max_train_steps:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in range(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_ids_set, bucket_id)
        # print(decoder_inputs)
        _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)
        current_step += 1

        epoch = current_step * FLAGS.batch_size // train_data_length
        for hook in hooks:
            hook(session, epoch, current_step, model, step_loss)

        sys.stdout.flush()


def create_model(session, forward_only, is_word2vec=False,
                 source_vocab_size=None,
                 target_vocab_size=None,
                 input_size=None,
                 size=None, num_layers=None,
                 max_gradient_norm=None,
                 batch_size=None, learning_rate=None,
                 learning_rate_decay_factor=None,
                 train_summary_dir=None,
                 word2vec_model=None,
                 dict_dir=None, is_trainable_embed=False):
    model = seq2seq_model.Seq2SeqModel(
        source_vocab_size,
        target_vocab_size, _buckets, size,
        input_size,
        num_layers, max_gradient_norm, batch_size,
        learning_rate, learning_rate_decay_factor,
        None, None,
        use_adam=FLAGS.is_adam,
        use_lstm=FLAGS.is_lstm,
        forward_only=forward_only)

    ckpt = tf.train.get_checkpoint_state(train_summary_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.initialize_all_variables())
        if is_word2vec:
            source_embedding_variable= [v for v in tf.trainable_variables() if SOURCE_EMBEDDING_KEY in v.name]
            if len(source_embedding_variable) != 1:
                print("source_embedding_variable variable not found or too many.")
                sys.exit(1)
            source_embedding_variable = source_embedding_variable[0]

            target_embedding_variable= [v for v in tf.trainable_variables() if TARGET_EMBEDDING_KEY in v.name]
            if len(target_embedding_variable) != 1:
                print("target_embedding_variable variable not found or too many.")
                sys.exit(1)
            target_embedding_variable = target_embedding_variable[0]

            source_vectors = source_embedding_variable.eval()
            target_vectors = target_embedding_variable.eval()

            source_embedding, target_embedding = embeddings_utils.get_pretrained_embeddings(session, word2vec_model,
                                                                                            input_size,
                                                                                            dict_dir,
                                                                                            source_vocab_size,
                                                                                            target_vocab_size,
                                                                                            source_vectors,
                                                                                            target_vectors)
            session.run([source_embedding_variable.initializer],
                {source_embedding_variable.initializer.inputs[1]: source_embedding})
            session.run([target_embedding_variable.initializer],
                {target_embedding_variable.initializer.inputs[1]: target_embedding})

        print("Created model with fresh parameters.")
    return model


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True, is_word2vec=FLAGS.has_word2vec_embed,
                             source_vocab_size=FLAGS.source_vocab_size,
                             target_vocab_size=FLAGS.target_vocab_size,
                             input_size=FLAGS.input_size,
                             size=FLAGS.size, num_layers=FLAGS.num_layers,
                             max_gradient_norm=FLAGS.max_gradient_norm,
                             batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
                             learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                             train_summary_dir=FLAGS.train_summary_dir,
                             word2vec_model=None,
                             dict_dir=FLAGS.dict_dir, is_trainable_embed=True)

        data_utils_folder = os.path.join('data_utils', 'datasets')
        sentiment140_folder = os.path.join(data_utils_folder, 'sentiment140')
        training_csv_file = os.path.join(sentiment140_folder, 'testdata.manual.2009.06.14.csv')
        test_texts, test_targets = preprocess_data.read_sentiment_csv(training_csv_file, max_count=FLAGS.data_size)

        total = 0
        pos = 0.0
        for text, target in zip(test_texts, test_targets):
            if target != "2":
                sentiment = sentiment_sentence(sess, model, text)
                print("inf: %s test: %s" % (sentiment, target))
                if sentiment == target:
                    pos += 1
                else:
                    print(text)
                total += 1
        print("Result: %f" % (pos / total))

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline().replace('\n', '').replace('\r', '')
        while sentence:
            print("given sentence: %s" % sentence)
            sentiment = sentiment_sentence(sess, model, sentence)
            print(sentiment)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline().replace('\n', '').replace('\r', '')


def sentiment_sentence(sess, gen_model, sentence, is_beam_search=False):
    sentence = sentence.rstrip('\n')
    # print("input: %s" % sentence)
    source_vocab_path, target_vocab_path = data_utils.get_source_target_vocab_path(FLAGS.dict_dir,
                                                                                   FLAGS.source_vocab_size,
                                                                                   FLAGS.target_vocab_size)
    # Load vocabularies.
    source_vocab, _ = initialize_vocabulary(source_vocab_path)
    _, rev_target_vocab = initialize_vocabulary(target_vocab_path)

    gen_model.batch_size = 1  # We decode one sentence at a time.
    # Get token-ids for the input sentence.
    token_ids = sentence_to_token_ids(sentence, source_vocab, tokenizer=basic_tokenizer)
    # Which bucket does it belong to? todo new
    seq = [b for b in range(len(_buckets))
                     if _buckets[b][0] > len(token_ids)]
    if seq:
        bucket_id = min(seq)
    else:
        bucket_id = len(_buckets)-1

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = gen_model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.

    _, _, output_logits = gen_model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)

    # TODO implement beam search
    # outputs = decoder_util.run_beam_op(sess, rev_target_vocab, decoder_inputs, output_logits)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    # print(output_logits)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    print(outputs)
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]

    gen_model.batch_size = FLAGS.batch_size  # Put back to original batch_size.

    output_string = " ".join([rev_target_vocab[output] for output in outputs if not output >= len(rev_target_vocab)])
    # remove space before punctuation
    # print("output: %s" % output_string)
    output_string = re.sub(r'\s([?,.!"](?:\s|$))', r'\1', output_string)
    # print("output: %s" % output_string)
    return output_string


# function for potential model tests
def self_test():
    pass


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
