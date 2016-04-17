import math
import os

from tensorflow.python.platform import gfile

from tfrnn.hooks import TraceHook, Hook

EVAL_LOSS_SUMMARY_TAG = "eval_loss_bucket_%d"
EVAL_PERP_SUMMARY_TAG = "eval_perplexity_bucket_%d"
EVAL_GRAD_NORM_SUMMARY_TAG = "eval_grad_norm_bucket_%d"
BLUE_SCORE_SUMMARY_TAG = "bleu_score"


class AccuracyOnDataSetHook(TraceHook):
    def __init__(self, summary_writer, data, bucket_range, iteration_interval, print_gradient_norm=False):
        super().__init__(summary_writer)
        self.data = data
        self.iteration_interval = iteration_interval
        self.bucket_range = bucket_range
        self.print_gradient_norm = print_gradient_norm

    def __call__(self, sess, epoch, iteration, model, loss):
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            for bucket_id in range(self.bucket_range):
                if len(self.data[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    self.data, bucket_id)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                self.update_summary(sess, iteration, EVAL_LOSS_SUMMARY_TAG % bucket_id, eval_loss)
                self.update_summary(sess, iteration, EVAL_PERP_SUMMARY_TAG % bucket_id, eval_ppx)
                if self.print_gradient_norm:
                    gradient_norm, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, bucket_id, False)
                    self.update_summary(sess, iteration, EVAL_GRAD_NORM_SUMMARY_TAG % bucket_id, gradient_norm)
                print("  eval: bucket %d loss %.2f gradient_norm %.2f perplexity %.2f" % (
                    bucket_id, eval_loss, -100.0, eval_ppx))


class SaveModelPerIterHook(Hook):
    def __init__(self, path, iteration_interval):
        self.path = os.path.join(path, "infer.ckpt")
        self.iteration_interval = iteration_interval

    def __call__(self, sess, epoch, iteration, model, loss):
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            print("Saving model within " + self.path)
            model.saver.save(sess, self.path, global_step=model.global_step)


class GenerateModelSamplesHook(Hook):
    def __init__(self, infer_sentence_method, source_data, parent_path, iteration_interval, data_label="data_name",
                 expected_generated_data=None):
        """
        Args:
            infer_sentence_method: Method with args (sess, model, source_sentence) that infers source_sentence
        """
        self.infer_sentence_method = infer_sentence_method
        self.source_data = source_data
        self.data_label = data_label
        self.iteration_interval = iteration_interval
        self.generated_samples_dir = os.path.join(parent_path, str(data_label) + str(len(source_data)))
        self.prepare_dir(self.generated_samples_dir, self.source_data, expected_generated_data)

    def __call__(self, sess, epoch, iteration, model, loss):
        if not iteration == 0 and iteration % self.iteration_interval == 0:
            output_file_path = os.path.join(self.generated_samples_dir, 'gen_epoch%d_iter%d.txt' % (epoch, iteration))
            with gfile.GFile(output_file_path, mode="w") as generated_output_file:
                print("Generate %d samples from %s.." % (len(self.source_data), self.data_label))
                for source_sentence in self.source_data:
                    hypothesis = self.infer_sentence_method(sess, model, source_sentence)
                    generated_output_file.write(hypothesis + "\n")

    @staticmethod
    def prepare_dir(gen_dir, premise_data, expected_hypothesis_data, input_data_filename='original_premise.txt',
                    expected_output_filename='expected_hypothesis.txt'):
        """
        Initialize directory and create file of original premise input data 'original_premise.txt'
        and if added to the hook expected hypothesis generated from the model
        Args:
            gen_dir: Directory path where files with generated model samples will be created.
            premise_data: Some input data from which model samples will be generated.
            expected_hypothesis_data: If added, expected_hypothesis file will be generated too, for fast comparison
            input_data_filename: generated file names
            expected_output_filename: generated file name
        """
        if not gfile.IsDirectory(gen_dir):
            gfile.MakeDirs(gen_dir)
        original_premise_filepath = os.path.join(gen_dir, input_data_filename)
        with gfile.GFile(original_premise_filepath, mode="w") as real_sentences_file:
            for source_sentence in premise_data:
                real_sentences_file.write(source_sentence)
        expected_hypothesis_filepath = os.path.join(gen_dir, expected_output_filename)
        with gfile.GFile(expected_hypothesis_filepath, mode="w") as expected_hypothesis_filepath:
            for target_sentence in expected_hypothesis_data:
                expected_hypothesis_filepath.write(target_sentence)
