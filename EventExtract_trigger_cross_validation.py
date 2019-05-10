# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhigangkan

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
from sklearn.metrics import confusion_matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklearn_metrics
import copy
import shutil

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "./glue_data/EE/the_last_data/single_data",
    "The input datadir.",
)
flags.DEFINE_string(
    "out_put_ann", "./1.ann",
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", "./multi_cased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "EE", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", "./output/result_dir_trigger_pos_windows_new/",
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", "./multi_cased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("add_pos", True, "Whether to add pos.")

flags.DEFINE_bool("use_windows", False, "Whether to use windows.")

flags.DEFINE_bool("predict_on_train_data", False, "Whether predict on train data.")

flags.DEFINE_bool("predict_on_dev_data", False, "Whether predict on dev data.")

flags.DEFINE_bool("predict_on_train_pool_data", False, "Whether predict on train_pool data.")

flags.DEFINE_bool("cross_validation", False, "Whether predict cross validation.")

flags.DEFINE_bool("predict_on_test_data", True, "Whether predict on test data.")

flags.DEFINE_bool("after_labeled", False, "Whether to use the labeled data to train the model.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", "./multi_cased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, pos, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.pos = pos


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, pos_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.pos_ids = pos_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, 'r+') as f:
            data_contect = f.read()
        lines = []
        words = []
        labels = []
        data_contect = data_contect.strip()
        all_sentences = data_contect.split('----------')[1:]
        for sentence in all_sentences:
            if len(sentence) == 0:
                continue
            sentence = sentence.strip()
            sentence_split = sentence.split('\n')
            sentence_id = sentence_split[0]
            # sentence_parments = sentence_split[1]
            # trigger分类部分只需要用到token——parments
            tokens_parments = sentence_split[1:]
            if len(tokens_parments) == 0:
                continue
            words = []
            labels = []
            poses = []
            for token_i_parments in tokens_parments:
                token_i_parments = token_i_parments.strip()
                token_i_parments_split = token_i_parments.split('\t')
                # print("token_i_parments_split:",token_i_parments_split)
                word = token_i_parments_split[0]
                pos = token_i_parments_split[2]
                label = token_i_parments_split[3]
                words.append(word)
                labels.append(label)
                poses.append(pos)
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            p = ' '.join([pos for pos in poses if len(pos) > 0])

            with open("text.txt", 'a+')as f1:
                f1.write('w:' + w + '\n')
                f1.write('l:' + l + '\n')
                f1.write('p:' + p + '\n')
                f1.write('\n')

            lines.append([l, w, p])
            # for line in f:
            #     contends = line.strip()
            #     word = line.strip().split('\t')[0]
            #     label = line.strip().split('\t')[-1]
            #     if contends.startswith("--tokens--"):
            #         words.append('')
            #         continue
            #     if len(contends) == 0 and words[-1] == '.':
            #         l = ' '.join([label for label in labels if len(label) > 0])
            #         w = ' '.join([word for word in words if len(word) > 0])
            #         lines.append([l, w])
            #         words = []
            #         labels = []
            #         continue
            #     words.append(word)
            #     labels.append(label)
        return lines


class EEProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir, test_data_source):
        if test_data_source == "train":
            return self._create_example(
                self._read_data(os.path.join(data_dir, "train.txt")), "test")
        elif test_data_source == "dev":
            return self._create_example(
                self._read_data(os.path.join(data_dir, "dev.txt")), "test")
        elif test_data_source == "train_pool":
            return self._create_example(
                self._read_data(os.path.join(data_dir, "train_pool.txt")), "test")
        else:
            return self._create_example(
                self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["Be-Born", "Die", "Marry", "Divorce", "Injure", "Transfer-Ownership", \
                "Transfer-Money", "Transport", "Start-Org", "End-Org", "Declare-Bankruptcy", \
                "Merge-Org", "Attack", "Demonstrate", "Meet", "Phone-Write", "Start-Position", \
                "End-Position", "Nominate", "Elect", "Arrest-Jail", "Release-Parole", \
                "Charge-Indict", "Trial-Hearing", "Sue", "Convict", "Sentence", "Fine", \
                "Execute", "Extradite", "Acquit", "Pardon", "Appeal", "None"]

    def get_pos(self):
        return ['EX', '$', '``', 'NNS', 'P', 'ETC', 'LS', 'WP', 'AS', 'AD', 'LC', 'CC', 'DT', '-RRB-', 'NNPS', 'VV', 'VBP', 'PDT', 'DER', 'OD', 'VBD', 'PRP', 'WRB', 'CD', 'URL', 'RBR', 'VBG', 'RB', 'DEV', 'JJR', 'VA', 'UH', 'WDT', 'WP$', ':', 'JJS', 'PN', 'NNP', 'RP', 'POS', 'VBZ', 'None', 'DEG', 'NN', 'VC', 'NT', 'MSP', 'LB', 'SYM', 'SP', 'JJ', 'CS', 'PRP$', '-LRB-', 'RBS', ',', 'VE', 'TO', 'VB', 'DEC', 'M', 'BA', 'VBN', 'SB', 'MD', '.', 'PU', 'FW', 'NR', 'IN']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            pos = tokenization.convert_to_unicode(line[2])
            examples.append(InputExample(guid=guid, text=text, pos=pos, label=label))
        return examples


def write_tokens(label_true_value, tokens, pos_ids_to_write, test_data_source, mode, output_dir_cv):
    if mode == "test":

        pos_list = ['EX', '$', '``', 'NNS', 'P', 'ETC', 'LS', 'WP', 'AS', 'AD', 'LC', 'CC', 'DT', '-RRB-', 'NNPS', 'VV', 'VBP', 'PDT', 'DER', 'OD', 'VBD', 'PRP', 'WRB', 'CD', 'URL', 'RBR', 'VBG', 'RB', 'DEV', 'JJR', 'VA', 'UH', 'WDT', 'WP$', ':', 'JJS', 'PN', 'NNP', 'RP', 'POS', 'VBZ', 'None', 'DEG', 'NN', 'VC', 'NT', 'MSP', 'LB', 'SYM', 'SP', 'JJ', 'CS', 'PRP$', '-LRB-', 'RBS', ',', 'VE', 'TO', 'VB', 'DEC', 'M', 'BA', 'VBN', 'SB', 'MD', '.', 'PU', 'FW', 'NR', 'IN']
        # print("!!!!tokens_list:",tokens_list)
        # print("!!!!label_true_value_list:",label_true_value_list)
        # len_pos_ids_list = len(pos_ids_list)
        pos_new_list = []
        for pos_id in pos_ids_to_write:
            pos_new_list.append(pos_list[pos_id - 1])

        path = os.path.join(output_dir_cv, test_data_source + "_token_" + mode + ".txt")
        path_label_true = os.path.join(output_dir_cv, test_data_source + "_label_true_in_" + mode + ".txt")
        path_pos = os.path.join(output_dir_cv, test_data_source + "_pos_" + mode + ".txt")

        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()
        wf2 = open(path_label_true, 'a')
        for label_i in label_true_value:
            # if token != "**NULL**":
            wf2.write(label_i + '\n')
        wf2.close()

        wf5 = open(path_pos, 'a')
        wf5.write('-' * 10 + '\n')
        for i in range(len(pos_ids_to_write)):
            wf5.write(pos_new_list[i] + '\n')
        wf5.close()


def convert_single_example(ex_index, example, label_list, pos_list, max_seq_length, tokenizer, test_data_source, mode,
                           output_dir_cv):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    pos_map = {}
    for (j, pos) in enumerate(pos_list, 1):
        pos_map[pos] = j
    with open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    # with open('./output/pos2id.pkl', 'wb') as w1:
    #     pickle.dump(pos_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    poslist = example.pos.split(' ')
    tokens = []
    labels = []
    poses = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        pos_1 = poslist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
                poses.append(pos_1)
            else:
                labels.append(label_1)
                poses.append(pos_1)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
        poses = poses[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    pos_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["None"])
    pos_ids.append(pos_map["None"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        pos_ids.append(pos_map[poses[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["None"])
    pos_ids.append(pos_map["None"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)

    pos_ids_to_write = copy.deepcopy(pos_ids)

    label_true_value = []
    for label_ids_i in label_ids:
        # print("label_ids_i:",label_ids_i)
        label_true_value.append(label_list[label_ids_i - 1])

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        pos_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("pos_ids: %s" % " ".join([str(x) for x in pos_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        pos_ids=pos_ids,
        # label_mask = label_mask
    )

    write_tokens(label_true_value, ntokens, pos_ids_to_write, test_data_source, mode, output_dir_cv)
    return feature


def filed_based_convert_examples_to_features(
        output_dir_cv, examples, label_list, pos_list, max_seq_length, tokenizer, output_file, test_data_source=None,
        mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, pos_list, max_seq_length, tokenizer,
                                         test_data_source, mode, output_dir_cv)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["pos_ids"] = create_int_feature(feature.pos_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "pos_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def get_tensor_after_move(tensor1, seq_length, type):
    """

    :param tensor1:输入张量
    :param type:控制移动方向和移动距离,
            -1 向右移动    1向左移动
    :return: 将输入张量平移后的张量，平移造成的孔雀位置补0
    """
    zero_tensor = tf.zeros_like(tensor1, dtype=tf.float32)
    zero_tensor_to_concat = tf.slice(zero_tensor, [0, 0, 0], [-1, abs(type), -1])

    if type > 0:
        tensor2_to_concat = tf.slice(tensor1, [0, type, 0], [-1, -1, -1])
        res_tensor_1 = tf.concat([tensor2_to_concat, zero_tensor_to_concat], 1)
    else:
        tensor2_to_concat = tf.slice(tensor1, [0, 0, 0], [-1, seq_length - abs(type), -1])
        res_tensor_1 = tf.concat([zero_tensor_to_concat, tensor2_to_concat], 1)

    return res_tensor_1


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, pos, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    # with tf.Session() as sess:
    #     sess.run()
    #     print("*****************!!!!!!!!!!!!!!!!!!!!output_layer.eval():",output_layer.eval())

    hidden_size = output_layer.shape[-1].value

    output_layer_slice = tf.slice(output_layer, [0, 0, 0], [1, 1, 1])
    print("!!!!!!!!!!!!!!!!!!!output_layer_slice:", output_layer_slice)

    if FLAGS.add_pos:
        pos_num = 70
        one_hot_pos = tf.one_hot(pos, depth=pos_num, dtype=tf.float32)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!one_hot_pos.get_shape():",one_hot_pos.get_shape())
        # one_hot_pos_new = tf.multiply(one_hot_pos , 1)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!one_hot_pos_new.get_shape():", one_hot_pos_new.get_shape())
        output_layer_new_1 = tf.concat([output_layer, one_hot_pos], -1)
    else:
        pos_num = 0
        output_layer_new_1 = output_layer

    if FLAGS.use_windows:
        # windows功能
        windows_size = 5
        seq_length = FLAGS.max_seq_length

        # output_layer_move_fu3  = get_tensor_after_move(output_layer,seq_length,-3)
        output_layer_move_fu2 = get_tensor_after_move(output_layer_new_1, seq_length, -2)
        output_layer_move_fu1 = get_tensor_after_move(output_layer_new_1, seq_length, -1)
        output_layer_move_1 = get_tensor_after_move(output_layer_new_1, seq_length, 1)
        output_layer_move_2 = get_tensor_after_move(output_layer_new_1, seq_length, 2)
        # output_layer_move_3  = get_tensor_after_move(output_layer,seq_length,3)

        # output_layer_new = tf.concat([output_layer_move_fu3,output_layer_move_fu2,output_layer_move_fu1,output_layer,output_layer_move_1,output_layer_move_2,output_layer_move_3],-1)
        output_layer_new_2 = tf.concat(
            [output_layer_move_fu2, output_layer_move_fu1, output_layer_new_1, output_layer_move_1,
             output_layer_move_2], -1)
        # output_layer_new = tf.concat([output_layer_move_fu1,output_layer,output_layer_move_1],-1)
    else:
        windows_size = 1
        output_layer_new_2 = output_layer_new_1

    output_layer_new = output_layer_new_2

    output_weight = tf.get_variable(
        "output_weights", [num_labels, (hidden_size + pos_num) * windows_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer_new = tf.nn.dropout(output_layer_new, keep_prob=0.9)
        output_layer_new = tf.reshape(output_layer_new, [-1, (hidden_size + pos_num) * windows_size])
        logits = tf.matmul(output_layer_new, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 35])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)

        return (loss, per_example_loss, logits, predict, probabilities)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, label_list=None):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        pos_ids = features["pos_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, pos_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_list, per_example_loss, label_ids, logits):
                # def metric_fn(label_ids, logits):

                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # with tf.variable_scope("loss"):
                # with sess.as_default():
                # # sess = tf.InteractiveSession()

                # print("*************************************************",label_ids)
                # print(type(label_ids))
                # label_ids_array = label_ids.eval()
                # predictions_array = predictions.eval()
                # print("***************************************label_ids_array:",label_ids_array)
                #
                # cm = sklearn_metrics.confusion_matrix(label_ids_array, predictions_array)
                # # cm = tf_metrics._streaming_confusion_matrix(label_ids, predictions, 35)
                # plot_confusion_matrix(cm, label_list)
                # # test!
                # accuracy = sklearn_metrics.accuracy_score(label_ids_array,predictions_array)
                # precision = sklearn_metrics.precision_score(label_ids_array, predictions_array,average="macro")
                # recall = sklearn_metrics.recall_score(label_ids_array, predictions_array,average="macro")
                # f = sklearn_metrics.f1_score(label_ids_array, predictions_array,average="macro")

                # accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions)
                # precision = tf_metrics.precision(label_ids, predictions, 35, average="macro")
                # recall = tf_metrics.recall(label_ids, predictions, 35, average="macro")
                # f = tf_metrics.f1(label_ids, predictions, 35, average="macro")

                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                # cm = tf_metrics._streaming_confusion_matrix(label_ids, predictions, 35)
                # plot_confusion_matrix(cm,label_list)
                # test!
                accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions)
                precision = tf_metrics.precision(label_ids, predictions, 35, average="weighted")
                recall = tf_metrics.recall(label_ids, predictions, 35, average="weighted")
                f = tf_metrics.f1(label_ids, predictions, 35, average="weighted")

                return {
                    "accuracy": accuracy,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [label_list, per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions={"probabilities": probabilities, "predicts": predicts}, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # with tf.Session() as sess:
    #     cm = cm.eval(session=sess)
    # print("cm******************:",cm)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
"""

def read_data(output_file):
    """Reads a BIO data."""
    with open(FLAGS.data_dir+"/test.txt", 'r') as f:
        data_contect = f.read()
    lines = []
    words = []
    labels = []
    data_contect = data_contect.strip()
    all_sentences = data_contect.split('----------')[1:]
    for sentence in all_sentences:
        if len(sentence) == 0:
            continue
        sentence = sentence.strip()
        sentence_split = sentence.split('\n')
        sentence_id = sentence_split[0]
        # sentence_parments = sentence_split[1]
        # trigger分类部分只需要用到token——parments
        tokens_parments = sentence_split[1:]
        if len(tokens_parments) == 0:
            continue
        words = []
        labels = []
        poses = []
        positions = []
        for token_i_parments in tokens_parments:
            token_i_parments = token_i_parments.strip()
            token_i_parments_split = token_i_parments.split('\t')
            # print("token_i_parments_split:",token_i_parments_split)
            word = token_i_parments_split[0]
            pos = token_i_parments_split[2]
            label = token_i_parments_split[3]
            position = token_i_parments_split[4]
            words.append(word)
            labels.append(label)
            poses.append(pos)
            positions.append(position)
        l = ' '.join([label for label in labels if len(label) > 0])
        w = ' '.join([word for word in words if len(word) > 0])
        p = ' '.join([pos for pos in poses if len(pos) > 0])
        s = ' '.join([position for position in positions if len(position) > 0])
        lines.append([l, w, p, s])
        # for line in f:
        #     contends = line.strip()
        #     word = line.strip().split('\t')[0]
        #     label = line.strip().split('\t')[-1]
        #     if contends.startswith("--tokens--"):
        #         words.append('')
        #         continue
        #     if len(contends) == 0 and words[-1] == '.':
        #         l = ' '.join([label for label in labels if len(label) > 0])
        #         w = ' '.join([word for word in words if len(word) > 0])
        #         lines.append([l, w])
        #         words = []
        #         labels = []
        #         continue
        #     words.append(word)
        #     labels.append(label)
    return lines
def do_predict_by_data(test_data_source, processor, label_list, pos_list, tokenizer, estimator, output_dir_cv,
                       data_dir_cv):
    token_path = os.path.join(output_dir_cv, test_data_source + "_token_test.txt")
    label_true_path = os.path.join(output_dir_cv, test_data_source + "_label_true_in_test.txt")
    with open('./output/label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    if os.path.exists(label_true_path):
        os.remove(label_true_path)
    if os.path.exists(token_path):
        os.remove(token_path)
    predict_examples = processor.get_test_examples(data_dir_cv, test_data_source)

    if test_data_source == "train":
        predict_file = os.path.join(output_dir_cv, "train_predict.tf_record")
    if test_data_source == "dev":
        predict_file = os.path.join(output_dir_cv, "dev_predict.tf_record")
    if test_data_source == "test":
        predict_file = os.path.join(output_dir_cv, "test_predict.tf_record")
    if test_data_source == "train_pool":
        predict_file = os.path.join(output_dir_cv, "train_pool_predict.tf_record")

    # predict_file = os.path.join(FLAGS.output_dir, predict_tf_record_file)
    filed_based_convert_examples_to_features(output_dir_cv, predict_examples, label_list, pos_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, test_data_source=test_data_source, mode="test")

    tf.logging.info("***** Running prediction for data from " + test_data_source + "*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    if FLAGS.use_tpu:
        # Warning: According to tpu_estimator.py Prediction on TPU is an
        # experimental feature and hence not supported here
        raise ValueError("Prediction in TPU not supported")
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = os.path.join(output_dir_cv, test_data_source + "_label_test.txt")
    output_predict_probability_file = os.path.join(output_dir_cv, test_data_source + "_label_probability_test.txt")
    pro_list = []

    with open(output_predict_file, 'w') as writer:
        for prediction in result:
            output_line = "\n".join(id2label[id] for id in prediction['predicts'] if id != 0) + "\n"

            for i in range(len(prediction['predicts'])):
                if prediction['predicts'][i] != 0:
                    word = prediction['probabilities'][i]
                    word_p = []
                    for word_type in word:
                        # print(word_type)
                        word_p.append(word_type)

                    word_p = np.array(word_p)
                    max_p = max(word_p)
                    # if max_p==1:
                    #     continue

                    max_index = np.argmax(word_p)
                    p = np.delete(word_p, max_index)
                    sen_max = max(p)
                    diff = max_p - sen_max
                    pro_list.append([max_p, "\t", sen_max, "\t", diff])

            writer.write(output_line)
    with open(output_predict_probability_file, 'w') as w:
        for item in pro_list:
            for i in item:
                w.write(str(i))
            w.write("\n \n")

    # 混淆矩阵可视化
    label_list_for_cmv = ["Attack", "Transport", "Die", "Transfer-Money", "End-Position", "Elect", "Meet",
                          "Phone-Write", "Charge-Indict",
                          "Arrest-Jail", "Transfer-Ownership", "Start-Position", "Trial-Hearing", "Sue", "Be-Born",
                          "Convict",
                          "Sentence", "Injure", "Start-Org", "End-Org", "Nominate", "Divorce", "Marry", "Appeal",
                          "Release-Parole",
                          "Fine", "Demonstrate", "Declare-Bankruptcy", "None", "Merge-Org", "Execute", "Extradite",
                          "Acquit", "Pardon"]
    label_true = []
    label_predictions = []
    # label_path = "/mnt/BERT/BERT-master/bert/output/result_dir_trigger_pos_windows/"
    label_path = output_dir_cv
    with open(label_path + test_data_source + "_label_test.txt", 'r')as f2:
        label_contect = f2.read()
        label_contect = label_contect.strip()
        label_predictions_1 = label_contect.split('\n')
        # label_predictions.extend(label_predictions_1)
        if len(label_predictions_1[-1]) == 0:
            label_predictions_1.pop(-1)
    with open(label_path + test_data_source + "_label_true_in_test.txt", 'r')as f2:
        label_true_contect = f2.read()
        label_true_contect = label_true_contect.strip()
        label_true_1 = label_true_contect.split('\n')
        # label_true.extend(label_true_1)
        if len(label_true_1[-1]) == 0:
            label_true_1.pop(-1)
    with open(label_path + test_data_source + "_token_test.txt", 'r')as f2:
        token_test_contect = f2.read()
        token_test_contect = token_test_contect.strip()
        tokens_test = token_test_contect.split('\n')
        # label_predictions.extend(label_predictions_1)
        if len(tokens_test[-1]) == 0:
            tokens_test.pop(-1)

    print("len(label_true_1):", len(label_true_1))
    print("len(label_predictions_1):", len(label_predictions_1))

    len_label_true_1 = len(label_true_1)
    if os.path.exists(output_dir_cv + test_data_source + "_data_pre_error_message_without_concat.txt"):
        os.remove(output_dir_cv + test_data_source + "_data_pre_error_message_without_concat.txt")
    error_token_count = 1
    error_diff_p = []
    with open(output_dir_cv + test_data_source + "_data_pre_error_message_without_concat.txt", 'a+')as f3:
        f3.write("token\ttrue_label\tpre_label\n0\n")
        for token_i1 in range(len_label_true_1):
            # f3.write(str(len_label_true_1) + ':\n')
            error_message_1 = ''

            if label_true_1[token_i1] != label_predictions_1[token_i1]:
                error_message_1 = 'error!!'
                error_diff_p.append(pro_list[token_i1][4])
            f3.write(tokens_test[token_i1] + '\t' + label_true_1[token_i1] + '\t' + label_predictions_1[
                token_i1] + '\t' + error_message_1 + '\t' + str(pro_list[token_i1][4]) + '\n')

            if tokens_test[token_i1] == "[SEP]":
                f3.write(str(error_token_count) + '--' * 20 + '\n')

    with open(output_dir_cv + test_data_source + "_error_prob.txt", "w") as w:
        w.write("avrage of error predict probabilities :")
        w.write("\n")
        if len(error_diff_p) != 0:
            w.write(str(sum(error_diff_p) / len(error_diff_p)))
        w.write("\n")
        for i in error_diff_p:
            w.write(str(i))
            w.write("\n")
    # 不将trigger拼接回来，评估
    token_true_label_count_for_r = 0
    token_pre_label_count_for_r = 0
    token_equal_label_count_for_r = 0
    token_equal_label_count = 0

    len_label_true_3 = len(label_true_1)
    for token_i2 in range(len_label_true_3):
        if label_true_1[token_i2] != "None":
            token_true_label_count_for_r += 1
        if label_predictions_1[token_i2] != "None":
            token_pre_label_count_for_r += 1
        if label_true_1[token_i2] != "None" and label_predictions_1[token_i2] != "None":
            token_equal_label_count_for_r += 1
        if label_true_1[token_i2] != "None" and label_true_1[token_i2] == label_predictions_1[token_i2]:
            token_equal_label_count += 1

    if token_pre_label_count_for_r != 0:
        token_precision_for_r_yang = token_equal_label_count_for_r / token_pre_label_count_for_r
    else:
        token_precision_for_r_yang = 0
    if token_true_label_count_for_r != 0:
        token_recall_for_r_yang = token_equal_label_count_for_r / token_true_label_count_for_r
    else:
        token_recall_for_r_yang = 0
    if token_precision_for_r_yang + token_recall_for_r_yang != 0:
        token_f1_for_r_yang = 2 * token_precision_for_r_yang * token_recall_for_r_yang / (
                token_precision_for_r_yang + token_recall_for_r_yang)
    else:
        token_f1_for_r_yang = 0
    if token_pre_label_count_for_r !=0:
        token_precision_yang = token_equal_label_count / token_pre_label_count_for_r
    else:
        token_precision_yang = 0
    if token_true_label_count_for_r != 0 :
        token_recall_yang = token_equal_label_count / token_true_label_count_for_r
    else:
        token_recall_yang = 0
    if token_precision_yang + token_recall_yang != 0:
        token_f1_yang = 2 * token_precision_yang * token_recall_yang / (token_precision_yang + token_recall_yang)
    else:
        token_f1_yang = 0
    output_path0 = "eval_result.txt"
    with open(output_dir_cv + output_path0, 'a+')as f4:
        f4.write("data from -->  " + test_data_source + '\n')
        f4.write("^^^^^^Do not concat the triggers back^^^^^^^" + '\n')
        f4.write("token_true_label_count_for_r:\t" + str(token_true_label_count_for_r) + '\n')
        f4.write("token_pre_label_count_for_r:\t" + str(token_pre_label_count_for_r) + '\n')
        f4.write("token_equal_label_count_for_r:\t" + str(token_equal_label_count_for_r) + '\n')
        f4.write("token_equal_label_count:\t" + str(token_equal_label_count) + '\n')
        f4.write('\n')
        f4.write("token_precision_for_r_yang:\t" + str(token_precision_for_r_yang) + '\n')
        f4.write("token_recall_for_r_yang:\t" + str(token_recall_for_r_yang) + '\n')
        f4.write("token_f1_for_r_yang:\t" + str(token_f1_for_r_yang) + '\n')
        f4.write('\n')
        f4.write("token_precision_yang:\t" + str(token_precision_yang) + '\n')
        f4.write("token_recall_yang:\t" + str(token_recall_yang) + '\n')
        f4.write("token_f1_yang:\t" + str(token_f1_yang) + '\n')
        f4.write('\n')

    if os.path.exists(label_path + "fenxi_" + test_data_source + "_error_messages.txt"):
        os.remove(label_path + "fenxi_" + test_data_source + "_error_messages.txt")
    with open(label_path + "fenxi_" + test_data_source + "_error_messages.txt", 'a+')as f3:
        f3.write("error_senyence_id\t/\ttoken\ttrue_label\tpre_label\n")

    # 不将trigger拼接回来，输出信息
    sep_indexs = []
    len_tokens_test = len(tokens_test)
    for index_i in range(len_tokens_test):
        if tokens_test[index_i] == "[SEP]":
            sep_indexs.append(index_i)
    sep_indexs = [-1] + sep_indexs

    error_count = 0

    if os.path.exists(label_path + "fenxi_" + test_data_source + "_error_messages_all.txt"):
        os.remove(label_path + "fenxi_" + test_data_source + "_error_messages_all.txt")
    with open(label_path + "fenxi_" + test_data_source + "_error_messages_all.txt", 'a+')as f3:
        f3.write("error_senyence_id\t/\ttoken\ttrue_label\tpre_label\n")
        f3.write('*' * 20 + '\n\n')

    for index_j in range(1, len(sep_indexs)):
        # mark_for_label_equal = True
        # for token_id in range(sep_indexs[index_j - 1] + 1, sep_indexs[index_j] + 1):
        #     if label_predictions_2[token_id] != label_true_2[token_id]:
        #         mark_for_label_equal = False
        # if not mark_for_label_equal:
        with open(label_path + "fenxi_" + test_data_source + "_error_messages_all.txt", 'a')as f3:

            f3.write(str(index_j) + '\n\n')
            for token_id2 in range(sep_indexs[index_j - 1] + 1, sep_indexs[index_j] + 1):
                error_message = ""
                if token_id2 < len(label_predictions_1) :
                    if label_true_1[token_id2] != label_predictions_1[token_id2]:
                        error_message = "error!!"
                        error_count += 1
                f3.write(tokens_test[token_id2] + '\t' + label_true_1[token_id2] + '\t' + \
                         label_predictions_1[token_id2] + '\t' + error_message + '\n')

            f3.write('*' * 20 + '\n\n')

    # 将被分割了的trigger再拼接回来
    label_i1 = 0
    while label_i1 < len(label_true_1) - 1:
        if label_true_1[label_i1] == label_true_1[label_i1 + 1] and label_true_1[label_i1] != "None":
            label_true_1.pop(label_i1 + 1)
            tokens_test[label_i1] = tokens_test[label_i1] + tokens_test[label_i1 + 1]
            tokens_test.pop(label_i1 + 1)
            if label_predictions_1[label_i1] != label_predictions_1[label_i1 + 1]:
                label_predictions_1[label_i1] = "None"
                label_predictions_1.pop(label_i1 + 1)
            else:
                label_predictions_1.pop(label_i1 + 1)
        else:
            label_i1 += 1

    token_list =[]
    ann_list = []
    with open(FLAGS.data_dir+"/test.txt","r",encoding="utf-8") as f:
        data_contect = f.read()
        data_contect = data_contect.strip()
        all_sentences = data_contect.split('----------')[1:]
        for sentence in all_sentences:
            token_list.append(sentence.split("\n"))


    label_i1 = 0
    print("label_predictions_1", label_predictions_1)
    while label_i1 < len(label_predictions_1) - 1:
        if label_predictions_1[label_i1] == label_predictions_1[label_i1 + 1] and label_predictions_1[label_i1] != "None":
            label_predictions_1.pop(label_i1 + 1)
            tokens_test[label_i1] = tokens_test[label_i1] + tokens_test[label_i1 + 1]
            tokens_test.pop(label_i1 + 1)
        else:
            label_i1 += 1
    # 输出信息，与工程师对接
    sentence_count = 0
    if os.path.exists(label_path + test_data_source + "_output_messages.txt"):
        os.remove(label_path + test_data_source + "_output_messages.txt")
    with open(label_path + test_data_source + "_output_messages.txt", 'a+')as f3:
        f3.write("token\ttrigger_pre_label\n")
        f3.write('-' * 10 + '\n')
        f3.write('--' + str(sentence_count) + '--' + '\n')
    with open(label_path + test_data_source + "_output_messages.txt", 'a+')as f3:
        for token_i3 in range(len(label_predictions_1)):
            if label_predictions_1[token_i3] != "None":
                trigger = tokens_test[token_i3].replace("##", "")
                print("trigger", trigger)
                for item in token_list:
                    for line in item:
                        if len(line) < 5:
                            continue
                        token = line.split("\t")[0]
                        token_start = line.split("\t")[4]
                        if token_start != '':
                            token_end = int(token_start) + len(token)
                            if trigger == token:
                                ann_list.append([label_predictions_1[token_i3],token_start,str(token_end),token])
            f3.write(tokens_test[token_i3] + '\t' + label_predictions_1[token_i3] + '\n')
            if token_i3 + 1 < len(label_predictions_1) and tokens_test[token_i3 + 1] == "[CLS]":
                sentence_count += 1
                f3.write('-' * 10 + '\n')
                f3.write('--' + str(sentence_count) + '--' + '\n')

    print(ann_list)
    if os.path.exists(FLAGS.out_put_ann):
        os.remove(FLAGS.out_put_ann)
    with open(FLAGS.out_put_ann, "w", encoding="utf-8") as fw:
        for i in range(len(ann_list)):
            fw.write("T" + str(i) + "\t")

            fw.write(ann_list[i][0] + " " + ann_list[i][1] + " " + ann_list[i][2] + "\t" + ann_list[i][3])
            fw.write("\n")

    with open(label_path + test_data_source + "_output_messages.txt", 'a+')as f3:
        for token_i3 in range(len(label_predictions_1)):

            f3.write(tokens_test[token_i3] + '\t' + label_predictions_1[token_i3] + '\n')
            if token_i3 + 1 < len(label_predictions_1) and tokens_test[token_i3 + 1] == "[CLS]":
                sentence_count += 1
                f3.write('-' * 10 + '\n')
                f3.write('--' + str(sentence_count) + '--' + '\n')


    # 杨森师兄的评估方法
    true_label_count_for_r = 0
    pre_label_count_for_r = 0
    equal_label_count_for_r = 0
    equal_label_count = 0

    return [token_true_label_count_for_r, token_pre_label_count_for_r, token_equal_label_count_for_r,
            token_equal_label_count], [true_label_count_for_r, pre_label_count_for_r, equal_label_count_for_r,
                                       equal_label_count]

def remove_new_checkpoint(new_path):
    dir_list = os.listdir(new_path)
    for dir in dir_list:
        os.remove(new_path + "/" + dir)

def get_trained_checkpoint_and_delete_others(checkpoint_path,new_path):
    dir_list = os.listdir(checkpoint_path)
    checkpoint_dir_list = []
    for dir in dir_list:
        if dir[:11] == "model.ckpt-":
            checkpoint_dir_list.append(dir)
    checkpoint_num_list = []
    for checkpoint_dir in checkpoint_dir_list:
        ckpt_num = checkpoint_dir.split("ckpt-")[1]
        ckpt_num = ckpt_num.split(".")[0]
        ckpt_num = int(ckpt_num)
        if ckpt_num not in checkpoint_num_list:
            checkpoint_num_list.append(ckpt_num)
    ckpt_num_max = max(checkpoint_num_list)
    final_checkpoint = "model.ckpt-" + str(ckpt_num_max)
    for checkpoint_dir in dir_list:

        len_file_num = len(final_checkpoint)
        if checkpoint_dir[:len_file_num] == final_checkpoint:
            checkpoint_file = checkpoint_path + "/" + checkpoint_dir
            shutil.move(checkpoint_file , new_path)
        else:
            if checkpoint_dir != "eval":
                checkpoint_file = checkpoint_path + checkpoint_dir
                os.remove(checkpoint_file)

    final_checkpoint = new_path + "/" + final_checkpoint
    return final_checkpoint


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ee": EEProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    # line = read_data("/home/feng/brat-v1.3_Crunchy_Frog/data/ace_input/PMID-1590827.txt")
    # print(line)
    label_list = processor.get_labels()

    pos_list = processor.get_pos()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    checkpoint_path = FLAGS.output_dir
    new_path = "new_checkpoint"
    remove_new_checkpoint(new_path)
    if FLAGS.after_labeled:
        init_checkpoint_final = get_trained_checkpoint_and_delete_others(checkpoint_path,new_path)
    else:
        init_checkpoint_final = FLAGS.init_checkpoint

    if not FLAGS.cross_validation:
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None

        if FLAGS.do_train:
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list) + 1,
            init_checkpoint=init_checkpoint_final,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu,
            label_list=label_list)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

        if FLAGS.do_train:
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(
                FLAGS.output_dir, train_examples, label_list, pos_list, FLAGS.max_seq_length, tokenizer, train_file)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        if FLAGS.do_eval:
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(
                FLAGS.output_dir, eval_examples, label_list, pos_list, FLAGS.max_seq_length, tokenizer, eval_file)

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d", len(eval_examples))
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
            eval_steps = None
            if FLAGS.use_tpu:
                eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        if FLAGS.do_predict:
            if os.path.exists(FLAGS.output_dir + "predict_test_result.txt"):
                os.remove(FLAGS.output_dir + "predicrt_test_result.txt")
            # 在训练集上做测试、评估
            if FLAGS.predict_on_train_data:
                test_data_source = "train"
                # test_data_source = test_data_source.decode('gbk')
                count1_train, count2_train = do_predict_by_data(test_data_source, processor, label_list, pos_list,
                                                                tokenizer, estimator, FLAGS.output_dir, FLAGS.data_dir)

            # 在验证集上做测试、评估
            if FLAGS.predict_on_dev_data:
                test_data_source = "dev"
                count1_dev, count2_dev = do_predict_by_data(test_data_source, processor, label_list, pos_list,
                                                            tokenizer, estimator, FLAGS.output_dir, FLAGS.data_dir)

            if FLAGS.predict_on_train_pool_data:
                test_data_source = "train_pool"
                # test_data_source = test_data_source.decode('gbk')
                count1_train, count2_train = do_predict_by_data(test_data_source, processor, label_list, pos_list,
                                                                tokenizer, estimator, FLAGS.output_dir, FLAGS.data_dir)


            # 在测试集上做测试、评估
            test_data_source = "test"
            count1_test, count2_test = do_predict_by_data(test_data_source, processor, label_list, pos_list, tokenizer,
                                                          estimator, FLAGS.output_dir, FLAGS.data_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("out_put_ann")
    tf.app.run()


