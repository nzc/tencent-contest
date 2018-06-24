"""
Created on Dec 10, 2017
@author: jachin,Nie

A tf implementation of NFM

Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics
    Xiangnan He,School of Computing,National University of Singapore,Singapore 117417,dcshex@nus.edu.sg
    Tat-Seng Chua,School of Computing,National University of Singapore,Singapore 117417,dcscts@nus.edu.sg
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import itertools
import random


class NFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes,
                 static_total_size_dict, dynamic_total_size_dict, dynamic_max_len_dict, exclusive_cols, extern_lr_size = 0, extern_lr_feature_size = 0,
                 embedding_size=8, dropout_fm=[1.0, 1.0], out = False, reduce = False,
                 deep_layers=[256, 128], dropout_deep=[1.0, 1.0, 1.0],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=1, batch_norm_decay=0.995,
                 verbose=True, random_seed=950104,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True, model_path=None):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        assert field_sizes[0] == len(static_total_size_dict) and field_sizes[1] == len(dynamic_total_size_dict)
        assert len(static_total_size_dict) > 0

        self.field_sizes = field_sizes
        self.total_field_size = field_sizes[0] + field_sizes[1]
        self.dynamic_total_size_dict = dynamic_total_size_dict
        self.static_total_size_dict = static_total_size_dict
        self.embedding_size = embedding_size
        self.dynamic_max_len_dict = dynamic_max_len_dict
        self.exclusive_cols = exclusive_cols
        self.extern_lr_size = extern_lr_size
        self.extern_lr_feature_size = extern_lr_feature_size
        self.dynamic_features = list(self.dynamic_total_size_dict.keys())
        self.static_features = list(self.static_total_size_dict.keys())
        self.total_features = list(self.static_total_size_dict.keys()) + list(self.dynamic_total_size_dict.keys())

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers

        self.out = out
        self.reduce = reduce

        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed + int(random.random() * 100)
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.model_path = model_path
        #self.train_result, self.valid_result = [], []

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)
            #static part input
            self.static_index_dict = {}
            for key in self.static_total_size_dict:
                self.static_index_dict[key] = tf.placeholder(tf.int32, shape=[None],
                                                 name=key+"_st_index")  # None
            #dynamic part input
            self.dynamic_index_dict = {}
            self.dynamic_lengths_dict = {}
            for key in self.dynamic_total_size_dict:
                self.dynamic_index_dict[key] = tf.placeholder(tf.int32, shape=[None, self.dynamic_max_len_dict[key]],
                                                 name=key+"_dy_index")  # None * max_len
                self.dynamic_lengths_dict[key] = tf.placeholder(tf.int32, shape=[None],
                                                 name=key+"_dy_length")  # None
            #others input
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # lr part
            self.static_lr_embs = [tf.gather(self.weights["static_lr_embeddings_dict"][key],
                                             self.static_index_dict[key]) for key in self.static_features] # static_feature_size * None * 1
            self.static_lr_embs = tf.concat(self.static_lr_embs, axis=1) # None * static_feature_size
            self.dynamic_lr_embs = [tf.reduce_sum(tf.gather(self.weights["dynamic_lr_embeddings_dict"][key],
                                              self.dynamic_index_dict[key]), axis=1) for key in self.dynamic_features] # dynamic_feature_size * None * 1
            self.dynamic_lr_embs = tf.concat(self.dynamic_lr_embs, axis=1) # None * dynamic_feature_size
            self.dynamic_lengths = tf.concat([tf.reshape(self.dynamic_lengths_dict[key],[-1,1]) for key in self.dynamic_features], axis=1)# None * dynamic_feature_size
            self.dynamic_lr_embs = tf.div(self.dynamic_lr_embs, tf.to_float(self.dynamic_lengths)) # None * dynamic_feature_size

            # ffm part
            embed_var_raw_dict = {}
            embed_var_dict = {}
            for key in self.static_features:
                embed_var_raw = tf.gather(self.weights["static_ffm_embeddings_dict"][key],
                                               self.static_index_dict[key]) # None * [k * F]
                embed_var_raw_dict[key] = tf.reshape(embed_var_raw, [-1, self.total_field_size, self.embedding_size])
            for key in self.dynamic_features:
                embed_var_raw = tf.gather(self.weights["dynamic_ffm_embeddings_dict"][key],
                                               self.dynamic_index_dict[key]) # None * max_len * [k * F]
                ffm_mask = tf.sequence_mask(self.dynamic_lengths_dict[key], maxlen=self.dynamic_max_len_dict[key]) # None * max_len
                ffm_mask = tf.expand_dims(ffm_mask, axis=-1) # None * max_len * 1
                ffm_mask = tf.concat([ffm_mask for i in range(self.embedding_size * self.total_field_size)],
                                     axis=-1) # None * max_len * [k * F]
                embed_var_raw = tf.multiply(embed_var_raw, tf.to_float(ffm_mask)) # None * max_len * [k * F]
                embed_var_raw = tf.reduce_sum(embed_var_raw, axis = 1) # None * [k*F]
                padding_lengths = tf.concat([tf.expand_dims(self.dynamic_lengths_dict[key], axis=-1)
                                             for i in range(self.embedding_size * self.total_field_size)], axis=-1) # None * [k*F]
                embed_var_raw = tf.div(embed_var_raw, tf.to_float(padding_lengths)) # None * [k*F]
                embed_var_raw_dict[key] = tf.reshape(embed_var_raw, [-1, self.total_field_size, self.embedding_size])

            for (i1, i2) in itertools.combinations(list(range(0, self.total_field_size)), 2):
                c1, c2 = self.total_features[i1], self.total_features[i2]
                if (c1, c2) in self.exclusive_cols:
                    continue
                embed_var_dict.setdefault(c1, {})[c2] = embed_var_raw_dict[c1][:, i2, :] # None * k
                embed_var_dict.setdefault(c2, {})[c1] = embed_var_raw_dict[c2][:, i1, :] # None * k

            x_mat = []
            y_mat = []
            input_size = 0
            for (c1, c2) in itertools.combinations(embed_var_dict.keys(), 2):
                if (c1, c2) in self.exclusive_cols:
                    continue
                input_size += 1
                x_mat.append(embed_var_dict[c1][c2]) #input_size * None * k
                y_mat.append(embed_var_dict[c2][c1]) #input_size * None * k
            x_mat = tf.transpose(x_mat, perm=[1, 0, 2]) # None * input_size * k
            y_mat = tf.transpose(y_mat, perm=[1, 0, 2]) # None * input_size * k

            if self.out:
                x_mat = tf.expand_dims(x_mat, 3)
                y_mat = tf.expand_dims(y_mat, 2)
                x = tf.matmul(x_mat, y_mat)
                x = tf.reshape(x, [-1, input_size, self.embedding_size * self.embedding_size])
            else:
                x = tf.multiply(x_mat, y_mat)

            if self.reduce:
                flat_vars = tf.reshape(tf.reduce_mean(x, axis=2), [-1, input_size])
            elif self.out:
                flat_vars = tf.reshape(x, [-1, input_size * self.embedding_size * self.embedding_size])
            else:
                flat_vars = tf.reshape(x, [-1, input_size * self.embedding_size])

            # ---------- Deep component ----------
            self.y_deep = flat_vars #
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- NFFM ----------
            #concat_input = tf.concat([self.static_lr_embs, self.dynamic_lr_embs, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            #self.out = tf.add(tf.reshape(tf.reduce_sum(self.out,axis=1),[-1,1]), self.weights['concat_bias'])
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.static_lr_embs,axis=1),[-1,1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.dynamic_lr_embs,axis=1),[-1,1]))

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            if not self.model_path:
                self.sess.run(init)
            else:
                self.load_model(self.model_path)


    def _init_session(self):
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()
        # lr part
        weights["static_lr_embeddings_dict"] = {}
        for key in self.static_total_size_dict:
            weights["static_lr_embeddings_dict"][key] = tf.Variable(
                tf.truncated_normal([self.static_total_size_dict[key], 1], 0.0, 0.0001),
                name=key + '_lr_embeddings')

        weights["dynamic_lr_embeddings_dict"] = {}
        for key in self.dynamic_total_size_dict:
            weights["dynamic_lr_embeddings_dict"][key] = tf.Variable(
            tf.truncated_normal([self.dynamic_total_size_dict[key], 1], 0.0, 0.0001),
            name=key+'_lr_embeddings')

        if self.extern_lr_size:
            weights["extern_lr_embeddings"] = tf.Variable(
            tf.truncated_normal([self.extern_lr_size, 1], 0.0, 0.0001),
            name="extern_lr_embeddings")

        # embeddings
        weights["static_ffm_embeddings_dict"] = {}
        for key in self.static_total_size_dict:
            weights["static_ffm_embeddings_dict"][key] = tf.Variable(
                tf.truncated_normal([self.static_total_size_dict[key],
                                     self.embedding_size * self.total_field_size], 0.0, 0.0001),
                name=key + '_ffm_embeddings')  # static_feature_size * [K * F]

        weights["dynamic_ffm_embeddings_dict"] = {}
        for key in self.dynamic_total_size_dict:
            weights["dynamic_ffm_embeddings_dict"][key] = tf.Variable(
                tf.truncated_normal([self.dynamic_total_size_dict[key],
                                     self.embedding_size * self.total_field_size], 0.0, 0.0001),
                name=key + '_ffm_embeddings') # dynamic_feature_size * [K * F]

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = 0
        features = self.total_features
        for (i1, i2) in itertools.combinations(list(range(0, len(features))), 2):
            c1, c2 = features[i1], features[i2]
            if (c1, c2) in self.exclusive_cols:
                continue
            input_size += 1
        if self.out:
            input_size *= self.embedding_size * self.embedding_size
        elif not self.reduce:
            input_size *= self.embedding_size
        #input_size = self.total_field_size * (self.total_field_size - 1) / 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            # weights["bias_%d" % i] = tf.Variable(
            #     np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
            #     dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        if self.extern_lr_size:
            input_size += self.extern_lr_feature_size
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, static_index_dict, dynamic_index_dict, dynamic_lengths_dict, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        batch_static_index_dict = {}
        batch_dynamic_index_dict = {}
        batch_dynamic_lengths_dict = {}
        for key in static_index_dict:
            batch_static_index_dict[key] = static_index_dict[key][start:end]
        for key in dynamic_index_dict:
            batch_dynamic_index_dict[key] = dynamic_index_dict[key][start:end]
        for key in dynamic_lengths_dict:
            batch_dynamic_lengths_dict[key] = dynamic_lengths_dict[key][start:end]
        return batch_static_index_dict, batch_dynamic_index_dict, batch_dynamic_lengths_dict,\
               [[y_] for y_ in y[start:end]]


    # shuffle four lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        for key in a:
            np.random.set_state(rng_state)
            np.random.shuffle(a[key])
        for key in b:
            np.random.set_state(rng_state)
            np.random.shuffle(b[key])
        for key in c:
            np.random.set_state(rng_state)
            np.random.shuffle(c[key])
        np.random.set_state(rng_state)
        np.random.shuffle(d)


    def fit_on_batch(self, static_index_dict, dynamic_index_dict, dynamic_lengths_dict, y):
        feed_dict = {self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        for key in self.static_features:
            feed_dict[self.static_index_dict[key]] = static_index_dict[key]
        for key in self.dynamic_features:
            feed_dict[self.dynamic_index_dict[key]] = dynamic_index_dict[key]
            feed_dict[self.dynamic_lengths_dict[key]] = dynamic_lengths_dict[key]
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, train_static_index_dict, train_dynamic_index_dict, train_dynamic_lengths_dict, train_y,
            valid_static_index_dict=None, valid_dynamic_index_dict=None, valid_dynamic_lengths_dict=None, valid_y=None,
            combine=False, show_eval = True, is_shuffle = True):
        """
        :param train_static_index:
        :param train_dynamic_index:
        :param train_dynamic_lengths:
        :param train_y:
        :param valid_static_index:
        :param valid_dynamic_index:
        :param valid_dynamic_lengths:
        :param valid_y:
        :return:
        """
        print "fit begin"
        has_valid = valid_static_index_dict is not None
        if has_valid and combine:
            for key in train_static_index_dict:
                train_static_index_dict[key] = np.concatenate([train_static_index_dict[key],
                                                               valid_static_index_dict[key]], axis=0)
            for key in train_dynamic_index_dict:
                train_dynamic_index_dict[key] = np.concatenate([train_dynamic_index_dict[key],
                                                               valid_dynamic_index_dict[key]], axis=0)
                train_dynamic_lengths_dict[key] = np.concatenate([train_dynamic_lengths_dict[key],
                                                                valid_dynamic_lengths_dict[key]], axis=0)
            train_y = np.concatenate([train_y, valid_y], axis=0)

        for epoch in range(self.epoch):
            total_loss = 0.0
            total_size = 0.0
            batch_begin_time = time()
            t1 = time()
            if is_shuffle:
                self.shuffle_in_unison_scary(train_static_index_dict, train_dynamic_index_dict,
                                            train_dynamic_lengths_dict, train_y)
            print "shuffle data cost %.1f" %(time()-t1)

            total_batch = int(len(train_y) / self.batch_size)
            for i in range(total_batch):
                offset = i * self.batch_size
                end = (i+1) * self.batch_size
                end = end if end < len(train_y) else len(train_y)
                static_index_batch_dict, dynamic_index_batch_dict, dynamic_lengths_batch_dict, y_batch\
                    = self.get_batch(train_static_index_dict, train_dynamic_index_dict, train_dynamic_lengths_dict,
                                     train_y, self.batch_size, i)
                batch_loss = self.fit_on_batch(static_index_batch_dict, dynamic_index_batch_dict, dynamic_lengths_batch_dict, y_batch)
                total_loss += batch_loss * (end - offset)
                total_size += end - offset
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss / total_size, time() - batch_begin_time))
                    total_loss = 0.0
                    total_size = 0.0
                    batch_begin_time = time()

            # evaluate training and validation datasets
            if not combine and show_eval:
                train_result = self.evaluate(train_static_index_dict, train_dynamic_index_dict,
                                             train_dynamic_lengths_dict, train_y)
            #self.train_result.append(train_result)
            if has_valid and not combine:
                valid_result = self.evaluate(valid_static_index_dict, valid_dynamic_index_dict,
                                             valid_dynamic_lengths_dict, valid_y)
            #    self.valid_result.append(valid_result)
            if self.verbose > 0 and not combine and show_eval:
                if has_valid and not combine:
                    print("[%d] train-result=%.6f, valid-result=%.6f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.6f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))

        print "fit end"


    def predict(self, static_index_dict, dynamic_index_dict, dynamic_lengths_dict, y = []):
        """
        :param static_index:
        :param dynamic_index:
        :param dynamic_lengths:
        :return:
        """
        print "predict begin"
        # dummy y
        if len(y) == 0:
            dummy_y = [1] * len(static_index_dict[self.static_features[0]])
        else:
            dummy_y = y
        batch_index = 0
        batch_size = 1024
        static_index_dict_batch, dynamic_index_dict_batch, dynamic_lengths_dict_batch, y_batch\
            = self.get_batch(static_index_dict, dynamic_index_dict, dynamic_lengths_dict, dummy_y, batch_size, batch_index)
        y_pred = None
        total_loss = 0.0
        total_size = 0.0
        while len(static_index_dict_batch[self.static_features[0]]) > 0:
            num_batch = len(y_batch)
            feed_dict = {
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            for key in self.static_features:
                feed_dict[self.static_index_dict[key]] = static_index_dict_batch[key]
            for key in self.dynamic_features:
                feed_dict[self.dynamic_index_dict[key]] = dynamic_index_dict_batch[key]
                feed_dict[self.dynamic_lengths_dict[key]] = dynamic_lengths_dict_batch[key]
            batch_out, batch_loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
            total_loss += batch_loss * num_batch
            total_size += num_batch
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            static_index_dict_batch, dynamic_index_dict_batch, dynamic_lengths_dict_batch, y_batch \
                = self.get_batch(static_index_dict, dynamic_index_dict, dynamic_lengths_dict, dummy_y, batch_size,
                                 batch_index)
        print "valid logloss is %.6f" % (total_loss / total_size)
        print "predict end"
        return y_pred


    def evaluate(self, static_index_dict, dynamic_index_dict, dynamic_lengths_dict, y):
        """
        :param static_index:
        :param dynamic_index:
        :param dynamic_lengths:
        :param y:
        :return:
        """
        print "evaluate begin"
        print "predicting ing"
        b_time = time()
        y_pred = self.predict(static_index_dict, dynamic_index_dict, dynamic_lengths_dict, y)
        print "predicting costs %.1f" %(time()- b_time)
        print "counting eval ing"
        b_time = time()
        res =  self.eval_metric(y, y_pred)
        print "counting eval cost %.1f" %(time()- b_time)
        print "evaluate end"
        return res

    def save_model(self, path, i):
        self.saver.save(self.sess, path, global_step=i)

    def load_model(self, path):
        model_file = tf.train.latest_checkpoint(path)
        print model_file,"model file"
        self.saver.restore(self.sess, path)