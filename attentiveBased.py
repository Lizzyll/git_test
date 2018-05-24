from preProcess import *
import logging
import tensorflow as tf
from tensorflow.contrib import rnn
import time, sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定GPU

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)


# 注意力求和，attention-sum过程
def sum_prob_of_word(word_ix, doc_idx, doc_attention_probs):
    word_ixs_in_sentence = tf.where(tf.equal(doc_idx, word_ix))
    begin_x = [0, 0]
    size_x = [tf.shape(word_ixs_in_sentence)[0], 1]
    word_ixs_in_sentence = tf.slice(word_ixs_in_sentence, begin_x, size_x)

    return tf.reduce_sum(tf.gather(doc_attention_probs, word_ixs_in_sentence))


def sum_probs_single_sentence(prev, cur):
    candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
    result = tf.scan(
        fn=lambda previous, x: sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
        elems=[candidate_indices_i],
        initializer=tf.constant(0., dtype="float32"))
    return result


def sum_probs_batch(scanList, doc_bt, doc_attention_probs_bt, dim=False):
    if type(dim) == bool:
        init = tf.constant([0], dtype="float32")  # 获取y_emb的第二个维度的长度[bt,ans_len]
    else:
        init = tf.zeros(shape=tf.shape(scanList)[1], dtype="float32")

    result = tf.scan(
        fn=sum_probs_single_sentence,
        elems=[scanList, doc_bt, doc_attention_probs_bt],
        initializer=init)

    return result


class Model_Baseline(object):
    def __init__(self):
        logging.info('let us dance!')

        # dev acc init
        self.dev_acc = 0.  # TODO: first dev accuracy displays here
        self.best_acc = 0

    def getData(self, ):
        # 获取训练数据
        if Debug:
            n_samples = 12000
        else:
            n_samples = 5000000  # 全部数据集为20000000
        self.train_examples = loadData(Train_path, max_examples=n_samples)
        self.dev_examples = loadData(Valid_path, 120000)
        plen = len(self.train_examples[0])
        qlen = len(self.train_examples[1])
        alen = len(self.train_examples[2])
        print(plen, qlen, alen)
        print('抽取训练数据条数为：%d' % plen)
        print('抽取验证数据条数为：%d' % len(self.dev_examples[0]))
        assert plen == qlen and qlen == alen, "获取的训练数据的p,q,a的长度不统一"

        # build word dictionary
        self.word_dict = build_dict(self.train_examples[0] + self.train_examples[1])

        # 获取embeding层初始状态
        # embedding_file = 'embding/zhwiki/zhwi_2017_03.sg_50d.word2vec'
        embedding_file = None
        self.embeddings = gen_embeddings(self.word_dict, embedding_size, embedding_file)
        self.embeddings = self.embeddings.astype('float32')

    def build_graph(self):
        # 构建graph
        logging.info('-' * 50)
        logging.info('Creating TF computation graph...')

        with tf.name_scope('input'):
            self.d_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input")
            self.q_input = tf.placeholder(dtype=tf.int32, shape=(None, None),
                                          name="q_input")  # [batch_size, max_seq_length_for_batch]
            self.y_emb = tf.placeholder(dtype=tf.int32, shape=(None, 1), name="answer")  # batch size vector

            self.d_input_set = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input_set")
            # d_input_len = tf.placeholder(dtype=tf.int32, shape=None, name="d_input_len")

            self.training = tf.placeholder(dtype=tf.bool)

            # actual_input_length=tf.shape(self.y_emb)[0] 类型为tensor,可以用在shape=的选项上

            tf.summary.histogram('d_input', self.d_input)
            tf.summary.histogram('q_input', self.q_input)
            tf.summary.histogram('y_emb', self.y_emb)
            tf.summary.histogram('d_input_set', self.d_input_set)

            # 载入词嵌入
            (vocab_size, embedding_size) = self.embeddings.shape
            self.word_embeddings = tf.get_variable("ch_glove", shape=(vocab_size, embedding_size),
                                                   initializer=tf.constant_initializer(self.embeddings))
            regularizer = tf.nn.l2_loss(self.word_embeddings)  # sum(embedding ** 2) / 2 为一个二范数的值
            print('word_embeddings shape:', self.word_embeddings.get_shape())
            tf.summary.histogram('word_embeddings', self.word_embeddings)
            # 载入d
            d_embed = tf.nn.embedding_lookup(self.word_embeddings,
                                             self.d_input)  # Apply embeddings: [batch, max passage length in batch, GloVe Dim]
            d_embed_dropout = tf.layers.dropout(d_embed, rate=dropout_rate,
                                                training=self.training)  # Apply Dropout to embedding layer
            # 载入q
            q_embed = tf.nn.embedding_lookup(self.word_embeddings, self.q_input)
            q_embed_dropout = tf.layers.dropout(q_embed, rate=dropout_rate, training=self.training)

        with tf.variable_scope('W_bilinear'):
            # 双线性权重 初始化
            W_bilinear = tf.Variable(tf.random_uniform((2 * hidden_size, 2 * hidden_size), minval=-0.01, maxval=0.01),
                                     name="W_Biliner")
            print('W_bilinear shape:', W_bilinear.get_shape())
            tf.summary.histogram('W_bilinear', W_bilinear)

        with tf.variable_scope('d_encoder',
                               initializer=orthogonal_initializer()):  # Encoding Step for Passage (d_ for document)

            d_cell_fw = rnn.LSTMCell(hidden_size)
            d_cell_bw = rnn.LSTMCell(hidden_size)

            d_outputs, _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, d_embed_dropout, dtype=tf.float32)

            d_output = tf.concat(d_outputs,
                                 axis=-1)  # [batch, len, 2h], len is the max passage length, and h is the hidden size
            print('d_output shape:', d_output.get_shape())
            tf.summary.histogram("d_outputs", d_output)

        with tf.variable_scope('q_encoder',initializer=orthogonal_initializer()):  # Encoding Step for Question
            q_cell_fw = rnn.LSTMCell(hidden_size)
            q_cell_bw = rnn.LSTMCell(hidden_size)
            q_outputs, q_laststates = tf.nn.bidirectional_dynamic_rnn(q_cell_fw, q_cell_bw, q_embed_dropout,
                                                                      dtype=tf.float32)

            q_output = tf.concat([q_laststates[0][-1], q_laststates[1][-1]], axis=-1)  # (batch, 2h)
            print('q_output shape:', q_output.get_shape())
            tf.summary.histogram("q_output", q_output)

        with tf.variable_scope('PQ_attention'):  # Bilinear Layer (Attention Step)
            # M computes the similarity between each passage word and the entire question encoding
            M = d_output * tf.expand_dims(tf.matmul(q_output, W_bilinear), axis=1)  # [batch, h] -> [batch, 1, h]
            # alpha represents the normalized weights representing how relevant the passage word is to the question
            alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2))  # [batch, d_len] 得到文中的每个词关于问题的权重表示

            # # this output contains the weighted combination of all contextual embeddings
            # bilinear_output = tf.reduce_sum(d_output * tf.expand_dims(alpha, axis=2), axis=1)  # [batch, h]
            print('alpha shape:', alpha.get_shape())
            tf.summary.histogram("alpha", alpha)

        with tf.variable_scope('merge'):  # 将相同词的attention值进行合并
            # 训练时取答案对应的概率值，并在loss计算时最大化此
            ansProbality = sum_probs_batch(self.y_emb, self.d_input, alpha)  # [batch, 1]对应答案词的attention值
            # 验证时，取出最大的概率值作为估计结果，从而计算acc
            # valid_ansProbality = sum_probs_batch(self.d_input_set, self.d_input, alpha, 1)  # [batch,d_input_len]

        with tf.variable_scope('train'):
            train_pred = tf.clip_by_value(ansProbality, 1e-7, 1.0 - 1e-7)  # 对概率进行限定，防止过小或过大的概率出现导致梯度消失或爆炸
            self.loss_op = tf.reduce_mean(-tf.log(train_pred)) + l2_reg * regularizer
            tf.summary.scalar("loss_op", self.loss_op)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer =tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
            self.train_op = optimizer.minimize(self.loss_op)
            logging.info('Done!')

        with tf.variable_scope('valid'):
            print('Vocab size is %d' % vocab_size)
            unstack_AD = zip(tf.unstack(alpha, batch_size), tf.unstack(self.d_input, batch_size))
            valid_ansProbality = tf.stack([tf.unsorted_segment_sum(a, d, vocab_size) for (a, d) in unstack_AD])

            # # 构造索引表，并通过检索表得到ans
            # ans_index = tf.expand_dims(tf.argmax(valid_ansProbality, 1), 1)
            # ans_index = tf.cast(ans_index, tf.int64)
            # # [batch,d_input_len]->argmax:[batch]->expand_dims:[batch,1]

            # sortList = tf.range(tf.shape(ans_index)[0], dtype=tf.int32)
            # # sortList = tf.range(32, dtype=tf.int32)
            # sortList = tf.to_int64(tf.reshape(sortList, [-1, 1]))
            #
            # ans_index = tf.concat([sortList, ans_index], 1)
            # ans = tf.cast(tf.gather_nd(self.d_input_set, ans_index), tf.int32)
            ans = tf.argmax(valid_ansProbality, 1)
            ans = tf.reshape(ans, [-1, 1])
            ans = tf.cast(ans, tf.int32)
            # 计算准确率
            self.acc = tf.reduce_mean(tf.cast(tf.equal(ans, self.y_emb), tf.float32))

            tf.summary.scalar("Dev_acc", self.acc)

        self.saver = tf.train.Saver()

    def test(self):
        logging.info('-' * 50)
        logging.info('Just Test...')

        Test_path = '../data/pd/pd.test'
        load_model_path = 'model_baseline/ch_attreader.ckpt'
        # load_model_path = 'model/ch_attreader.ckpt'
        test_examples = loadData(Test_path, 200000)
        test_x1, test_x2, test_y = vectorize(test_examples, self.word_dict, self.word_dict)  # 将原始的字符转为索引表示
        all_test = gen_examples(test_x1, test_x2, test_y, batch_size)
        with tf.Session() as sess:
            self.saver.restore(sess, load_model_path)

            correct = 0
            n_examples = 0
            for t_x1, t_x2, t_y in all_test:
                if len(t_x1) != 32:
                    print("batch is not 32 and pass")
                else:
                    t_input_len, t_input_set_deal = get_set(t_x1)
                    t_y = [[i] for i in t_y]
                    correct += sess.run(self.acc, feed_dict={self.d_input: t_x1, self.q_input: t_x2, self.y_emb: t_y,
                                                             self.training: False,
                                                             self.d_input_set: t_input_set_deal})
                    n_examples += 1

            test_acc = correct * 100 / n_examples
            logging.info('test_acc: %.2f %%' % test_acc)
        sys.exit(0)

    def trainModel(self):
        logging.info('-' * 50)
        logging.info('Initial Test...')
        dev_x1, dev_x2, dev_y = vectorize(self.dev_examples, self.word_dict, self.word_dict)  # 将原始的字符转为索引表示
        all_dev = gen_examples(dev_x1, dev_x2, dev_y, batch_size)

        logging.info('-' * 50)
        logging.info('Start training...')
        train_x1, train_x2, train_y = vectorize(self.train_examples, self.word_dict, self.word_dict)
        all_train = gen_examples(train_x1, train_x2, train_y, batch_size)

        init = tf.global_variables_initializer()
        n_updates = 0

        start_time = time.time()

        with tf.Session(config=config) as sess:
            merged = tf.summary.merge_all()

            train_writer = tf.summary.FileWriter(train_summary_path)
            valid_writer = tf.summary.FileWriter(valid_summary_path)

            sess.run(init)
            train_writer.add_graph(sess.graph)  # 添加网络结构到整个网络中

            earlyStop = False  # 控制程序提早结束
            earlyStop_count = 0
            no_improvement_N = 10

            with tf.device('/gpu:0'):
                for e in range(num_epoches):
                    np.random.shuffle(all_train)
                    for idx, (mb_x1, mb_x2, mb_y) in enumerate(all_train):
                        # logging.info(
                        #     'Batch Size = %d, # of Examples = %d, max_len = %d' % (
                        #         mb_x1.shape[0], len(mb_x1), mb_x1.shape[1]))
                        # print(mb_x1[:5])
                        # print('mb_x1 shape:', tf.shape(mb_x1))

                        # print(len(mb_x1),len(mb_x2),len(mb_y))
                        if len(mb_y) != 32:
                            print('batch size 不等于32 ,不进行训练,当前batch=%d' % len(mb_y))
                        else:
                            mb_y = [[i] for i in mb_y]
                            d_input_len_train, d_input_set_train = get_set(mb_x1)
                            # print(mb_y)
                            _, train_loss = sess.run([self.train_op, self.loss_op],
                                                     feed_dict={self.d_input: mb_x1, self.q_input: mb_x2,
                                                                self.y_emb: mb_y,
                                                                self.training: True})
                            logging.info('Epoch = %d, Iter = %d (max = %d), Loss = %.2f, Elapsed Time = %.2f (s)' %
                                         (e, idx, mb_x1.shape[1], train_loss, time.time() - start_time))


                            # train summary
                            summary = sess.run(merged,
                                               feed_dict={self.d_input: mb_x1, self.q_input: mb_x2, self.y_emb: mb_y,
                                                          self.training: True,
                                                          self.d_input_set: d_input_set_train})
                            train_writer.add_summary(summary, idx)

                        n_updates += 1
                        if Debug:  # 为debug时,尽快确认下验证数据集是否有问题
                            cond = True
                        else:
                            cond = n_updates % eval_iter == 0
                        if cond:
                            self.saver.save(sess, model_path, global_step=e)
                            correct = 0
                            n_examples = 0
                            for d_x1, d_x2, d_y in all_dev:
                                # print(len(d_input_set_deal))
                                # print(d_input_set_deal[:2])
                                if len(d_input_set_deal) != 32 or len(d_y) != 32:
                                    print('batch is not 32 and not excuted! d_input_len=%d,ans_len=%d' % (
                                        len(d_input_set_deal), len(d_y)))
                                else:
                                    d_input_len, d_input_set_deal = get_set(d_x1)
                                    d_y = [[i] for i in d_y]
                                    correct += sess.run(self.acc,
                                                        feed_dict={self.d_input: d_x1, self.q_input: d_x2,
                                                                   self.y_emb: d_y,
                                                                   self.training: False,
                                                                   self.d_input_set: d_input_set_deal})
                                    n_examples += 1

                                    valid_summary = sess.run(merged,
                                                             feed_dict={self.d_input: d_x1, self.q_input: d_x2,
                                                                        self.y_emb: d_y,
                                                                        self.d_input_set: d_input_set_deal,
                                                                        self.training: False})
                                    valid_writer.add_summary(valid_summary, idx)

                            dev_acc = correct * 100 / n_examples
                            logging.info('Dev Accuracy: %.2f %%' % dev_acc)
                            if dev_acc > self.best_acc:
                                earlyStop_count = 0  # best_acc更新则重置
                                self.best_acc = dev_acc
                                logging.info('Best Dev Accuracy: epoch = %d, n_updates (iter) = %d, acc = %.2f %%' %
                                             (e, n_updates, dev_acc))
                            else:
                                earlyStop_count += 1
                            if earlyStop_count >= no_improvement_N - 1:
                                earlyStop = True
                            if earlyStop:
                                break

                    if earlyStop:
                        break

            logging.info('-' * 50)
            logging.info('Training Finished...')
            logging.info("Model saved in file: %s" % self.saver.save(sess, model_path))
        train_writer.close()
        valid_writer.close()


Train_path = '../data/pd/pd.train'
Valid_path = '../data/pd/pd.valid'
embedding_size = 128
hidden_size = 128
dropout_rate = 0.2
batch_size = 32
num_epoches = 50
learning_rate = 0.1
eval_iter = 5000  # Evaluation on dev set after K updates
l2_reg = 0.0001  # 正则项,对word embding 数据大小进行限制
model_path = 'model/ch_attreader.ckpt'
optimizer = 'sgd'
train_summary_path = 'summary/train'
valid_summary_path = 'summary/valid'
Debug = False  # 控制时候调试代码还是运行代码

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    mkSummaryDir(train_summary_path)
    mkSummaryDir(valid_summary_path)

    bs = Model_Baseline()
    bs.getData()  # 准备数据
    bs.build_graph()

    test_Mode = False
    if test_Mode:
        bs.test()
    else:
        bs.trainModel()
