
import logging
from collections import Counter
import numpy as np
import os,shutil
import re
import tensorflow as tf

vocab_size=0
# 每隔n行抛出
def readNline(path,n):
    f = open(path)
    line = f.readline().strip('\n')
    lineList = []
    i = 0
    while line:
        lineList.append(line)
        if i % n == 0 and i != 0:
            yield (lineList)
            lineList = []
        i += 1
        line = f.readline().strip('\n')
    # 未组成n行抛出时，将剩余的一起抛出
    if lineList:
        yield (lineList)


def getIndex(line):
    i = int(line.split('|||')[0].strip(' '))
    return i


def getQueryAndAnswer(line):
    splitLine = line.split('|||')
    assert len(splitLine) == 3, "输入的行格式不对，原始内容为：%s" % line
    q = splitLine[1].strip(' ')
    a = splitLine[2].strip(' ')
    return q, a

#读取三行
def loadData_3(path, max_examples=False):
    pList = []
    qList = []
    aList = []
    getNewPara = False  # 控制是否开启新的para
    pOne = ''  # 一个完整para放在一起
    n = 0
    for multiLine in readNline(path,3):
        # q+answer
        if len(multiLine) == 1:
            q, a = getQueryAndAnswer(multiLine[0])
            qList.append(q)
            aList.append(a)
        # 0:p, 1:q+a
        elif len(multiLine) == 2:
            ptmp = multiLine[0].split('|||')[1]
            pOne += ptmp

            q, a = getQueryAndAnswer(multiLine[1])
            qList.append(q)
            aList.append(a)
            # all p
        elif len(multiLine) == 4:
            ptmp = ''.join([line.split('|||')[1] for line in multiLine])
            pOne += ptmp
        # p 或者 存在 q+a
        elif len(multiLine) == 3:
            # 首先判断是否存在q+a
            index1 = getIndex(multiLine[0])
            index2 = getIndex(multiLine[1])
            index3 = getIndex(multiLine[2])

            if index3 > index2 and index2 > index1:
                # 皆为p
                if len(multiLine[2].split('|||')) == 2:
                    if index1 == 1:
                        # 不为空，排除第三句为q+a,重复append的情况，只有等下次结束时，才进行append
                        if pOne:
                            pList.append(pOne)
                            pOne = ''
                    ptmp = ''.join([line.split('|||')[1] for line in multiLine])
                    pOne += ptmp
                    # 第三句为q+a
                elif len(multiLine[2].split('|||')) == 3:
                    q, a = getQueryAndAnswer(multiLine[2])
                    qList.append(q)
                    aList.append(a)

                    ptmp = ''.join([line.split('|||')[1] for i, line in enumerate(multiLine) if i < 2])
                    pOne += ptmp
                    pList.append(pOne)
                    pOne = ''

                    # 第一句为q+a
            elif index2 < index1:
                q, a = getQueryAndAnswer(multiLine[0])
                qList.append(q)
                aList.append(a)
                pList.append(pOne)
                pOne = ''

                ptmp = ''.join([line.split('|||')[1] for i, line in enumerate(multiLine) if i < 2])
                pOne += ptmp


            # 第二句为q+a
            elif index3 < index2:
                ptmp = multiLine[0].split('|||')[1]
                pOne += ptmp
                pList.append(pOne)
                pOne = multiLine[2].split('|||')[1]  # 作为p的第一句

                q, a = getQueryAndAnswer(multiLine[1])
                qList.append(q)
                aList.append(a)
        if max_examples:
            # 读取多少行
            if int(n * 3) > max_examples:
                break
            n += 1
    return pList, qList, aList


#读取2行
def loadData(path, max_examples=False):
    pList = []
    qList = []
    aList = []
    getNewPara = False  # 控制是否开启新的para
    pOne = ''  # 一个完整para放在一起
    n = 0
    for multiLine in readNline(path,2):
        # q+answer
        if len(multiLine) == 1:
            q, a = getQueryAndAnswer(multiLine[0])
            q=Chinese_word_extraction(q)
            a=Chinese_word_extraction(a)
            qList.append(q)
            aList.append(a)
        #刚开始皆为一个段落里的内容
        elif len(multiLine) == 3:
            ptmp = ''.join([line.split('|||')[1] for i, line in enumerate(multiLine)])
            pOne += ptmp
        elif len(multiLine) == 2:
            index1 = getIndex(multiLine[0])
            index2 = getIndex(multiLine[1])
            #第一句为一段话最后一句
            if index2<index1:
                q, a = getQueryAndAnswer(multiLine[0])
                pOne=Chinese_word_extraction(pOne)
                q=Chinese_word_extraction(q)
                a=Chinese_word_extraction(a)
                qList.append(q)
                aList.append(a)
                pList.append(pOne)
                pOne = multiLine[1].split('|||')[1]  # 作为p的第一句
            #第二句为最后一句
            elif len(multiLine[1].split('|||')) == 3:
                    q, a = getQueryAndAnswer(multiLine[1])
                    q=Chinese_word_extraction(q)
                    a=Chinese_word_extraction(a)
                    qList.append(q)
                    aList.append(a)

                    ptmp = multiLine[0].split('|||')[1]
                    pOne += ptmp
                    pOne=Chinese_word_extraction(pOne)
                    pList.append(pOne)
                    pOne = ''
            #都不为文章最后一句
            else:
                ptmp = ''.join([line.split('|||')[1] for i, line in enumerate(multiLine)])
                pOne += ptmp

        if max_examples:
            # 读取多少行
            if int(n * 3) > max_examples:
                break
            n += 1

    return pList, qList, aList

def Chinese_word_extraction(content_raw):
    chinese_pattern = "([\u4e00-\u9fa5]+)"
    re_data = re.findall(chinese_pattern,content_raw)
    content_clean  = ' '.join(re_data)
    return content_clean

def build_dict(sentences, max_words=100000):
    wc = Counter()
    for s in sentences:
        for w in s.split(' '):
            wc[w] += 1

    ls = wc.most_common(max_words) #只显示topN的元素
    vocab_size=len(ls)
    logging.info('# of Words: %d -> %d' % (len(wc), len(ls)))

    for k in ls[:15]:
        logging.info(k)
    logging.info('...')
    for k in ls[-5:]:
        logging.info(k)

    return {w[0]: i+2 for (i,w) in enumerate(ls)}

def gen_embeddings(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = np.random.uniform(low=-0.01, high=0.01, size=(num_words, dim))
    logging.info('Embedding Matrix: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file at %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            v = line.split()
            assert len(v) == dim + 1
            if v[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[v[0]]:] = [float(x) for x in v[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                        (pre_trained, pre_trained * 100.0 / num_words))

    return embeddings



#将原始的字符转为索引表示
def vectorize(examples, word_dict, entity_dict,
              sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
    """
    in_x1 = []
    in_x2 = []
    in_y = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        # 此处未限制答案必须在原文中
        # assert (a in d_words)
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]  # 0 for unk
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_y.append(entity_dict[a] if a in entity_dict else 0)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

    return in_x1, in_x2, in_y


def get_minibatches(n, mb_size, shuffle=False):
    idx_list = np.arange(0, n, mb_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for i in idx_list:
        minibatches.append(np.arange(i, min(n, i+mb_size)))
    return minibatches


# 以最大seq长度进行填充序列
def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    # x_mask = np.zeros((n_samples, max_len)).astype(float)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        # x_mask[idx, :lengths[idx]] = 1.0
    return x  # , x_mask


def gen_examples(x1, x2, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_y = [y[t] for t in minibatch]

        mb_x1 = prepare_data(mb_x1)  # pading
        mb_x2 = prepare_data(mb_x2)
        all_ex.append((mb_x1, mb_x2, mb_y))
    return all_ex

def mkSummaryDir(summary_dir):
    # log_dir = 'summary/graph2/'
    if os.path.exists(summary_dir):   # 删掉以前的summary，以免重合
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)
    print('created summary path',summary_dir)




def get_set(oriList):
    # newList = copy.deepcopy(oriList)
    d_input_len = oriList.shape[1]  # 获取每个文章的有效字的个数
    x = np.array([[-1] * d_input_len] * oriList.shape[0])  # 构造一个形如mb_x1的皆为-1的数组
    for idx, nL in enumerate(oriList):
        setList = list(set(nL))
        # print(idx,setList)
        x[idx, :len(setList)] = setList

    return d_input_len, x


def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer