import numpy as np
import tensorflow.contrib.keras as kr

def batch_iter(x, y, batch_size=64):
    '''
    :param x: content2id
    :param y: label2id
    :param batch_size: 每次进入模型的句子数量
    :return:
    '''
    data_len = len(x)
    x = np.array(x)
    y = np.array(y)
    num_batch = int((data_len - 1) / batch_size) + 1  # 计算一个epoch,需要多少次batch

    indices = np.random.permutation(data_len)  # 生成随机数列
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def process_seq(x_batch):
    '''
    :param x_batch: 计算一个batch里面最长句子 长度n
    :param y_batch:动态RNN 保持同一个batch里句子长度一致即可，sequence为实际句子长度
    :return: 对所有句子进行padding,长度为n
    '''
    seq_len = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in range(len(x_batch)):
        seq_len.append(len(x_batch[i]))

    x_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_len, padding='post', truncating='post')
    # y_pad = kr.preprocessing.sequence.pad_sequences(y_batch, max_len, padding='post', truncating='post')

    return x_pad, seq_len
