from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(tf.__version__)
import pandas as pd
from sql.mysql import sqlHelper
import gensim

tdPath = "../data/fixdata_train.csv"
predictdata = "../data/fixdata_vail.csv"
wordin_path = "../data/wordindex.txt"
modelsavepath = "../data/lstm_model"
modelpath_w2v = '../data/word2VecModel'

vocab_size = 5000
embedding_dim = 64
max_length = 150
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = backend.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = backend.softmax(backend.tanh(backend.dot(x, self.W) + self.b))
        outputs = backend.permute_dimensions(a * x, (0, 2, 1))
        outputs = backend.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



def train(tdPath,wordin_path,modelsavepath,vocab_size = 5000,
            embedding_dim = 64,
            max_length = 150,
            trunc_type = 'post',
            padding_type = 'post',
            oov_tok = '<OOV>'):

    def getWordEmbedding(w2vpath, words):

        model = gensim.models.Word2Vec.load(w2vpath)
        wordEmbedding = []
        for word in words:
            try:
                # 中文
                vector = model[word]
                wordEmbedding.append(vector)
            except:
                wordEmbedding.append(np.random.randn(embedding_dim))
                print(word + " : 不存在于词向量中")
        return  np.array(wordEmbedding)

    data = pd.read_csv(tdPath)
    X_train, X_validation, Y_train, Y_validation = train_test_split(data[['cut']], data[["label"]], test_size=0.2,
                                                                    stratify=data.label)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train.cut.values)
    word_index = tokenizer.word_index
    words = [k for (k,v) in word_index.items() if v <=vocab_size]
    wordEmbedding = getWordEmbedding(modelpath_w2v, words)
    print(wordEmbedding.shape)
    with open(wordin_path,'w',encoding='utf-8') as ff:
        for (k,v) in word_index.items():
            ff.write(str(k)+"\t"+str(v)+"\n")
    # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    X_train_sequences = tokenizer.texts_to_sequences(X_train.cut.values)
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    X_validation_sequences = tokenizer.texts_to_sequences(X_validation.cut.values)
    X_validation_padded = pad_sequences(X_validation_sequences, maxlen=max_length, padding=padding_type,
                                        truncating=trunc_type)
    Y_training_cat_seq = np.array(Y_train.label.values)
    Y_validation_cat_seq = np.array(Y_validation.label.values)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[wordEmbedding]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,dropout=0.5,recurrent_dropout=0.5,return_sequences=True)),
        AttentionLayer(),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(22, activation='softmax')
    ])

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 10
    history = model.fit(X_train_padded,
                        Y_training_cat_seq,
                        epochs=num_epochs,
                        validation_data=(X_validation_padded, Y_validation_cat_seq),
                        verbose=2)

    # 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署
    if os.path.exists(modelsavepath):
        shutil.rmtree(modelsavepath)
        print("{} 删除成功".format(modelsavepath))
    model.save(modelsavepath, save_format="tf")


def predict(predictdata,modelpath,wordin_path):

    pre_data = pd.read_csv(predictdata)
    word_index = {}
    with open(wordin_path,'r',encoding='utf-8') as ff:
        for line in ff.readline():
            kv = line.strip().split('\t')
            word_index[kv[0]] = kv[1]
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok,word_index = word_index,index_word=reverse_word_index)
    pre_sequences = tokenizer.texts_to_sequences(pre_data.cut.values)
    pre_sequences_pad = pad_sequences(pre_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    model_loaded = tf.keras.models.load_model(modelpath)

    import time
    n = 0
    batch = 100
    helper = sqlHelper('10.240.5.5', 31005, 'epc', 'native', 'native')
    pre_data_id = pre_data['id'].values.tolist()
    while (n * batch < len(pre_sequences_pad)):
        q_slice = pre_sequences_pad[n * batch:(n + 1) * batch]
        q_pre_data_id = pre_data_id[n * batch:(n + 1) * batch]
        np_test = np.array(q_slice)
        q_tensor = tf.convert_to_tensor(np_test)
        q_class = model_loaded.predict_classes(q_tensor)
        sql_data = [(int(str(i[0])), i[1]) for i in zip(q_class, q_pre_data_id)]
        a = helper.updateMany('update tb_inform_fix_case set pre_label = %s where id = %s', sql_data)
        print(a)
        n = n + 1
        time.sleep(1)

def main(model="train"):
    if model == 'train':
        train(tdPath,wordin_path,modelsavepath,vocab_size = 5000,
            embedding_dim = 64,
            max_length = 150,
            trunc_type = 'post',
            padding_type = 'post',
            oov_tok = '<OOV>')
    if model == 'predict':
        predict()
    pass

if __name__ == '__main__':
    main(model="train")



# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_'+string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.legend([string, 'val_'+string])
#     plt.show()
#
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")
#
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix
#
# y_pred = model.predict(X_validation_padded)
# y_pred = y_pred.argmax(axis=1)
#
# labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
# conf_mat = confusion_matrix(Y_validation_cat_seq, y_pred)
# fig, ax = plt.subplots(figsize=(10,8))
# sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=labels, yticklabels=labels)
# plt.ylabel('actual results',fontsize=18);
# plt.xlabel('predict result',fontsize=18);
#
# from  sklearn.metrics import classification_report
# print('accuracy %s' % accuracy_score(y_pred, Y_validation_cat_seq))
# print(classification_report(Y_validation_cat_seq, y_pred,target_names=[str(w) for w in labels]))

