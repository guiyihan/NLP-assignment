# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, re
print(os.listdir("../input"))
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, Bidirectional,concatenate
from keras.layers import LSTM,  GlobalMaxPooling1D, GlobalAveragePooling1D,CuDNNLSTM,Conv1D,MaxPooling1D,Flatten,Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from keras.losses import sparse_categorical_crossentropy
from keras import backend as K

DATA_DIR="../input/relation-classification/open_data/open_data/"
EMBEDDING_FILES = [
    ('../input/chinese-word-embedding/tencent_ailab_chineseembedding/Tencent_AILab_ChineseEmbedding.txt',200)
    # ,('../input/fasttext-chinese-word-embedding/cc.zh.300.vec/cc.zh.300.vec',300),
    # ,('../input/chinesewordvectors/sgns.merge.word.txt', 300)
]
BATCH_SIZE = 64
DENSE_HIDDEN_UNITS = 200
CONV_SIZE = 64
EPOCHS = 100
PATIENCE = 10
MAX_LEN = 72
LEARN_RATE= 0.001


def load_data(concate=True):
    file = open(DATA_DIR+'sent_train.txt',encoding='utf-8')
    train_df = file.readlines()
    file.close()
    train_df = [i.split('\t',3) for i in train_df]
    train_df = np.asarray(train_df)
    file = open(DATA_DIR+'sent_dev.txt',encoding='utf-8')
    valid_df = file.readlines()
    file.close()
    valid_df = [i.split('\t',3) for i in valid_df]
    valid_df = np.asarray(valid_df)
    file = open(DATA_DIR+'sent_test.txt',encoding='utf-8')
    test_df = file.readlines()
    file.close()
    test_df = [i.split('\t',3) for i in test_df]
    test_df = np.asarray(test_df)

    train_relation_df=pd.read_table(DATA_DIR+'sent_relation_train.txt',encoding='utf-8', header=None)
    valid_relation_df=pd.read_table(DATA_DIR+'sent_relation_dev.txt',encoding='utf-8', header=None)
    test_relation_df=pd.read_table(DATA_DIR+'sent_relation_test.txt',encoding='utf-8', header=None)

    x_train = train_df[:,1:].astype(str)
    y_train = np.asarray(train_relation_df.iloc[:,1])

    x_valid = valid_df[:,1:].astype(str)
    y_valid = np.asarray(valid_relation_df.iloc[:,1])

    x_test = test_df[:,1:].astype(str)
    if concate==True:
        x_train = np.concatenate([x_train,x_valid],axis=0)
        y_train = np.concatenate([y_train, y_valid],axis=0)
    return x_train,y_train,x_valid,y_valid,x_test
    
def preprocessing(x_train,y_train,x_valid,y_valid,x_test):
    assert x_train.shape[0]==y_train.shape[0]

    punct = "[a-zA-Z.\"！*\[\]\-》,：<>'“﹝﹞—、（）()？?·~~/【】，。；：:《》…’“”‘]"
    punct = re.compile(punct)

    for i in range(x_train.shape[0]):
        text=x_train[i,2]
        x_train[i,2]=punct.sub('',text)
    
    for i in range(x_valid.shape[0]):
        text=x_valid[i,2]
        x_valid[i,2]=punct.sub('',text)

    for i in range(x_test.shape[0]):
        text=x_test[i,2]
        x_test[i,2]=punct.sub('',text)
    
    for i in range(len(y_train)):
        if isinstance(y_train[i],str):
            y_train[i]=y_train[i].split(' ')[0]
    y_train = y_train.astype(float)
    
    for i in range(len(y_valid)):
        if isinstance(y_valid[i],str):
            y_valid[i]=y_valid[i].split(' ')[0]
    y_valid = y_valid.astype(float)    

    return x_train,y_train,x_valid,y_valid,x_test
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding='UTF-8') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path, word_embedding_length=300):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, word_embedding_length))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

def custom_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred)*y_true[:,1]

def loss_weight(y_true):
    weight = np.ones(y_true.shape)
    for i in range(len(y_true)):
        if y_true[i]==0:
            weight[i]=0.2
    return np.vstack([y_true, weight]).T
    # return [y_true, weight]

def window_x_train(x_train):
    x_train1 = np.zeros(x_train.shape)
    x_train1[:,0:-1] = x_train[:,1:]
    x_train2 = np.zeros(x_train.shape)
    x_train2[:,1:] = x_train[:,0:-1]
    return [x_train,x_train1,x_train2]

def build_model(embedding_matrix):
    words1 = Input(shape=(MAX_LEN,))
    words2 = Input(shape=(MAX_LEN,))
    words3 = Input(shape=(MAX_LEN,))
    embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=True)
    x1 = embedding_layer(words1)
    x2 = embedding_layer(words2)
    x3 = embedding_layer(words3)
    x = concatenate([x1,x2,x3])
    x = SpatialDropout1D(0.2)(x)
    
    x = Dense(DENSE_HIDDEN_UNITS, activation="linear")(x)

    # x1 = Conv1D(CONV_SIZE, 3, activation='relu', padding='same')(x)
    # x1 = MaxPooling1D(5, padding='same')(x1)
    # x1 = Conv1D(CONV_SIZE, 3, activation='relu', padding='same')(x)
    # x1 = MaxPooling1D(5, padding='same')(x1)
    # x1 = Flatten()(x1)
    # x2 = Conv1D(CONV_SIZE, 4, activation='relu', padding='same')(x)
    # x2 = MaxPooling1D(5, padding='same')(x2)
    # x2 = Conv1D(CONV_SIZE, 4, activation='relu', padding='same')(x)
    # x2 = MaxPooling1D(5, padding='same')(x2)
    # x2 = Flatten()(x2)
    # # x3 = Conv1D(conv_size, 5, activation='relu', padding='same')(x)
    # # x3 = MaxPooling1D(5, padding='same')(x3)
    # # x3 = Flatten()(x3)
    # x = concatenate([x1,x2])

    # x = concatenate([
    #     GlobalMaxPooling1D()(x),
    #     GlobalAveragePooling1D()(x),
    # ])
    # x = Dense(dense_units, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    result = Dense(35, activation="softmax")(x)

    model = Model(inputs=[words1,words2,words3], outputs=result)
    model.compile(loss=custom_loss, optimizer='adam') #, metrics=['acc']
    model.summary()

    return model

def model_train():
    x_train,y_train,x_valid,y_valid,x_test = load_data(concate=True)
    x_train,y_train,x_valid,y_valid,x_test = preprocessing(x_train,y_train,x_valid,y_valid,x_test)
    x_train=x_train[:,2]
    x_valid=x_valid[:,2]
    x_test=x_test[:,2]
    tokenizer = text.Tokenizer(lower=False)
    tokenizer.fit_on_texts(list(np.concatenate([x_train,x_valid,x_test],axis=0)))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_valid = tokenizer.texts_to_sequences(x_valid)
    x_valid = sequence.pad_sequences(x_valid, maxlen=MAX_LEN)
    
    embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f, length) for f,length in EMBEDDING_FILES], axis=-1)
    num_folds = 5
    patience = PATIENCE
    folds = KFold(n_splits=num_folds, shuffle=True)

    predict_sparse = np.zeros(len(x_train))
    predict = np.zeros((len(y_valid),35))
    for fold_n, (train_index, valid_index) in enumerate(folds.split(x_train)):
        model = build_model(embedding_matrix)
        X_valid = x_train[valid_index]
        Y_valid = y_train[valid_index]

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        
        X_train = window_x_train(X_train)
        X_valid = window_x_train(X_valid)
        
        Y_train = loss_weight(Y_train)
        Y_valid = loss_weight(Y_valid)
        earlyStop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        TB = TensorBoard(log_dir='./log_fold' + str(fold_n), histogram_freq=0, batch_size=BATCH_SIZE,
                         write_graph=True, write_grads=False, write_images=False,
                         embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None, embeddings_data=None,
                         update_freq='epoch')

        model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            validation_data=(X_valid, Y_valid),
            callbacks=[earlyStop, TB]
        )
        
        predict_valid = model.predict(X_valid, BATCH_SIZE)
        for i in range(len(predict_valid)):
            predict_sparse[valid_index[i]] = predict_valid[i].argmax()
        del model


    Y_valid = y_train
    print('acc:',accuracy_score(Y_valid, predict_sparse))
    print('F1 micro:', f1_score(Y_valid, predict_sparse, average='micro'))
    print('F1 macro:', f1_score(Y_valid, predict_sparse, average='macro'))
    
    print('confusion_matrix')
    file = open('./result.txt',mode='w')
    file.write('Acc:'+str(accuracy_score(Y_valid, predict_sparse))+'\n')
    file.write('F1 macro:'+str(f1_score(Y_valid, predict_sparse, average='macro'))+'\n')
    
    file.close()
    file=pd.DataFrame(confusion_matrix(Y_valid, predict_sparse))
    file.to_csv('confusion_matrix.csv')


model_train()