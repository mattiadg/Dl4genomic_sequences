from keras.layers import Convolution1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, Reshape, LSTM, merge, Input
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.preprocessing.text import one_hot


def build_conv(kernel_len, seq_len, encoding, nb_classes):
    beta=1e-6
    model = Sequential()
    #print model.output_shape
    model.add(Embedding(input_dim=encoding, output_dim=10, input_length=seq_len))
    model.add(Convolution1D(10, kernel_len, activation='tanh'))
    model.add(MaxPooling1D())
    #print model.output_shape
    model.add(Flatten())
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    return model

def build_lstm(emb_size, nb_classes):
    input_layer = Input(shape=(1573, ))
    emb = Embedding(output_dim=emb_size, input_dim=16)(input_layer)
    emb = MaxPooling1D()(emb)
    lstm_f = LSTM(40, return_sequences=True)(emb)
    lstm_b = LSTM(40, return_sequences=True, go_backwards=True)(emb)
    lstm = merge([lstm_f, lstm_b], mode='concat')
    lstm = Flatten()(lstm)
    dense = Dense(500, activation='relu')(lstm)
    out = Dense(nb_classes, activation='softmax', name='phylum')(dense)

    model = Model(input=input_layer, output=out)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    return model

def build_conv_lstm(kernel_len, seq_len, encoding, nb_classes):
    beta = 1e-2
    inputs = Input(shape=(seq_len,), dtype='int32')
    emb = Embedding(output_dim=10, input_dim=encoding, input_length=seq_len)(inputs)
    conv2 = Convolution1D(10, kernel_len, activation='relu')(emb)
    conv2 = MaxPooling1D(stride=2)(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv3 = Convolution1D(10, kernel_len, activation='relu')(emb)
    conv3 = MaxPooling1D(stride=2)(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv4 = Convolution1D(10, kernel_len, activation='relu')(emb)
    conv4 = MaxPooling1D(stride=2)(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv = merge([conv2, conv3, conv4], mode='concat')
    lstm_f = LSTM(20, return_sequences=True, dropout_U=0.1)(conv)#, dropout_U=0.5, dropout_W=0.5))
    lstm_b = LSTM(20, return_sequences=True, dropout_U=0.1)(conv)
    lstm = merge([lstm_f, lstm_b], mode='concat')
    #model.add(MaxPooling1D())
    lstm = MaxPooling1D(2, 2)(lstm)
    lstm = Dropout(0.5)(lstm)
    #model.add(Dense(128, activation='relu'))
    flat = Flatten()(lstm)
    dense = Dense(2*nb_classes, activation='relu')(flat)
    dense = Dropout(0.5)(dense)
    softmax = Dense(nb_classes, activation='softmax', W_regularizer=l2(beta))(dense)

    model = Model(input=inputs, output=softmax)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    return model
