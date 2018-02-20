# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:05:25 2018

@author: tgill
"""

from keras.layers import Input, Dense, Dropout, Activation, Bidirectional, CuDNNGRU, Embedding, Concatenate, CuDNNLSTM, Multiply, Add, Lambda, TimeDistributed, Dot, GlobalAvgPool1D, GlobalMaxPool1D, Permute, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.activations import softmax
from keras import backend as K

def nn(input_dim=56, output_dim=2, layers=3, units=32, dropout=0.2):
    inputs = Input(shape=(input_dim,))
    x=Dense(units, activation='relu')(inputs)
    x=Dropout(dropout)(x)
    x=Dense(units, activation='relu')(x)
    x=Dropout(dropout)(x)
    x=Dense(units, activation='relu')(x)
    x=Dropout(dropout)(x)
    x=Dense(output_dim)(x)
    x=Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def siamois(maxlen, max_features):
    inp1 = Input(shape=(maxlen,))
    inp2 = Input(shape=(maxlen,))
    
#    com = Bidirectional(CuDNNGRU(64, return_sequences=True))
#    com = Dropout(0.3)(com)
    emb = Embedding(max_features, 256)
    #com = Bidirectional(CuDNNGRU(64, return_sequences=False))
    com = CuDNNLSTM(64, return_sequences=False)
    #com2 = CuDNNGRU(64, return_sequences=False)
    
    x1 = emb(inp1)
    x1 = com(x1)
    #x1 = Dropout(0.2)(x1)
    #x1 = com2(x1)
    
    x2 = emb(inp2)
    x2 = com(x2)
    #x2 = Dropout(0.2)(x2)
    #x2 = com2(x2)
    
    #merge=Concatenate()([x1, x2])
    merge = submult(x1, x2)
    merge = Dropout(0.2)(merge)
    merge = Dense(512, activation='relu')(merge)
    merge = Dropout(0.2)(merge)
    #merge = Dense(256, activation='relu')(merge)
    #merge = Dropout(0.2)(merge)
    
    preds = Dense(2, activation='softmax')(merge)
    
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

def siamois_seq(maxlen, max_features):
    inp1 = Input(shape=(maxlen,))
    inp2 = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, 256)
    com = CuDNNGRU(64, return_sequences=True)
    
    x1 = emb(inp1)
    x1 = com(x1)
    
    x2 = emb(inp2)
    x2 = com(x2)
    
    pool = GlobalMaxPool1D()
    avg = GlobalAvgPool1D()
    
    x1 = Concatenate()([pool(x1), avg(x1)])
    x2 = Concatenate()([pool(x2), avg(x2)])
    
    merge = submult(x1, x2)
    merge = Dropout(0.1)(merge)
    merge = Dense(512, activation='relu')(merge)
    merge = Dropout(0.2)(merge)
    
    preds = Dense(2, activation='softmax')(merge)
    
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

def decomposable_attention(maxlen, max_features, projection_hidden=0, projection_dropout=0.2, projection_dim=64, compare_dim=128, compare_dropout=0.2, dense_dim=64, dense_dropout=0.2):#maxlen, max_features, projection_hidden=0, projection_dropout=0.2, projection_dim=300, compare_dim=500, compare_dropout=0.2, dense_dim=300, dense_dropout=0.2
    inp1 = Input(shape=(maxlen,))
    inp2 = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, 256)
    emb1 = emb(inp1)
    emb2 = emb(inp2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation='relu'),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    encoded1 = time_distributed(emb1, projection_layers)
    encoded2 = time_distributed(emb2, projection_layers)
    
    # Attention
    att1, att2 = soft_attention_alignment(encoded1, encoded2)
    
    # Compare
    combine1 = Concatenate()([encoded1, att2, submult(encoded1, att2)])
    combine2 = Concatenate()([encoded2, att1, submult(encoded2, att1)])
    compare_layers = [
        Dense(compare_dim, activation='relu'),
        Dropout(compare_dropout),
        Dense(compare_dim, activation='relu'),
        Dropout(compare_dropout),
    ]
    compare1 = time_distributed(combine1, compare_layers)
    compare2 = time_distributed(combine2, compare_layers)
    
    # Aggregate
    agg1 = apply_multiple(compare1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    agg2 = apply_multiple(compare2, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Merge
    merge = Concatenate()([agg1, agg2])
    #merge = BatchNormalization()(merge)
    dense = Dense(dense_dim, activation='relu')(merge)
    dense = Dropout(dense_dropout)(dense)
    #dense = BatchNormalization()(dense)
    #dense = Dense(dense_dim, activation='relu')(dense)
    #dense = Dropout(dense_dropout)(dense)
    
    preds = Dense(2, activation='softmax')(dense)
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model
    

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_
    
def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_

def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_

def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned
    
def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_
    
    
def esim(maxlen, max_features, lstm_dim=32, dense_dim=64, dense_dropout=0.5):
    inp1 = Input(shape=(maxlen,))
    inp2 = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, 256)
    emb1 = emb(inp1)
    emb2 = emb(inp2)
    
    #Encode
    encode = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    encoded1=encode(emb1)
    encoded2=encode(emb2)
    
    #Attention
    att1, att2 = soft_attention_alignment(encoded1, encoded2)
    
    #Compose
    comb1 = Concatenate()([encoded1, att2, submult(encoded1, att2)])
    comb2 = Concatenate()([encoded2, att1, submult(encoded2, att1)])
    
    compose = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    compare1 = compose(comb1)
    compare2 = compose(comb2)
    
    #Aggregate
    agg1 = apply_multiple(compare1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    agg2 = apply_multiple(compare2, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    #Merge
    merge = Concatenate()([agg1, agg2])
    dense = Dense(dense_dim, activation='relu')(merge)
    dense = Dropout(dense_dropout)(dense)
    
    preds = Dense(2, activation='softmax')(dense)
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

def multiple_conv(input_, convs, pool):
    agg_ = []
    for conv in convs:
        agg_.append(pool(conv(input_)))
    out_ = Concatenate()(agg_)
    return out_

def apply_serie(input_, layers):
    x = input_
    for layer in layers:
        x = layer(x)
    return x
    
    

def siamois_cnn(maxlen, max_features, filters=64, sizes=[2, 3, 5, 8], embedding_matrix=None):
    inp1 = Input(shape=(maxlen,))
    inp2 = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, 128)
    if embedding_matrix is not None:
        embed_size = embedding_matrix.shape[1]
        emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)
    
    
    convs=[]
    for size in sizes:
        convs.append(Conv1D(filters=filters, kernel_size=size, activation='relu', padding='valid'))
    
    emb1 = emb(inp1)
#    x1a = apply_multiple(conv1(emb1), layers)
#    x1b = apply_multiple(conv2(emb1), layers)
#    x1c = apply_multiple(conv3(emb1), layers)
#    x1d = apply_multiple(conv4(emb1), layers)
#    x1a = pool(conv1(emb1))
#    x1b = pool(conv2(emb1))
#    x1c = pool(conv3(emb1))
#    x1d = pool(conv4(emb1))
#    x1e = pool(conv5(emb1))
    
    
    emb2 = emb(inp2)
#    x2a = apply_multiple(conv1(emb2), layers)
#    x2b = apply_multiple(conv2(emb2), layers)
#    x2c = apply_multiple(conv3(emb2), layers)
#    x2d = apply_multiple(conv4(emb2), layers)
#    x2a = pool(conv1(emb2))
#    x2b = pool(conv2(emb2))
#    x2c = pool(conv3(emb2))
#    x2d = pool(conv4(emb2))
#    x2e = pool(conv5(emb2))
    
#    x1 = Concatenate()([x1a, x1b, x1c, x1d, x1e])
#    x2 = Concatenate()([x2a, x2b, x2c, x2d, x2e])
    
    x1 = multiple_conv(emb1, convs, GlobalMaxPool1D())
    x2 = multiple_conv(emb2, convs, GlobalMaxPool1D())
    
    merge = submult(x1, x2)
    
    #gru = CuDNNGRU(64, return_sequences=True)
    #gru1 = apply_multiple(gru(emb1), [GlobalAvgPool1D(), GlobalMaxPool1D()])
    #gru2 = apply_multiple(gru(emb2), [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    #gru_merge = submult(gru1, gru2)
    
    #merge = Concatenate()([merge, gru_merge])
    
    merge = Dropout(0.1)(merge)
    merge = Dense(512, activation='relu')(merge)
    merge = Dropout(0.2)(merge)
    
    preds = Dense(2, activation='softmax')(merge)
    
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

def siamois_char(maxlen, max_features, filters=64, sizes=[2, 3, 5, 8], embedding_matrix=None):
    inp1 = Input(shape=(maxlen,), dtype='uint8')
    inp2 = Input(shape=(maxlen,), dtype='uint8')
    
    emb = Embedding(max_features, 16)
    #emb = Lambda(K.one_hot, arguments={'num_classes':max_features}, output_shape=(maxlen, max_features))
    
    emb1 = emb(inp1)  
    emb2 = emb(inp2)
    
    convs = []
    convs.append(Conv1D(filters=filters, kernel_size=7, padding='same', activation='relu'))
    convs.append(MaxPooling1D(pool_size=3))
    convs.append(Conv1D(filters=filters, kernel_size=7, padding='same', activation='relu'))
    convs.append(MaxPooling1D(pool_size=3))
    convs.append(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
    convs.append(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
    convs.append(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
    convs.append(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
    convs.append(MaxPooling1D(pool_size=3))
    
#    x1 = Concatenate()([x1a, x1b, x1c, x1d, x1e])
#    x2 = Concatenate()([x2a, x2b, x2c, x2d, x2e])
    
    x1 = apply_serie(emb1, convs)
    x2 = apply_serie(emb2, convs)
    
    x1 = GlobalMaxPool1D()(x1)
    x2 = GlobalMaxPool1D()(x2)
    
#    x1 = Flatten()(x1)
#    x2 = Flatten()(x2)
    
    merge = submult(x1, x2)
    
    merge = Dropout(0.1)(merge)
    merge = Dense(512, activation='relu')(merge)
    merge = Dropout(0.2)(merge)
#    merge = Dense(512, activation='relu')(merge)
#    merge = Dropout(0.2)(merge)
    
    preds = Dense(2, activation='softmax')(merge)
    
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

def deep_char(maxlen, max_features=64, nblocks=4, ns = [2, 2, 2, 2]):
    inp1 = Input(shape=(maxlen,), dtype='uint8')
    inp2 = Input(shape=(maxlen,), dtype='uint8')
    
    emb = Embedding(max_features, 16)
    
    emb1 = emb(inp1)
    emb2 = emb(inp2)
    
    first_conv = Conv1D(64, kernel_size=3, activation='relu')
    x1 = first_conv(emb1)
    x2 = first_conv(emb2)
    for i in range(nblocks):
        conv1 =  Conv1D(64*2**(i), kernel_size=3, padding='same')
        y1 = conv1(x1)
        y2 = conv1(x1)
        #y1 = BatchNormalization()(y1)
        #y2 = BatchNormalization()(y2)
        y1 = Activation('relu')(y1)
        y2 = Activation('relu')(y2)
        
        conv2 = Conv1D(64*2**(i), kernel_size=3, padding='same')
        y1 = conv2(y1)
        y2 = conv2(y2)
        #y1 = BatchNormalization()(y1)
        #y2 = BatchNormalization()(y2)
        y1 = Activation('relu')(y1)
        y2 = Activation('relu')(y2)
        
#        l = ns[i]
#        for j in range(l):
#            conv =  Conv1D(64*2**(i), kernel_size=3, padding='same')
#            y1 = conv(x1)
#            y2 = conv(x1)
#            y1 = BatchNormalization()(y1)
#            y2 = BatchNormalization()(y2)
#            y1 = Activation('relu')(y1)
#            y2 = Activation('relu')(y2)
        
        
        #conv = Conv1D(filters=64*2**(i), kernel_size=1, padding='same')(x)
        #y1 = Add()([x1, y1])
        #y2 = Add()([x2, y2])
        
        if i!=nblocks-1:
            x1 = MaxPooling1D(pool_size=3, strides=2)(y1)
            x2 = MaxPooling1D(pool_size=3, strides=2)(y2)
#    x1 = Flatten()(x1)
#    x2 = Flatten()(x2)
    x1 = GlobalMaxPool1D()(x1)
    x2 = GlobalMaxPool1D()(x2)
    merge = submult(x1, x2)
    merge = Dropout(0.1)(merge)
    x = Dense(512, activation='relu')(merge)
    x = Dropout(0.2)(x)
    #x = Dense(512, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=[inp1, inp2], outputs=preds)
    print(model.summary())
    return model

