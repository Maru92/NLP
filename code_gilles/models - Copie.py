# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:05:25 2018

@author: tgill
"""

from keras.layers import Input, Dense, Dropout, Activation, Bidirectional, CuDNNGRU, Embedding, Concatenate, CuDNNLSTM, Multiply, Add, Lambda, TimeDistributed, Dot, GlobalAvgPool1D, GlobalMaxPool1D, Permute, BatchNormalization
from keras.models import Model
from keras.activations import softmax

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
    merge = Dropout(0.2)(merge)
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
    