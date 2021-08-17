# Keras

import tensorflow as tf
from keras import backend as K

input = K.placeholder(shape=(10,32))    # 케라스

tf.placeholder(tf.float32,shape=(10,32))    # 텐서플로



#  Sequential Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(units=64, input_dim=784))
model.add(Activation('softmax'))


model = Sequential([
    Dense(64, input_shape=(748,),activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.8, nesterov=True)


from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                           patience=10, verbose=0, mode='auto')
model.fit(x_train, y_train, epochs=10, batch_size=64,
          callbacks=[TensorBoard(log_dir='/models/autoencoder'),
                     early_stop])

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
classes = model.predict(x_test, batch_size=64)



# Functional Model

inputs = Input(shape(784,))


x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)


model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
classes = model.predict(x_test, batch_size=64)
