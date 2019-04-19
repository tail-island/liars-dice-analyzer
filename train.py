from data_set           import *
from funcy              import *
from keras.callbacks    import LearningRateScheduler
from keras.layers       import *
from keras.models       import Model, save_model
from keras.regularizers import l2
from operator           import getitem


def computational_graph():
    def dense(unit_size):
        return Dense(unit_size, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

    def relu():
        return Activation('relu')

    def softmax():
        return Activation('softmax')

    return rcompose(dense(1024), relu(),
                    dense( 512), relu(),
                    dense( 256), relu(),
                    dense( 128), relu(),
                    dense(  64), relu(),
                    dense(   5), softmax())


def main():
    (x, y), (validation_x, validation_y) = load_data()

    model = Model(*juxt(identity, computational_graph())(Input(shape=x[0].shape)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x, y, 100, 150, validation_data=(validation_x, validation_y), callbacks=[LearningRateScheduler(partial(getitem, tuple(take(150, concat(repeat(0.001, 100), repeat(0.0005, 25), repeat(0.00025))))))])

    save_model(model, "model.h5")


if  __name__ == '__main__':
    main()
