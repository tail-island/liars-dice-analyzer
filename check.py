import numpy as np

from data_set     import *
from funcy        import *
from keras.models import load_model
from statistics   import mode


def main():
    _, (validation_x, validation_y) = load_data()

    xs = tuple(map(rcompose(partial(map, first), tuple), partition_by(second, sorted(zip(validation_x, validation_y), key=second))))

    model = load_model("model.h5")

    t = 0
    c = 0

    for x, y_true in zip(xs, count()):
        for x_batch in partition(10, x):
            y = np.array(tuple(map(np.argmax, model.predict(np.array(x_batch), batch_size=len(x)))))

            try:
                y_mode = mode(y)
            except:
                y_mode = 9

            print('{} -> {}, {}'.format(y_true, y_mode, y))

            c += y_mode == y_true
            t += 1

    print('accuracy = {}'.format(c / t))


if __name__ == '__main__':
    main()
