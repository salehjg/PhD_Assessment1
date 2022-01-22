import sys
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras import Sequential
import tensorflow as tf
from keras.utils import to_categorical
import h5py as h5
from keras import backend as K


class Question1Version1:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        print("Train-set Data Shape: ", self.x_train.shape, " Train-set Label Shape:", self.y_train.shape)
        print("Test-set Data Shape: ", self.x_test.shape, " Test-set Label Shape:", self.y_test.shape)
        self.model = self.get_model()
        self.BASE_DIR = '../dumps/'
        self.BASE_W_DIR = self.BASE_DIR + 'weights/'
        self.BASE_I_DIR = self.BASE_DIR + 'inference_outputs/'
        self.KERAS_W_DIR = 'keras_h5/'
        self.NUMPY_W_DIR = 'numpy_npy/'
        self.KERAS_W_FNAME = 'latest.h5'
        self.w = {}

    def get_model(self):
        model = Sequential()
        # 0
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

        # 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))

        # 2
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 3
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

        # 4
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))

        # 5
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 6
        model.add(Flatten())

        # 7
        model.add(Dense(512, activation='relu'))

        # 8
        model.add(Dense(10, activation='softmax'))

        model.summary()
        return model

    def train(self, batch_size, epochs):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        self.model.save_weights(filepath=self.BASE_W_DIR + self.KERAS_W_DIR + self.KERAS_W_FNAME, overwrite=True)

    def test(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def convert_weights_npy_v1(self):
        # conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'dense_1', 'dense_2', 'flatten_1', 'max_pooling2d_1', 'max_pooling2d_2'
        with h5.File(self.BASE_W_DIR + self.KERAS_W_DIR + self.KERAS_W_FNAME, "r") as f5:
            self.w['conv2d_1'] = {}
            self.w['conv2d_1']['bias'] = np.array(f5['conv2d_1']['conv2d_1']['bias:0'])
            self.w['conv2d_1']['weight'] = np.array(f5['conv2d_1']['conv2d_1']['kernel:0'])

            self.w['conv2d_2'] = {}
            self.w['conv2d_2']['bias'] = np.array(f5['conv2d_2']['conv2d_2']['bias:0'])
            self.w['conv2d_2']['weight'] = np.array(f5['conv2d_2']['conv2d_2']['kernel:0'])

            self.w['conv2d_3'] = {}
            self.w['conv2d_3']['bias'] = np.array(f5['conv2d_3']['conv2d_3']['bias:0'])
            self.w['conv2d_3']['weight'] = np.array(f5['conv2d_3']['conv2d_3']['kernel:0'])

            self.w['conv2d_4'] = {}
            self.w['conv2d_4']['bias'] = np.array(f5['conv2d_4']['conv2d_4']['bias:0'])
            self.w['conv2d_4']['weight'] = np.array(f5['conv2d_4']['conv2d_4']['kernel:0'])

            self.w['dense_1'] = {}
            self.w['dense_1']['bias'] = np.array(f5['dense_1']['dense_1']['bias:0'])
            self.w['dense_1']['weight'] = np.array(f5['dense_1']['dense_1']['kernel:0'])

            self.w['dense_2'] = {}
            self.w['dense_2']['bias'] = np.array(f5['dense_2']['dense_2']['bias:0'])
            self.w['dense_2']['weight'] = np.array(f5['dense_2']['dense_2']['kernel:0'])

            # flatten_1, max_pooling2d_1, and max_pooling2d_2 have no weights or biases.

    def save_item(self, dict_w, item_index, tag_name, npy_dir):
        def tuple2str(shape):
            s = '('
            for i in shape:
                s += str(i) + ','
            s += ')'
            return s

        np.save(
            npy_dir + (str(item_index) + '.') + tag_name + '.bias.npy',
            dict_w[tag_name]['bias']
        )
        print("** " + (tag_name + '.bias.npy') + tuple2str(dict_w[tag_name]['bias'].shape))
        np.save(
            npy_dir + (str(item_index) + '.') + tag_name + '.weight.npy',
            dict_w[tag_name]['weight']
        )
        print("** " + (tag_name + '.weight.npy') + tuple2str(dict_w[tag_name]['weight'].shape))

    def export_numpy_weights_only(self):
        npy_dir = self.BASE_W_DIR + self.NUMPY_W_DIR
        self.save_item(self.w, 0, 'conv2d_1', npy_dir)
        self.save_item(self.w, 1, 'conv2d_2', npy_dir)
        self.save_item(self.w, 3, 'conv2d_3', npy_dir)
        self.save_item(self.w, 4, 'conv2d_4', npy_dir)
        self.save_item(self.w, 7, 'dense_1', npy_dir)
        self.save_item(self.w, 8, 'dense_2', npy_dir)

    def export_input_data(self):
        npy_dir = self.BASE_DIR
        np.save(
            npy_dir + 'input_test_data.npy',
            self.x_test
        )
        np.save(
            npy_dir + 'input_test_label.npy',
            self.y_test
        )

    def run_inference(self, batch_size, export_intermediate_tensors=False):
        method1_results = self.model.predict(self.x_test, batch_size=batch_size)

        if export_intermediate_tensors:
            inp = self.model.input  # input placeholder
            out = [layer.output for layer in self.model.layers]  # all layer outputs
            get_outputs = K.function([inp, K.learning_phase()], out)

            layer_outs = get_outputs([self.x_test, 1.])

            if np.sum(method1_results - layer_outs[-1]) > 1e-4:
                print(
                    "Something went wrong. The results of inference runs for method1 and method2 do not match. Terminating ...")
                sys.exit(1)

            for i, tensor in enumerate(layer_outs, start=0):
                print(i, ': ', tensor.shape)
                np.save(
                    self.BASE_I_DIR + str(i) + '.output.tensor.npy',
                    tensor
                )


if __name__ == "__main__":
    q1 = Question1Version1()

    if len(sys.argv) == 2:
        if sys.argv[1] == '-t':  # train
            q1.train(batch_size=100, epochs=10)
            q1.test()
        else:
            if sys.argv[1] == '-e':  # export all
                # export the input tensors of the test-set (data and label).
                q1.export_input_data()

                # inference with intermediate tensor dumps (the outputs of each layer).
                q1.run_inference(batch_size=100, export_intermediate_tensors=True)

                # extract the weights from the previously saved keras weights file (h5) and then save the extracted weights in numpy format.
                q1.convert_weights_npy_v1()
                q1.export_numpy_weights_only()

            else:
                if sys.argv[1] == '-i':
                    q1.run_inference(batch_size=100, export_intermediate_tensors=False)
    else:
        print(
            "Wrong arguments. "
            "Use args listed below:\n"
            "\t-t\tTrain\n"
            "\t-e\tRun inference and then export all the weights and the inputs and the output tensors into the numpy files.\n"
            "\t-i\tInference Only")
