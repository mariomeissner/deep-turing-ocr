from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers import add, GRU, CuDNNGRU
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.models import Model
import keras.backend as K


def create_model(params, gpu=False, two_rnns=False):

    input_data = Input(name="input", shape=params["input_shape"], dtype="float32")
    conv1 = Conv2D(
        params["conv_filters"],
        params["kernel_size"],
        padding="same",
        activation=params["act"],
        kernel_initializer="he_normal",
        name="conv1",
    )(input_data)
    conv1 = MaxPooling2D(
        pool_size=(params["pool_size"], params["pool_size"]), name="max1"
    )(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(
        params["conv_filters"],
        params["kernel_size"],
        padding="same",
        activation=params["act"],
        kernel_initializer="he_normal",
        name="conv2",
    )(conv1)
    conv2 = MaxPooling2D(
        pool_size=(params["pool_size"], params["pool_size"]), name="max2"
    )(conv2)
    conv2 = Dropout(0.2)(conv2)

    # conv1shape = (img_w // (pool_size ** (num_convs - 1)),
    #                     (img_h // (pool_size ** (num_convs - 1))) * conv_filters)
    conv2shape = (
        params["img_w"] // (params["pool_size"] ** params["num_convs"]),
        (params["img_h"] // (params["pool_size"] ** params["num_convs"]))
        * params["conv_filters"],
    )

    # Failed attempt to do a skip connection

    # conv1 = Reshape(target_shape=conv1shape)(conv1)
    # conv2 = Reshape(target_shape=conv2shape)(conv2)
    # inner = concatenate([conv1, conv2], axis=2)

    inner = Reshape(target_shape=conv2shape, name="reshape")(conv2)

    # cuts down input size going into RNN:
    inner = Dense(params["time_dense_size"], activation=params["act"], name="dense1")(
        inner
    )

    if gpu:
        gru_1 = CuDNNGRU(
            params["rnn1_size"],
            return_sequences=True,
            kernel_initializer="he_normal",
            name="gru1",
        )(inner)
        gru_1b = CuDNNGRU(
            params["rnn1_size"],
            return_sequences=True,
            go_backwards=True,
            kernel_initializer="he_normal",
            name="gru1_b",
        )(inner)
    else:
        gru_1 = GRU(
            params["rnn1_size"],
            return_sequences=True,
            kernel_initializer="he_normal",
            name="gru1",
            reset_after=True,
            recurrent_activation="sigmoid",
        )(inner)
        gru_1b = GRU(
            params["rnn1_size"],
            return_sequences=True,
            go_backwards=True,
            kernel_initializer="he_normal",
            name="gru1_b",
            reset_after=True,
            recurrent_activation="sigmoid",
        )(inner)

    gru1_merged = add([gru_1, gru_1b])
    if two_rnns:
        if gpu:
            gru_2 = CuDNNGRU(
                params["rnn2_size"],
                return_sequences=True,
                kernel_initializer="he_normal",
                name="gru2",
            )(gru1_merged)
            gru_2b = CuDNNGRU(
                params["rnn2_size"],
                return_sequences=True,
                go_backwards=True,
                kernel_initializer="he_normal",
                name="gru2_b",
            )(gru1_merged)
        else:
            gru_2 = GRU(
                params["rnn2_size"],
                return_sequences=True,
                kernel_initializer="he_normal",
                name="gru2",
                reset_after=True,
                recurrent_activation="sigmoid",
            )(gru1_merged)
            gru_2b = GRU(
                params["rnn2_size"],
                return_sequences=True,
                go_backwards=True,
                kernel_initializer="he_normal",
                name="gru2_b",
                reset_after=True,
                recurrent_activation="sigmoid",
            )(gru1_merged)

    # transforms RNN output to character activations:
    if two_rnns:
        inner = Dense(
            params["output_size"], kernel_initializer="he_normal", name="dense2"
        )(concatenate([gru_2, gru_2b]))
    else:
        inner = Dense(
            params["output_size"], kernel_initializer="he_normal", name="dense2"
        )(gru1_merged)

    y_pred = Activation("softmax", name="softmax")(inner)
    output_labels = Input(
        name="the_labels", shape=[params["max_string_len"]], dtype="float32"
    )
    input_lengths = Input(name="input_length", shape=[1], dtype="int64")
    label_lengths = Input(name="label_length", shape=[1], dtype="int64")

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    # The loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, params["ctc_cut"] :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, output_labels, input_lengths, label_lengths]
    )

    train_model = Model(
        inputs=[input_data, output_labels, input_lengths, label_lengths], outputs=loss_out
    )

    top_k_dec_list, _ = K.ctc_decode(
        y_pred[:, params["ctc_cut"] :, :],
        K.squeeze(input_lengths, axis=1),
        greedy=False,
        top_paths=3,
    )
    decoder0 = K.function([input_data, input_lengths], [top_k_dec_list[0]])
    decoder1 = K.function([input_data, input_lengths], [top_k_dec_list[1]])
    decoder2 = K.function([input_data, input_lengths], [top_k_dec_list[2]])
    decoder_models = decoder0, decoder1, decoder2

    return train_model, decoder_models
