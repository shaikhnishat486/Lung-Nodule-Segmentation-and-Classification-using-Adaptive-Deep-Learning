import tensorflow as tf
from keras import layers, models
from Evaluation import net_evaluation


def conv_block(x, filters, kernel_size, dilation_rate=1):
    """ Convolutional block with Batch Normalization and ReLU activation. """
    x = layers.Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def dense_block(x, filters, layers_in_block):
    """ Dense block with multiple conv layers and dense connections. """
    for _ in range(layers_in_block):
        conv = conv_block(x, filters, kernel_size=(3, 3))
        x = layers.Concatenate()([x, conv])
    return x


def aspp_block(x, filters):
    """ Atrous Spatial Pyramid Pooling block. """
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, -1))(pool)
    pool = layers.Conv2D(filters, (1, 1), padding="same")(pool)
    pool = tf.image.resize(pool, (x.shape[1], x.shape[2]))

    conv1 = conv_block(x, filters, kernel_size=(1, 1), dilation_rate=1)
    conv6 = conv_block(x, filters, kernel_size=(3, 3), dilation_rate=6)
    conv12 = conv_block(x, filters, kernel_size=(3, 3), dilation_rate=12)
    conv18 = conv_block(x, filters, kernel_size=(3, 3), dilation_rate=18)

    x = layers.Concatenate()([pool, conv1, conv6, conv12, conv18])
    x = layers.Conv2D(filters, (1, 1), padding="same")(x)
    return x


def trans_dilated_dense_aspp(input_shape, num_classes, filters=64, depth=4, layers_per_block=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Initial Conv Layer
    x = conv_block(x, filters, kernel_size=(3, 3))

    # Dense Blocks with Transition Layers
    for _ in range(depth):
        x = dense_block(x, filters, layers_per_block)
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)  # transition layer
        x = layers.MaxPooling2D((2, 2))(x)

    # ASPP Block
    x = aspp_block(x, filters)

    # Upsampling and Final Layers
    x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, x)
    return model


def Model_DenseASPP(Image, gt):
    model = trans_dilated_dense_aspp(Image.shape, 10)
    model.fit(Image, steps_per_epoch=500, epochs=10)
    model.summary()
    results = model.predict(Image.shape, gt.shape, verbose=1)

    Eval = net_evaluation(Image, results)
    return Eval, results
