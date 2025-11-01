import numpy as np
from Evaluation import evaluation
import cv2 as cv
from keras.layers import Input, Conv3D, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model


def residual_block(input_tensor, filters, dilation_rate=1):
    """
    Residual block with dilated convolutions.
    """
    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    # Residual connection
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def dilated_residual_attention_network(input_shape, num_classes=1):
    """
    Builds a Dilated Residual Attention Network for 3D image classification.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Stack of Residual Blocks with increasing dilation rates
    x = residual_block(x, 32, dilation_rate=1)
    x = residual_block(x, 32, dilation_rate=2)
    x = residual_block(x, 32, dilation_rate=4)

    # Further Convolutions for feature extraction
    x = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Global Pooling and Output
    x = GlobalAveragePooling2D()(x)

    # Classification Layer
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification

    model = Model(inputs, outputs)
    return model


def Model_RAN(Train_Data, Train_Target, Test_data, Test_Target):
    IMG_SIZE = [32, 32, 3]
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = cv.resize(Train_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Feat1 = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Test_data.shape[0]):
        Feat2[i, :] = cv.resize(Test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Feat2 = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    input_shape = (32, 32, 3)  # Add channel dimension for grayscale
    # Instantiate the model
    model = dilated_residual_attention_network(input_shape, num_classes=Train_Target.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Print the model summary
    model.summary()
    model.fit(Feat1, Train_Target, epochs=10)
    pred = model.predict(Feat2)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_Target)
    return np.asarray(Eval), pred

