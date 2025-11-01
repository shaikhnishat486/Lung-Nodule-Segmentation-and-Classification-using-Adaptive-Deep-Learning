import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.layers import Dense, Attention
from Evaluation import evaluation


def attention_block(input_tensor):
    query = Dense(input_tensor.shape[-1])(input_tensor)
    value = Dense(input_tensor.shape[-1])(input_tensor)
    attention_output = Attention()([query, value])
    return attention_output


# Define Vision Transformer model
def create_vit_model(train_data, train_target, test_data, test_target, sol):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Patch embedding
    x = layers.Conv2D(64, kernel_size=16, strides=16, activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=16, strides=16, activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=16, strides=16, activation="relu")(x)
    x = layers.Flatten()(x)
    # Positional embeddings
    x = layers.Embedding(input_dim=16 * 16, output_dim=64)(x)
    # Transformer encoder
    transformer_block = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)
    x = transformer_block(x, x)
    # Attention block
    x = attention_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(train_target.shape[1], activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Display model summary
    model.summary()

    model.fit(train_data, train_target, batch_size=64, epochs=sol[1])

    return model


def Model_ViT(Feat, Target, sol=None):
    if sol is None:
        sol = [5, 5, 0.01]
    learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
    train_data = Feat[:learnperc, :]
    train_target = Target[:learnperc, :]
    test_data = Feat[learnperc:, :]
    test_target = Target[learnperc:, :]

    IMG_SIZE = 224
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    model = create_vit_model(Train_X, train_target, Test_X, test_target, sol)
    pred = model.predict(test_data)
    Eval = evaluation(pred, test_target)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    return Eval, pred



