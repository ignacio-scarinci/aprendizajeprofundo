"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""

import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from skopt import gp_minimize
from skopt.space import Real, Integer


TARGET_COL = 'AdoptionSpeed'


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='./data/', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    args = parser.parse_args()

    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # TODO Create and append numeric columns
    # Don't forget to normalize!
    # ....
    for num_column in numeric_columns:
        numericos = tf.keras.utils.normalize(df[[num_column]].values)
        direct_features.append(numericos)

    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None

    return features, targets


def load_dataset(dataset_dir, test_size):
    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)

    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))

    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))

    return dataset, dev_dataset, test_dataset

def get_search_space():
    search_space = {
        "epochs": Integer(25, 55, name="epochs"),
        "hidden_layer_1": Integer(65, 130, name="hidden_layer_1"),
        "hidden_layer_2": Integer(65, 130, name="hidden_layer_2"),
        "hidden_layer_3": Integer(65, 130, name="hidden_layer_3"),
        "hidden_layer_4": Integer(65, 130, name="hidden_layer_4"),
        "hidden_layer_5": Integer(65, 130, name="hidden_layer_5"),
        "dropout_1": Real(low=0.1, high=0.5, prior='log-uniform', name="dropout_1"),
        "dropout_2": Real(low=0.01, high=0.5, prior='log-uniform', name="dropout_2"),
        "learning_rate": Real(low=1e-4, high=1e-3, prior='log-uniform', name='learning_rate')
    }
    return search_space

def hyperparam_value(param_name, param_list):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    return param_list[search_space_keys[param_name]]

def print_selected_hyperparams(param_values):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    for param_name in search_space_keys:
        print("\t", param_name, hyperparam_value(param_name, param_values))

def show_best(res):
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}
    print("Best value: %.4f" % res.fun)
    param_names = {idx: param_name for param_name, idx in search_space_keys.items()}
    best_params = {param_names[i]: param_value for i, param_value in enumerate(res.x)}
    print("Best params:")
    print(best_params)



def objetivo(params):
    tf.keras.backend.clear_session()
    batch_size = 32
    args = read_args()
    print_selected_hyperparams(params)

    epochs = hyperparam_value("epochs", params)
    learning_rate = hyperparam_value("learning_rate", params)
    hidden_layer_1 = hyperparam_value("hidden_layer_1", params)
    hidden_layer_2 = hyperparam_value("hidden_layer_2", params)
    hidden_layer_3 = hyperparam_value("hidden_layer_3", params)
    hidden_layer_4 = hyperparam_value("hidden_layer_4", params)
    hidden_layer_5 = hyperparam_value("hidden_layer_5", params)
    drop_1 = hyperparam_value("dropout_1", params)
    drop_2 = hyperparam_value("dropout_2", params)


    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]

    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: max(dataset[one_hot_col].max(), dev_dataset[one_hot_col].max(), test_dataset[one_hot_col].max())
        for one_hot_col in
        ['Gender', 'Color1', 'Type', 'MaturitySize', 'Sterilized', 'Color2', 'Color3', 'Vaccinated', 'Health']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1', 'Breed2']
    }
    numeric_columns = ['Age', 'Fee']

    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)

    # Create the tensorflow Dataset
    # TODO shuffle the train dataset!
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(buffer_size=800)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)[0]).batch(batch_size)

    # TODO: Build the Keras model
    # model = ....
    # Add one input and one embedding for each embedded column

    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])
    #  Modelo 4
    bn = layers.BatchNormalization(momentum=0)(features)
    dense1 = layers.Dense(hidden_layer_1, activation='relu', use_bias=True)(bn)
    dropout_1 = layers.Dropout(drop_1)(dense1)
    dense2 = layers.Dense(hidden_layer_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout_1)
    dense3 = layers.Dense(hidden_layer_3, activation='relu')(dense2)
    dropout_2 = layers.Dropout(drop_2)(dense3)
    dense4 = layers.Dense(hidden_layer_4, activation='relu')(dropout_2)
    dense_ = layers.Dense(hidden_layer_5, activation='relu')(dense4)
    ## Modelo 2
    # bn = layers.BatchNormalization(momentum=0)(features)
    # dense1 = layers.Dense(hidden_layer_1, activation='relu')(bn)
    # dropout_1 = layers.Dropout(drop_1)(dense1)
    #dense2 = layers.Dense(hidden_layer_2, activation='relu')(dropout_1)
    # dense_ = layers.Dense(hidden_layer_3, activation='relu')(dense2)
    ### Modelo  1
    #  dense1 = layers.Dense(hidden_layer_1, activation='relu')(features)
    #  dense_ = layers.Dense(hidden_layer_2, activation='relu')(dense1)
    ###  Modelo == 3:
    #dense1 = layers.Dense(hidden_layer_1,activation='relu')(features)
    #dense2 = layers.Dense(hidden_layer_2,activation='relu')(dense1)
    #dropout_1 = layers.Dropout(drop_1)(dense2)
    #dense_ = layers.Dense(hidden_layer_3,activation='relu')(dropout_1)

    output_layer = layers.Dense(nlabels, activation='softmax')(dense_)

    model = models.Model(inputs=inputs, outputs=output_layer)

    optimizador = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizador,
                  metrics=['accuracy'])

    # TODO: Fit the model
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_1', hidden_layer_1)
        mlflow.log_param('hidden_layer_2', hidden_layer_2)
        mlflow.log_param('hidden_layer_3', hidden_layer_3)
        mlflow.log_param('hidden_layer_4', hidden_layer_4)
        mlflow.log_param('hidden_layer_5', hidden_layer_5)
        mlflow.log_param('dropout_1', drop_1)
        mlflow.log_param('dropout_2', drop_2)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('epochs', epochs)

        # Train
        history = model.fit(train_ds, epochs=epochs)

        loss, accuracy = 0, 0
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)
        return (accuracy * (-1))

def main():
    search_space = get_search_space()
    search_space_keys, search_space_vals = zip(*search_space.items())
    search_space_keys = {param_name: idx
                         for idx, param_name in enumerate(search_space_keys)}

    iterations = 50

    exploration_result = gp_minimize(objetivo, search_space_vals, random_state=42, verbose=1,
                                     n_calls=iterations)
    show_best(exploration_result)

    print('All operations completed')

if __name__ == '__main__':
    main()
## Mejor model 2 {'epochs': 25, 'hidden_layer_1': 72, 'hidden_layer_2': 125, 'hidden_layer_3': 83,'dropout_1': 0.01, 'dropout_2': 0.4861876942269708, 'learning_rate':  0.0009780188953570491}
## Mejor modelo 1 {'epochs': 25, 'hidden_layer_1': 65, 'hidden_layer_2': 65, 'learning_rate': 0.0001}
## Mejor modelo 3 {'epochs': 25, 'hidden_layer_1': 84, 'hidden_layer_2': 130, 'hidden_layer_3': 65, 'dropout_1': 0.49999999999999994, 'learning_rate': 0.0001}
## Mejor modelo 4 {'epochs': 55, 'hidden_layer_1': 65, 'hidden_layer_2': 65, 'hidden_layer_3': 130, 'hidden_layer_4': 65, 'hidden_layer_5': 65, 'dropout_1': 0.49999999999999994, 'dropout_2': 0.49999999999999994, 'learning_rate': 0.001}
## Mejor modelo 4, bis {'epochs': 55, 'hidden_layer_1': 130, 'hidden_layer_2': 130, 'hidden_layer_3': 130, 'hidden_layer_4': 65, 'hidden_layer_5': 65, 'dropout_1': 0.49999999999999994, 'dropout_2': 0.49999999999999994, 'learning_rate': 0.0001}
