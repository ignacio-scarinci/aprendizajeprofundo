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
from tensorflow.keras import layers, models, callbacks


from sklearn.preprocessing import MinMaxScaler

TARGET_COL = 'AdoptionSpeed'


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='./data/', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    parser.add_argument('--learning_rate',default=1e-4,type=float,
                        help='Tasa de aprendizaje del optimizador')
    parser.add_argument('--modelo',type=int,default=1,
                        help='Modelo a experimentar')
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


class MlflowCallback(callbacks.Callback):

    # This function will be called after each epoch.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        # Log the metrics from Keras to MLflow
        mlflow.log_metric('loss', logs['loss'], step = epoch)
        mlflow.log_metric('val_loss',logs['val_loss'],step=epoch)
        mlflow.log_metric('accuracy', logs['accuracy'], step = epoch)
        mlflow.log_metric('val_accuracy',logs['val_accuracy'],step=epoch)

        # This function will be called after training completes.
    def on_train_end(self, logs=None):
        mlflow.log_param('num_layers', len(self.model.layers))
        mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)

def main():
    tf.keras.backend.clear_session()
    args = read_args()
    batch_size = args.batch_size
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: max(dataset[one_hot_col].max(), dev_dataset[one_hot_col].max(), test_dataset[one_hot_col].max())
        for one_hot_col in ['Gender', 'Color1','Type','MaturitySize','Sterilized','Color2','Color3','Vaccinated','Health']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1','Breed2']
    }
    numeric_columns = ['Age', 'Fee']
    
    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    
    # Create the tensorflow Dataset
    # TODO shuffle the train dataset!
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size) .shuffle(buffer_size=800)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)) .batch(batch_size)
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

    modelo = args.modelo
    if modelo == 1:
        dense1 = layers.Dense(130,activation='relu')(features)
        dense_ = layers.Dense(65,activation='relu')(dense1)
    elif modelo == 2:
        bn = layers.BatchNormalization(momentum=0)(features)
        dense1 = layers.Dense(65,activation='relu')(bn)
        dropout_1 = layers.Dropout(0.499)(dense1)
        dense2 = layers.Dense(65,activation='relu')(dropout_1)
        dense_ = layers.Dense(130,activation='relu')(dense2)
    elif modelo == 3:
        dense1 = layers.Dense(65,activation='relu')(features)
        dense2 = layers.Dense(130,activation='relu')(dense1)
        dropout_1 = layers.Dropout(0.01)(dense2)
        dense_ = layers.Dense(130,activation='relu')(dropout_1)
    elif modelo == 4:
        bn = layers.BatchNormalization(momentum=0)(features)
        dense1 = layers.Dense(130,activation='relu',use_bias=True)(bn)
        dropout_1 = layers.Dropout(0.4999)(dense1)
        dense2 = layers.Dense(65,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout_1)
        dense3 = layers.Dense(130,activation='relu')(dense2)
        dropout_2 = layers.Dropout(0.4999)(dense3)
        dense4 = layers.Dense(130,activation='relu')(dropout_2)
        dense_ = layers.Dense(130,activation='relu')(dense4)

    output_layer = layers.Dense(nlabels, activation='softmax')(dense_)

    model = models.Model(inputs=inputs, outputs=output_layer)
    lr=args.learning_rate
    optimizador = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizador,
                  metrics=['accuracy'])
    model.summary()

    # TODO: Fit the model
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
       # mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numerical_columns', numeric_columns)  # Not using these yet
        mlflow.log_param('epochs', args.epochs)


        # Train
        history = model.fit(train_ds, epochs=args.epochs,callbacks=[MlflowCallback()],validation_data=dev_ds)

        # TODO: analyze history to see if model converges/overfits
        
        # TODO: Evaluate the model, calculating the metrics.
        # Option 1: Use the model.evaluate() method. For this, the model must be
        # already compiled with the metrics.
        # performance = model.evaluate(X_test, y_test)

        loss, accuracy = 0, 0
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        #mlflow.log_metric('loss', loss)
        #mlflow.log_metric('accuracy', accuracy)
        
        # Option 2: Use the model.predict() method and calculate the metrics using
        # sklearn. We recommend this, because you can store the predictions if
        # you need more analysis later. Also, if you calculate the metrics on a
        # notebook, then you can compare multiple classifiers.
        archivo = 'modelo_{}.png'.format(modelo)
        tf.keras.utils.plot_model(model, to_file=archivo,show_shapes=False)
        predictions = 'No prediction yet'
        # predictions = model.predict(test_ds)

        # TODO: Convert predictions to classes
        # TODO: Save the results for submission
        # ...
        print(predictions)


print('All operations completed')

if __name__ == '__main__':
    main()
