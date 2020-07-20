import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import nfp

from preprocess_inputs import preprocessor
preprocessor.from_json('tfrecords/preprocessor.json')

def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        **preprocessor.tfrecord_features,
        **{'bde': tf.io.FixedLenFeature([], dtype=tf.string),
           'bdfe': tf.io.FixedLenFeature([], dtype=tf.string)}})

    # All of the array preprocessor features are serialized integer arrays
    for key, val in preprocessor.tfrecord_features.items():
        if val.dtype == tf.string:
            parsed[key] = tf.io.parse_tensor(
                parsed[key], out_type=preprocessor.output_types[key])
    
    # Pop out the prediction target from the stored dictionary as a seperate input
    parsed['bde'] = tf.io.parse_tensor(parsed['bde'], out_type=tf.float64)
    parsed['bdfe'] = tf.io.parse_tensor(parsed['bdfe'], out_type=tf.float64)
    
    bde = parsed.pop('bde')
    bdfe = parsed.pop('bdfe')    
    
    return parsed, bde

max_atoms = 48  # These allow prespecifying array shapes, i.e. for CUDA cores
max_bonds = 64
batch_size = 128
atom_features = 128
num_messages = 6


# Here, we have to add the prediction target padding onto the input padding
padded_shapes = (preprocessor.padded_shapes(max_atoms=max_atoms, max_bonds=max_bonds),
                 [max_bonds])

padding_values = (preprocessor.padding_values,
                  tf.constant(np.nan, dtype=tf.float64))

train_dataset = tf.data.TFRecordDataset('tfrecords/train.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=500)\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset('tfrecords/valid.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=500)\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


# Define keras model
n_atom = layers.Input(shape=[], dtype=tf.int64, name='n_atom')
n_bond = layers.Input(shape=[], dtype=tf.int64, name='n_bond')
bond_indices = layers.Input(shape=[None], dtype=tf.int64, name='bond_indices')
atom_class = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond_class = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

input_tensors = [n_atom, n_bond, bond_indices, atom_class, bond_class, connectivity]

# Initialize the atom states
atom_state = layers.Embedding(preprocessor.atom_classes, atom_features,
                              name='atom_embedding', mask_zero=True)(atom_class)

# Initialize the bond states
bond_state = layers.Embedding(preprocessor.bond_classes, atom_features,
                              name='bond_embedding', mask_zero=True)(bond_class)

# Initialize the bond states
bond_mean = layers.Embedding(preprocessor.bond_classes, 1,
                             name='bond_mean', mask_zero=True)(bond_class)

def message_block(original_atom_state, original_bond_state, connectivity, i):
    
    atom_state = layers.LayerNormalization()(original_atom_state)
    bond_state = layers.LayerNormalization()(original_bond_state)
    
    source_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 1])(connectivity)])
    target_atom = nfp.Gather()([atom_state, nfp.Slice(np.s_[:, :, 0])(connectivity)])

    # Edge update network
    new_bond_state = layers.Concatenate(name='concat_{}'.format(i))(
        [source_atom, target_atom, bond_state])
    new_bond_state = layers.Dense(
        2*atom_features, activation='relu')(new_bond_state)
    new_bond_state = layers.Dense(atom_features)(new_bond_state)

    bond_state = layers.Add()([original_bond_state, new_bond_state])

    # message function
    source_atom = layers.Dense(atom_features)(source_atom)    
    messages = layers.Multiply()([source_atom, bond_state])
    messages = nfp.Reduce(reduction='sum')(
        [messages, nfp.Slice(np.s_[:, :, 0])(connectivity), atom_state])
    
    # state transition function
    messages = layers.Dense(atom_features, activation='relu')(messages)
    messages = layers.Dense(atom_features)(messages)
    
    atom_state = layers.Add()([original_atom_state, messages])
    
    return atom_state, bond_state

for i in range(num_messages):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity, i)

bond_state = nfp.Reduce(reduction='mean')([bond_state, bond_indices, bond_state])
bond_mean = nfp.Reduce(reduction='mean')([bond_mean, bond_indices, bond_mean])

bond_state = layers.Dense(1)(bond_state)
bond_state = layers.Add()([bond_state, bond_mean])

model = tf.keras.Model(input_tensors, [bond_state])

learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-3, 1, 1E-5)
model.compile(loss=nfp.masked_mean_absolute_error, optimizer=tf.keras.optimizers.Adam(learning_rate))

model_name = '20200614_trained_model'

if not os.path.exists(model_name):
    os.makedirs(model_name)

filepath = model_name + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
csv_logger = tf.keras.callbacks.CSVLogger(model_name + '/log.csv')

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=500,
          callbacks=[checkpoint, csv_logger],
          verbose=2)
