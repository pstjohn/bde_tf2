import os

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers
import tensorflow_addons as tfa

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
    
    return parsed, {'bde': bde, 'bdfe': bdfe}

max_atoms = 32
max_bonds = 64
batch_size = 128
atom_features = 128
num_messages = 6

# Here, we have to add the prediction target padding onto the input padding
padded_shapes = (preprocessor.padded_shapes(max_atoms=max_atoms, max_bonds=max_bonds),
                 {'bde': [max_bonds,], 'bdfe': [max_bonds,]})

nan = tf.constant(np.nan, dtype=tf.float64)
padding_values = (preprocessor.padding_values, {'bde': nan, 'bdfe': nan})

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
bde_mean = layers.Embedding(preprocessor.bond_classes, 1,
                             name='bde_mean', mask_zero=True)(bond_class)

bdfe_mean = layers.Embedding(preprocessor.bond_classes, 1,
                             name='bdfe_mean', mask_zero=True)(bond_class)

for _ in range(num_messages):  # Do the message passing
    bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])    

bond_state = nfp.Reduce(reduction='mean')([bond_state, bond_indices, bond_state])
bde_mean = nfp.Reduce(reduction='mean')([bde_mean, bond_indices, bde_mean])
bdfe_mean = nfp.Reduce(reduction='mean')([bdfe_mean, bond_indices, bdfe_mean])

bde_pred = layers.Dense(1)(bond_state)
bde_pred = layers.Add(name='bde')([bde_pred, bde_mean])

bdfe_pred = layers.Dense(1)(bond_state)
bdfe_pred = layers.Add(name='bdfe')([bdfe_pred, bdfe_mean])

model = tf.keras.Model(input_tensors, [bde_pred, bdfe_pred])

learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-3, 1, 1E-5)
weight_decay  = tf.keras.optimizers.schedules.InverseTimeDecay(1E-4, 1, 1E-5)
optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
model.compile(loss=nfp.masked_mean_absolute_error, optimizer=optimizer)

model_name = '20201010_model_more_wd'

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
