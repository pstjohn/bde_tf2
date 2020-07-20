import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs

                
def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        nfp.get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(nfp.get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()

preprocessor = nfp.SmilesPreprocessor(
    atom_features=atom_featurizer, bond_features=bond_featurizer)
    

if __name__ == '__main__':
            
    bde = pd.read_csv('20200614_rdf_new_elements.csv.gz', index_col=0)

    train = bde[bde.set == 'train'].molecule.unique()
    valid = bde[bde.set == 'valid'].molecule.unique()
    
    def inputs_generator(smiles_iterator, train=True):
        for smiles in tqdm(smiles_iterator):
            input_dict = preprocessor.construct_feature_matrices(smiles, train=train)
            bde_df = bde[bde.molecule == smiles]
            input_dict['bde'] = bde_df.set_index('bond_index').bde.reindex(
                np.arange(input_dict['n_bond'])).values
            input_dict['bdfe'] = bde_df.set_index('bond_index').bdfe.reindex(
                np.arange(input_dict['n_bond'])).values
        
            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()
    
    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(train, train=True),
        output_types=tf.string, output_shapes=())
    
    filename = 'tfrecords/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)
    
    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    
    preprocessor.to_json('tfrecords/preprocessor.json')
