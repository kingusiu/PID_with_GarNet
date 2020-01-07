import numpy as np
import os
from scipy import stats
import keras
from keras import backend as K
import tensorflow as tf
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import caloGraphNN_keras as graphNN
from dataReader import *
from plotting import *
import setGPU

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
tf.logging.set_verbosity(tf.logging.WARN)

save_model = True
load_model = False

class ParticleID( Enum ):
    GAMMA = 0
    MUON = 1
    ELECTRON = 2
    PION = 3


def make_input_target_dataset_for_particle( particle_events, num_2d_cluster, num_features, class_value ):
    particle_events_padded = get_padded_dataset(particle_events, num_2d_cluster, num_features)
    particle_target = np.full( len( particle_events_padded ), class_value )
    return [particle_events_padded, particle_target]


########        prepare data        ########

print("> reading datasets")

datadir = "/afs/cern.ch/user/k/kiwoznia/public/data/hgcal/"
#datadir = "../../Data"
gammas_events = read_dataset_by_event( os.path.join( datadir, "gamma.h5" ) )
muons_events = read_dataset_by_event( os.path.join( datadir, "muon.h5" ) )
electrons_events = read_dataset_by_event( os.path.join( datadir, "electron.h5" ) )
pions_events = read_dataset_by_event( os.path.join( datadir, "pion_c.h5" ) )

print('> read #events: ', len(gammas_events), ' gammas, ', len(muons_events), ' muons, ', len(electrons_events), ' electrons, ', len(pions_events), ' pions')

print("> calculating number of 2d layer clusters")

max_num_2dcluster = find_max_number_of_2d_layer_clusters( [gammas_events, muons_events, electrons_events, pions_events] ) # get maximum number of 2d clusters of all events
num_features = get_number_of_features( )

print("> creating padded input datasets")

gammas_input, gammas_target = make_input_target_dataset_for_particle( gammas_events, max_num_2dcluster, num_features, ParticleID.GAMMA.value )
muons_input, muons_target = make_input_target_dataset_for_particle( muons_events, max_num_2dcluster, num_features, ParticleID.MUON.value )
electrons_input, electrons_target = make_input_target_dataset_for_particle( electrons_events, max_num_2dcluster, num_features, ParticleID.ELECTRON.value )
pions_input, pions_target = make_input_target_dataset_for_particle( pions_events, max_num_2dcluster, num_features, ParticleID.PION.value )

all_data = np.vstack( [ gammas_input, muons_input, electrons_input, pions_input ] )
y = np.concatenate( [ gammas_target, muons_target, electrons_target, pions_target ] )
y = keras.utils.to_categorical( y )

print('all data shape ', all_data.shape)
print( "y shape ", y.shape )

X_train, X_test, y_train, y_test = train_test_split( all_data, y )

if load_model :

    model = load_model('models/model.h5')

else:

    n_output_features = 15
    n_aggregators = 5
    batch_size = 16
    epochs = 100

    inputs = keras.Input( shape=( max_num_2dcluster, num_features ), name='input' )
    gar1 = graphNN.GarNet( n_aggregators,  5, n_output_features, name='gar_1' )( inputs ) # 3 aggregators, n_output_features transformed features;
    n_output_features = int( n_output_features * 1.5 )
    gar2 = graphNN.GarNet( n_aggregators, 10, n_output_features, name='gar_2' )( gar1 ) # 3 aggregators, n_output_features transformed features;
    n_output_features = int( n_output_features * 1.5 )
    gar3 = graphNN.GarNet( n_aggregators, 15, n_output_features, name='gar_3' )( gar2 ) # 3 aggregators, 22 transformed features;
    n_output_features = int( n_output_features * 1.5 )
    gar4 = graphNN.GarNet( n_aggregators, 15, n_output_features, name='gar_4' )( gar3 )
    n_output_features = int( n_output_features * 1.5 )
    gar5 = graphNN.GarNet( n_aggregators, 20, n_output_features, name='gar_5' )( gar4 )
    n_output_features = int( n_output_features * 1.5 )
    gar6 = graphNN.GarNet( n_aggregators, 25, n_output_features, name='gar_6' )( gar5 )
    n_output_features = int( n_output_features * 1.5 )
    gar7 = graphNN.GarNet( n_aggregators, 30, n_output_features, name='gar_7' )( gar6 )
    #flat = keras.layers.Flatten()( gar3 ) # flatten before last layer s.t. final output is B x last_out * V
    averaged = keras.layers.Lambda( lambda x: K.mean( x, axis = -2 ) )( gar7 )
    outputs = keras.layers.Dense( len(ParticleID), activation='softmax', name='output' )( averaged )

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("> training model")
    
    history = model.fit( X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2 )

    # list all data in history
    print(history.history.keys())
    # plot training loss
    plot_semilogy([history.history['loss'], history.history['val_loss']], 'loss during training', ['epoch', 'log loss'],['training', 'validation'])

model.summary( )

if save_model:
    model.save('models/model.h5')

print("> predicting data")

predicted_particleID = model.predict( X_test )

y_true = y_test.argmax(axis=1)
y_pred = predicted_particleID.argmax(axis=1)

plot_confusion_matrix( y_test.argmax(axis=1), predicted_particleID.argmax(axis=1), np.array([p.name for p in ParticleID]), normalize=True )




