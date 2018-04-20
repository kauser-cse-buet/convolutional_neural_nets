
# coding: utf-8

# In[92]:

# import notMNIST_dataset
from six.moves import cPickle as pickle
import theano
import numpy as np
import theano.tensor as T
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU


# In[93]:

notMNIST_pickle_name = "notMNIST.pickle"


# In[94]:

def load_pickle_file(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        loaded_file = pickle.load(f)
        
    return loaded_file

def get_notMNIST_dataset_MNIST_format(notMNIST_dataset_list):
    train_dataset = notMNIST_dataset_list['train_dataset']
    train_labels = notMNIST_dataset_list['train_labels']
    valid_dataset = notMNIST_dataset_list['valid_dataset']
    valid_labels = notMNIST_dataset_list['valid_labels']
    test_dataset = notMNIST_dataset_list['test_dataset']
    test_labels = notMNIST_dataset_list['test_labels']
    
    train_dataset_flat = []
    test_dataset_flat = []
    valid_dataset_flat = []
    
    for i in range(len(train_dataset)):
        train_dataset_flat.append(train_dataset[i].flatten())
        
    for i in range(len(test_dataset)):
        test_dataset_flat.append(test_dataset[i].flatten())

    for i in range(len(valid_dataset)):
        valid_dataset_flat.append(valid_dataset[i].flatten())
    
    
    training_data = (train_dataset_flat, train_labels)
    test_data = (test_dataset_flat, test_labels)
    validation_data = (valid_dataset_flat, valid_labels)
    
    return training_data, validation_data, test_data

def load_data_shared(pickle_filename):
    notMNIST_dataset_list = load_pickle_file(pickle_filename = notMNIST_pickle_name)
    training_data, validation_data, test_data = get_notMNIST_dataset_MNIST_format(notMNIST_dataset_list)
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


# In[95]:

training_data, validation_data, test_data = load_data_shared(pickle_filename = notMNIST_pickle_name)


# In[96]:

mini_batch_size = 10


# In[ ]:

net = Network([
    ConvPoolLayer(
        image_shape=(mini_batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2,2),
        activation_fn=ReLU
    ),
    ConvPoolLayer(
        image_shape=(mini_batch_size, 20, 12, 12),
        filter_shape=(40, 20, 5, 5),
        poolsize=(2,2),
        activation_fn=ReLU
    ),
    FullyConnectedLayer(
        n_in=40*4*4, 
        n_out=1000,
        activation_fn=ReLU,
        p_dropout=0.5
    ),
    FullyConnectedLayer(
        n_in=1000, 
        n_out=1000,
        activation_fn=ReLU,
        p_dropout=0.5
    ),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
    mini_batch_size
) 

net.SGD(
    training_data, 
    60, 
    mini_batch_size, 
    0.03,
    validation_data, 
    test_data, 
    lmbda=0.1
)

