import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()

mini_batch_size = 10
# print("# step 5.")
# net = Network([
#     FullyConnectedLayer(n_in=784, n_out=100),
#     SoftmaxLayer(n_in=100, n_out=10)],
#     mini_batch_size)


# print("# step 6.")
# net.SGD(
#     training_data,
#     60,
#     mini_batch_size,
#     0.1,
#     validation_data,
#     test_data
# )


# create CNN 

print("#step 7.")
net = Network([
    ConvPoolLayer(
        image_shape=(mini_batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2,2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)],
    mini_batch_size
)

print("# step 8.")
epochs = 70
print("epochs: " + str(epochs))
net.SGD(
    training_data = training_data, 
    epochs = epochs, 
    mini_batch_size = mini_batch_size, 
    eta = 0.1,
    validation_data = validation_data, 
    test_data = test_data
)

print("# step 9:")
print("Modify 7 by adding convolutional layers following the pooling with the following specs:")
print("40 convolutional layers")
print("5x5 filters")
print("Stride of 20")
print("Pool 2x2")

net = Network([
    ConvPoolLayer(
        image_shape=(mini_batch_size, 1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2,2)),
    ConvPoolLayer(
        image_shape=(mini_batch_size, 20, 12, 12),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2,2)),
    FullyConnectedLayer(n_in=20*12*12, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)],
    mini_batch_size
) 






