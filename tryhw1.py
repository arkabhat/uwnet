from uwnet import *

def conv_net():
    """
    Number of ops for a convolutional layer:
    - The number of output pixels will be
        (input height * input_width * input_channels) / stride
    - For each filter, there are size * size operations, so in total, there
        will be size * size * filters
    - Overall: ((input height * input_width * input_channels) / stride)
        * (size * size * filters)

    1st convolutional layer: ((32 * 32 * 3) / 1) * (3 * 3 * 8) = 221,184
    2nd convolutional layer: ((16 * 16 * 8) / 1) * (3 * 3 * 16) = 294,912
    3rd convolutional layer: ((8 * 8 * 16) / 1) * (3 * 3 * 32) = 294,912
    4th convolutional layer: ((4 * 4 * 32) / 1) * (3 * 3 * 64) = 294,912
    Total for convolutional layers = 1,105,920

    There is one connected layer as well: 256 * 10 = 2,560

    Total operations = 1,105,920 + 2,560 = 1,108,480 operations
    """

    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conn_net():
    """
    Number of operations:
    3072 * 324 + 324 * 256 + 256 * 64 + 64 * 64 + 64 * 10 = 1,099,392 operations
    """

    l = [   make_connected_layer(3072, 324),
            make_activation_layer(RELU),
            make_connected_layer(324, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conn_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# Convolutional net:
# training accuracy: %f 0.7101399898529053
# test accuracy:     %f 0.6617000102996826
#
# Fully connected net:
# training accuracy: %f 0.5583800077438354
# test accuracy:     %f 0.515500009059906
#
# The convolutional network exhibits ~15% higher train and test accuracies
# than the fully connected network. This would largely be due to the idea that
# fully connected networks will have more weights/parameters since the whole
# images are passed through the net, whereas with the convolutional net, the
# nput size (and overall number of params/weights) is reduced significantly
# which also makes the convolutional net less prone to overfitting by becmoing
# dependent on the shapes/patterns in the training set images.
