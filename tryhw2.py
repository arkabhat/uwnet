from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_net_with_batchnorm():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
# Original hyperparameters
"""
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005
"""

# Learning rate experiment hyperparameters
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005

#m = conv_net()
m = conv_net_with_batchnorm()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
#
# Convnet without batch normalization (original hyperparameters):
# training accuracy: %f 0.39772000908851624
# test accuracy:     %f 0.39989998936653137
#
# Convnet with batch normalization (original hyperparameters):
# training accuracy: %f 0.5558800101280212
# test accuracy:     %f 0.5419999957084656
#
# -----------------------------------------------------------------------------
#
# Batch normalization learning rate experiments:
# Learning rate: 0.1
# training accuracy: %f 0.535860002040863
# test accuracy:     %f 0.525600016117096
#
# Learning rate: 0.09
# training accuracy: %f 0.5464000105857849
# test accuracy:     %f 0.5297999978065491
#
# Learning rate: 0.075
# training accuracy: %f 0.5397599935531616
# test accuracy:     %f 0.5299000144004822
#
# Learning rate: 0.05
# training accuracy: %f 0.5394399762153625
# test accuracy:     %f 0.5324000120162964
#
# Learning rate: 0.0375
# training accuracy: %f 0.5496199727058411
# test accuracy:     %f 0.5411999821662903
#
# Learning rate: 0.025
# training accuracy: %f 0.5454199910163879
# test accuracy:     %f 0.5365999937057495
#
# Learning rate: 0.01
# training accuracy: %f 0.5558800101280212
# test accuracy:     %f 0.5419999957084656

# From the above we can see that batch normalization improved the test and training accuracy of 
# convnet overall significantly. The normalization of elements after convolutional layers likely work 
# to help better modulate gradient explosions and help better generalize the input data, resulting in an
# increase in accuracy. Further, we are able to push our learning rates to very small values without 
# greatly impacting our model's accuracy and convergence, likely due to the fact that no particular neurons 
# are being drastically impacted by outliers.
# The model has less fluctuation between epochs in terms of accuracy as well, and seems to converge more easily 
# and more smoothly, due to the normalization after layers. 