"""
Pixel RNN on MNIST
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.register_crash_notifier()
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import theano
import theano.tensor as T
import lib
import lasagne
import scipy.misc

import time
import functools
import itertools

MODEL = 'pixel_rnn' # either pixel_rnn or pixel_cnn

# Hyperparams
BATCH_SIZE = 100
DIM = 64 # Model dimensionality.
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
N_CHANNELS = 1
WIDTH = 28
HEIGHT = 28

# Other constants
TEST_BATCH_SIZE = 100 # batch size to use when evaluating on dev/test sets. This should be the max that can fit into GPU memory.
EVAL_DEV_COST = True # whether to evaluate dev cost during training
GEN_SAMPLES = True # whether to generate samples during training (generating samples takes WIDTH*HEIGHT*N_CHANNELS full passes through the net)
TRAIN_MODE = 'iters' # 'iters' to use PRINT_ITERS and STOP_ITERS, 'time' to use PRINT_TIME and STOP_TIME
PRINT_ITERS = 5000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 100000 # Stop after this many iterations
PRINT_TIME = 60*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*2 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)

lib.utils.print_model_settings(locals().copy())

def relu(x):
    # Using T.nnet.relu gives me NaNs. No idea why.
    return T.switch(x > lib.floatX(0), x, lib.floatX(0))

def Conv2D(name, input_dim, output_dim, filter_size, inputs, mask_type=None, he_init=False):
    """
    inputs.shape: (batch size, height, width, input_dim)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, height, width, output_dim)
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters_init = uniform(
        1./numpy.sqrt(input_dim * filter_size * filter_size),
        # output dim, input dim, height, width
        (output_dim, input_dim, filter_size, filter_size)
    )

    if he_init:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    if mask_type is not None:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    filters = lib.param(
        name+'.Filters',
        filters_init
    )

    if mask_type is not None:
        mask = numpy.ones(
            (output_dim, input_dim, filter_size, filter_size), 
            dtype=theano.config.floatX
        )
        center = filter_size//2
        for i in xrange(filter_size):
            for j in xrange(filter_size):
                    if (j > center) or (j==center and i > center):
                        mask[:, :, j, i] = 0.
        for i in xrange(N_CHANNELS):
            for j in xrange(N_CHANNELS):
                if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                    mask[
                        j::N_CHANNELS,
                        i::N_CHANNELS,
                        center,
                        center
                    ] = 0.

        filters = filters * mask

    # conv2d takes inputs as (batch size, input channels, height, width)
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='half', filter_flip=False)

    biases = lib.param(
        name+'.Biases',
        numpy.zeros(output_dim, dtype=theano.config.floatX)
    )
    result = result + biases[None, :, None, None]

    return result.dimshuffle(0, 2, 3, 1)

def Conv1D(name, input_dim, output_dim, filter_size, inputs, apply_biases=True):
    """
    inputs.shape: (batch size, height, input_dim)
    output.shape: (batch size, height, output_dim)
    * performs valid convs
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters = lib.param(
        name+'.Filters',
        uniform(
            1./numpy.sqrt(input_dim * filter_size),
            # output dim, input dim, height, width
            (output_dim, input_dim, filter_size, 1)
        )
    )

    # conv2d takes inputs as (batch size, input channels, height[?], width[?])
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1, inputs.shape[2]))
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='valid', filter_flip=False)

    if apply_biases:
        biases = lib.param(
            name+'.Biases',
            numpy.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases[None, :, None, None]

    result = result.dimshuffle(0, 2, 3, 1)
    return result.reshape((result.shape[0], result.shape[1], result.shape[3]))

def Skew(inputs):
    """
    input.shape: (batch size, HEIGHT, WIDTH, dim)
    """
    buffer = T.zeros(
        (inputs.shape[0], inputs.shape[1], 2*inputs.shape[2] - 1, inputs.shape[3]),
        theano.config.floatX
    )

    for i in xrange(HEIGHT):
        buffer = T.inc_subtensor(buffer[:, i, i:i+WIDTH, :], inputs[:,i,:,:])

    return buffer

def Unskew(padded):
    """
    input.shape: (batch size, HEIGHT, 2*WIDTH - 1, dim)
    """
    return T.stack([padded[:, i, i:i+WIDTH, :] for i in xrange(HEIGHT)], axis=1)

def DiagonalLSTM(name, input_dim, inputs):
    """
    inputs.shape: (batch size, height, width, input_dim)
    outputs.shape: (batch size, height, width, DIM)
    """
    inputs = Skew(inputs)

    input_to_state = Conv2D(name+'.InputToState', input_dim, 4*DIM, 1, inputs, mask_type='b')

    batch_size = inputs.shape[0]

    c0_unbatched = lib.param(
        name + '.c0',
        numpy.zeros((HEIGHT, DIM), dtype=theano.config.floatX)
    )
    c0 = T.alloc(c0_unbatched, batch_size, HEIGHT, DIM)

    h0_unbatched = lib.param(
        name + '.h0',
        numpy.zeros((HEIGHT, DIM), dtype=theano.config.floatX)
    )
    h0 = T.alloc(h0_unbatched, batch_size, HEIGHT, DIM)

    def step_fn(current_input_to_state, prev_c, prev_h):
        # all args have shape (batch size, height, DIM)

        # TODO consider learning this padding
        prev_h = T.concatenate([
            T.zeros((batch_size, 1, DIM), theano.config.floatX), 
            prev_h
        ], axis=1)
        state_to_state = Conv1D(name+'.StateToState', DIM, 4*DIM, 2, prev_h, apply_biases=False)

        gates = current_input_to_state + state_to_state

        o_f_i = T.nnet.sigmoid(gates[:,:,:3*DIM])
        o = o_f_i[:,:,0*DIM:1*DIM]
        f = o_f_i[:,:,1*DIM:2*DIM]
        i = o_f_i[:,:,2*DIM:3*DIM]
        g = T.tanh(gates[:,:,3*DIM:4*DIM])

        new_c = (f * prev_c) + (i * g)
        new_h = o * T.tanh(new_c)

        return (new_c, new_h)

    outputs, _ = theano.scan(
        step_fn,
        sequences=input_to_state.dimshuffle(2,0,1,3),
        outputs_info=[c0, h0]
    )
    all_cs = outputs[0].dimshuffle(1,2,0,3)
    all_hs = outputs[1].dimshuffle(1,2,0,3)

    return Unskew(all_hs)

def DiagonalBiLSTM(name, input_dim, inputs):
    """
    inputs.shape: (batch size, height, width, input_dim)
    inputs.shape: (batch size, height, width, DIM)
    """
    forward = DiagonalLSTM(name+'.Forward', input_dim, inputs)
    backward = DiagonalLSTM(name+'.Backward', input_dim, inputs[:,:,::-1,:])[:,:,::-1,:]
    batch_size = inputs.shape[0]
    backward = T.concatenate([
        T.zeros([batch_size, 1, WIDTH, DIM], dtype=theano.config.floatX),
        backward[:, :-1, :, :]
    ], axis=1)

    return forward + backward

# inputs.shape: (batch size, height, width, channels)
inputs = T.tensor4('inputs')

output = Conv2D('InputConv', N_CHANNELS, DIM, 7, inputs, mask_type='a')

if MODEL=='pixel_rnn':

    output = DiagonalBiLSTM('LSTM1', DIM, output)
    output = DiagonalBiLSTM('LSTM2', DIM, output)

elif MODEL=='pixel_cnn':
    # The paper doesn't specify how many convs to use, so I picked 4 pretty
    # arbitrarily.
    for i in xrange(4):
        output = Conv2D('PixelCNNConv'+str(i), DIM, DIM, 3, output, mask_type='b', he_init=True)
        output = relu(output)

output = Conv2D('OutputConv1', DIM, DIM, 1, output, mask_type='b', he_init=True)
output = relu(output)

output = Conv2D('OutputConv2', DIM, DIM, 1, output, mask_type='b', he_init=True)
output = relu(output)

# TODO: for color images, implement a 256-way softmax for each RGB channel here
output = Conv2D('OutputConv3', DIM, 1, 1, output, mask_type='b')
output = T.nnet.sigmoid(output)

cost = T.mean(T.nnet.binary_crossentropy(output, inputs))

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.utils.print_params_info(params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=1e-3)

train_fn = theano.function(
    [inputs],
    cost,
    updates=updates,
    on_unused_input='warn'
)

eval_fn = theano.function(
    [inputs],
    cost,
    on_unused_input='warn'
)

sample_fn = theano.function(
    [inputs],
    output,
    on_unused_input='warn'
)

train_data, dev_data, test_data = lib.mnist.load(BATCH_SIZE, TEST_BATCH_SIZE)

def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (numpy.random.uniform(size=images.shape) < images).astype('float32')

def generate_and_save_samples(tag):

    def save_images(images, filename):
        """
        images.shape: (batch, height, width, channels)
        """
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))

        scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save('{}_{}.jpg'.format(filename, tag))

    samples = numpy.zeros((100, HEIGHT, WIDTH, 1), dtype='float32')

    for i in xrange(HEIGHT):
        for j in xrange(WIDTH):
            for k in xrange(N_CHANNELS):
                next_sample = binarize(sample_fn(samples))
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, 'samples')

print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
for epoch in itertools.count():

    costs = []
    data_feeder = train_data()

    for images, targets in data_feeder:

        images = binarize(images.reshape((BATCH_SIZE, HEIGHT, WIDTH, 1)))

        start_time = time.time()
        cost = train_fn(images)
        total_time += time.time() - start_time
        total_iters += 1

        costs.append(cost)

        if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
            (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME):

            dev_costs = []
            if EVAL_DEV_COST:
                for images, targets in dev_data():
                    images = images.reshape((-1, HEIGHT, WIDTH, 1))
                    binarized = binarize(images)
                    dev_cost = eval_fn(binarized)
                    dev_costs.append(dev_cost)
            else:
                dev_costs.append(0.)

            print "epoch:{}\ttotal iters:{}\ttrain cost:{}\tdev cost:{}\ttotal time:{}\ttime per iter:{}".format(
                epoch,
                total_iters,
                numpy.mean(costs),
                numpy.mean(dev_costs),
                total_time,
                total_time / total_iters
            )

            tag = "iters{}_time{}".format(total_iters, total_time)
            if GEN_SAMPLES:
                generate_and_save_samples(tag)
            lib.save_params('params_{}.pkl'.format(tag))

            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
            (TRAIN_MODE=='time' and total_time >= STOP_TIME):

            print "Done!"

            try: # This only matters on Ishaan's computer
                import experiment_tools
                experiment_tools.send_sms("done!")
            except ImportError:
                pass

            sys.exit()