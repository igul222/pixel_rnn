import lib
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=234)

def Linear(
        name,
        input_dims,
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=False
        ):
    """
    Compute a linear transform of one or more inputs, optionally with a bias.

    input_dims: list of ints, or int (if single input); the dimensionality of
                the input(s).
    output_dim: the dimensionality of the output.
    biases:     whether or not to include a bias term.
    inputs:     a theano variable, or list of variables (if multiple inputs);
                the inputs to which to apply the transform.
    initialization: one of None, `lecun`, `he`, `orthogonal`
    """

    if not isinstance(input_dims, list):
        input_dims = [input_dims]
        inputs = [inputs]

    terms = []

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    for i, (inp, inp_dim) in enumerate(zip(inputs, input_dims)):
        if initialization == 'lecun' or (initialization == None and inp_dim != output_dim):
            weight_values = uniform(numpy.sqrt(1. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'he':
            weight_values = uniform(numpy.sqrt(2. / inp_dim), (inp_dim, output_dim))
        elif initialization == 'orthogonal' or (initialization == None and inp_dim == output_dim):
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], numpy.prod(shape[1:]))
                # TODO: why normal and not uniform?
                a = numpy.random.normal(0.0, 1.0, flat_shape)
                u, _, v = numpy.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype(theano.config.floatX)
            weight_values = sample((inp_dim, output_dim))
        else:
            raise Exception("Invalid initialization!")

        weight = lib.param(
            name + '.W'+str(i),
            weight_values
        )

        if weightnorm:
            norm_values = numpy.linalg.norm(weight_values, axis=0)
            norms = lib.param(
                name + '.g'+str(i),
                norm_values
            )

            normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
            terms.append(T.dot(inp, normed_weight))
        else:        
            terms.append(T.dot(inp, weight))

    if biases:
        terms.append(lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        ))

    out = reduce(lambda a,b: a+b, terms)
    out.name = name + '.output'
    return out