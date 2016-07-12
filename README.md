# Pixel RNN

Theano implementation of the Diagonal BiLSTM and Pixel CNN models from Pixel Recurrent Neural Networks (van der Oord et al.) on binarized MNIST.

Not yet implemented:
- Residual connections
- 256-way softmax output

The Diagonal BiLSTM gets a dev-set NLL of 82.12 after 100K iterations (at which point it still hasn't fully converged).

Samples from the Diagonal BiLSTM after 100K iterations:

![Diagonal BiLSTM samples](https://raw.githubusercontent.com/igul222/pixel_rnn/master/samples.jpg)

Ground-truth binarized MNIST samples:

![Ground-truth samples](https://raw.githubusercontent.com/igul222/pixel_rnn/master/groundtruth.jpg)