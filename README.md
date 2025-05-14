# krykhitgrad: A Lightweight Automatic Differentiation Engine and Neural Network Library

## Prerequisites

GCC, CMAKE, Eigen, Indicators

## Installation
```
git clone https://github.com/bulkobubulko/krykhitgrad
cd krykhitgrad
```

## Compilation
```
mkdir build && cd build
cmake ..
```

## Available Executables
```
# build and run the main executable
make krykhitgrad
./krykhitgrad

# build and run the MNIST example
make mnist_example
./mnist_example

# build and run the autograd tests
make test_autograd
./test_autograd
```

To test on MNIST dataset, please download it (`train-images.idx3-ubyte` and `train-labels.idx1-ubyte`) and move to `/data/mnist/` directory.

## References
- pytorch, https://github.com/pytorch/pytorch
- micrograd, https://github.com/karpathy/micrograd
- tinygrad, https://github.com/tinygrad/tinygrad
- Mathematics for Machine Learning, https://mml-book.github.io/book/mml-book.pdf
- Automatic Differentiation in Machine Learning: a Survey, https://www.jmlr.org/papers/volume18/17-468/17-468.pdf
- The Simple Essence of Automatic Differentiation, https://arxiv.org/pdf/1804.00746