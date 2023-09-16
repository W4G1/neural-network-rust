# A Minimal Feed-Forward Neural Network made in Rust

This repository contains a minimal but fully functioning Feed-Forward Neural Network (FNN) implemented in the Rust programming language. The neural network uses batch gradient descent for backpropagation. It is single-threaded and uses CPU only for computations.

This implementation uses the mean squared error loss function and includes two types of activation functions: Sigmoid and Rectified Linear Unit (ReLU).

## Goal
The goal of this project is to get a better understanding of neural networks by creating one from scratch.

## Getting Started

To get a copy of this project up and running on your local machine, you will need [Rust](https://www.rust-lang.org/tools/install).

Clone this repository:
```bash
git clone https://github.com/w4g1/neural-network-rust.git
```

Go into the repository:
```bash
cd neural-network-rust
```

<!-- Compile the project:
```bash
cargo build
``` -->

Run XOR example:
```bash
cargo run --example xor --release

Epoch: 250000 Error: 0.004982472275727541
Epoch: 500000 Error: 0.0022680697570409874
Epoch: 750000 Error: 0.0014475361058490137
Epoch: 1000000 Error: 0.0010574201380490365
Epoch: 1250000 Error: 0.0008307775961258309
Training completed in 844.0897ms
0 XOR 0 = 0.0364493729489178
0 XOR 1 = 0.9629761743234105
1 XOR 0 = 0.9597184054455132
1 XOR 1 = 0.040673502104589074
```

## Usage

<!-- Optionally define a random generator that is used to seed the neuron's initial bias and connection weights:

```rust
lazy_static! {
    static ref RNG: Mutex<Pcg64Mcg> = Mutex::new(Pcg64Mcg::new(890749));
}

fn random() -> f64 {
    let mut rng = RNG.lock().unwrap();
    rng.gen()
}
``` -->

This implementation abstracts loss and activation functions into separate traits, providing flexibility:

```rust
trait LossFunction {
    fn compute(&self, target: f64, output: f64) -> f64;
    fn derivative(&self, target: f64, output: f64) -> f64;
}

trait ActivationFunction {
    fn compute(&self, input: f64) -> f64;
    fn derivative(&self, output: f64) -> f64;
}
```

A `Neuron` struct supports arbitrary number of inputs and keeps track of its output, error, and the activation function it should use. Various other structural elements such as `Connection`, `Layer`, `NeuralNetworkConfig`, `TrainConfig`, and `NeuralNetwork` are defined to accommodate a fully working feed forward neural network.

To build a simple XOR logic function learning neural network, use the following:

```rust
let mut network = NeuralNetwork::new(
    vec![
        Layer::new(2, &Activation::Sigmoid(Sigmoid)),
        Layer::new(5, &Activation::Sigmoid(Sigmoid)),
        Layer::new(1, &Activation::Sigmoid(Sigmoid)),
    ],
    NeuralNetworkConfig {
        loss_function: &Loss::MeanSquaredError(MeanSquaredError),
    },
);
```

To train the network, use the `train` method on the instance and pass in training data and Configuration for the training.

```rust
let config = TrainConfig {
    learning_rate: 0.5,
    steps: -1,
    epochs: -1,
    error_threshold: 0.001,
    eval_frequency: 100_000,
};

network.train(&dataset, config);
```

## License
This project is open-source software licensed under the MIT license.

## Contributions
Pull requests are always welcome to improve this repository. Please feel free to fork this package and contribute by submitting a pull request to enhance the functionalities.