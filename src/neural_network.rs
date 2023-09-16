extern crate lazy_static;
extern crate rand;
use lazy_static::lazy_static;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::sync::Mutex;

lazy_static! {
    static ref RNG: Mutex<Pcg64Mcg> = Mutex::new(Pcg64Mcg::new(890749));
}

fn random() -> f64 {
    let mut rng = RNG.lock().unwrap();
    rng.gen()
}

trait LossFunction {
    fn compute(&self, target: f64, output: f64) -> f64;
    fn derivative(&self, target: f64, output: f64) -> f64;
}

trait ActivationFunction {
    fn compute(&self, input: f64) -> f64;
    fn derivative(&self, output: f64) -> f64;
}

pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn compute(&self, target: f64, output: f64) -> f64 {
        0.5 * (target - output).powi(2)
    }

    fn derivative(&self, target: f64, output: f64) -> f64 {
        target - output
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn compute(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn derivative(&self, output: f64) -> f64 {
        output * (1.0 - output)
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn compute(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn derivative(&self, output: f64) -> f64 {
        if output > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub enum Activation {
    Sigmoid(Sigmoid),
    ReLU(ReLU),
}

impl ActivationFunction for Activation {
    fn compute(&self, input: f64) -> f64 {
        match self {
            Activation::Sigmoid(sigmoid) => sigmoid.compute(input),
            Activation::ReLU(relu) => relu.compute(input),
        }
    }

    fn derivative(&self, output: f64) -> f64 {
        match self {
            Activation::Sigmoid(sigmoid) => sigmoid.derivative(output),
            Activation::ReLU(relu) => relu.derivative(output),
        }
    }
}

pub enum Loss {
    MeanSquaredError(MeanSquaredError),
}

impl LossFunction for Loss {
    fn compute(&self, target: f64, input: f64) -> f64 {
        match self {
            Loss::MeanSquaredError(mean_squared_error) => mean_squared_error.compute(target, input),
        }
    }

    fn derivative(&self, target: f64, output: f64) -> f64 {
        match self {
            Loss::MeanSquaredError(mean_squared_error) => {
                mean_squared_error.derivative(target, output)
            }
        }
    }
}

struct Neuron<'a> {
    activation_function: &'a Activation,
    connections: Vec<Connection>,
    bias: f64,
    output: f64,
    error: f64,
}

impl<'a> Neuron<'a> {
    fn new(activation_function: &'a Activation) -> Self {
        Neuron {
            activation_function,
            connections: Vec::new(),
            bias: random(),
            output: 0.0,
            error: 0.0,
        }
    }

    fn activate(neuron: &mut Neuron, prev_layer_output: &[f64]) {
        let sum = neuron.connections.iter().fold(neuron.bias, |accum, conn| {
            accum + conn.weight * prev_layer_output[conn.prev_layer_neuron_index]
        });

        neuron.output = neuron.activation_function.compute(sum);
    }

    fn derivative(&mut self) -> f64 {
        self.activation_function.derivative(self.output)
    }
}

struct Connection {
    weight: f64,
    prev_layer_neuron_index: usize,
}
pub struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    pub fn new(size: usize, activation_function: &'a Activation) -> Self {
        let mut neurons: Vec<Neuron> = Vec::new();

        for _ in 0..size {
            neurons.push(Neuron::new(&activation_function));
        }

        Layer { neurons }
    }
}

pub struct NeuralNetworkConfig<'a> {
    pub loss_function: &'a Loss,
}

pub struct TrainConfig {
    pub learning_rate: f64,
    pub steps: i32,
    pub epochs: i32,
    pub error_threshold: f64,
    pub eval_frequency: i32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            learning_rate: 0.1,
            steps: -1,
            epochs: -1,
            error_threshold: 0.01,
            eval_frequency: 10_000,
        }
    }
}

pub struct NeuralNetwork<'a> {
    layers: Vec<Layer<'a>>,
    loss_function: &'a Loss,
}

impl<'a> NeuralNetwork<'a> {
    pub fn new(layers: Vec<Layer<'a>>, config: NeuralNetworkConfig<'a>) -> Self {
        let mut network = NeuralNetwork {
            layers,
            loss_function: config.loss_function,
        };

        let layers_len = network.layers.len();
        for i in 1..layers_len {
            let prev_neurons_len = network.layers[i - 1].neurons.len();
            for neuron in &mut network.layers[i].neurons {
                for j in 0..prev_neurons_len {
                    neuron.connections.push(Connection {
                        weight: random() * 2.0 - 1.0,
                        prev_layer_neuron_index: j,
                    });
                }
            }
        }
        network
    }

    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], config: TrainConfig) {
        let mut epoch: i32 = 0;
        let mut step: i32 = 0;
        let mut error = f64::INFINITY;
        let mut new_error;
        let mut will_eval = false;

        // Validate dataset
        for &(ref inputs, ref outputs) in dataset {
            if inputs.len() != self.layers[0].neurons.len() {
                panic!(
                    "Input size of {} does not match the input layer size of {}.",
                    inputs.len(),
                    self.layers[0].neurons.len()
                );
            }

            if outputs.len() != self.layers.last().unwrap().neurons.len() {
                panic!(
                    "Output size of {} does not match the output layer size of {}.",
                    outputs.len(),
                    self.layers.last().unwrap().neurons.len()
                );
            }
        }

        let TrainConfig {
            learning_rate,
            steps,
            epochs,
            error_threshold,
            eval_frequency,
        } = config;

        loop {
            epoch += 1;
            new_error = 0.0;

            for chunk in dataset.chunks(4) {
                for &(ref inputs, ref targets) in chunk {
                    step += 1;
                    will_eval = step % eval_frequency == 0;

                    self.activate(inputs);
                    self.backpropagate(targets, learning_rate);

                    if will_eval {
                        let output = self.get_output();
                        new_error += self.loss_function.compute(targets[0], output[0]);
                    }

                    if steps > 0 && step >= steps {
                        break;
                    }
                }
            }

            if will_eval {
                error = new_error;

                println!("Epoch: {} Error: {}", epoch, error);

                if error < error_threshold {
                    break;
                }
            }

            if epochs > 0 && epoch >= epochs {
                break;
            }
        }
    }

    pub fn predict(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.activate(inputs);
        self.get_output()
    }

    fn activate(&mut self, inputs: &Vec<f64>) {
        // Apply inputs to the neurons of the first layer
        for (i, neuron) in self.layers[0].neurons.iter_mut().enumerate() {
            neuron.output = inputs[i];
        }

        // Activate each layer after the first one
        for i in 1..self.layers.len() {
            let prev_layer_output = self.layers[i - 1]
                .neurons
                .iter()
                .map(|n| n.output)
                .collect::<Vec<_>>();
            let layer = &mut self.layers[i];

            for neuron_index in 0..layer.neurons.len() {
                Neuron::activate(&mut layer.neurons[neuron_index], &prev_layer_output);
            }
        }
    }

    fn get_output(&mut self) -> Vec<f64> {
        let output_layer = self.layers.last().unwrap();
        return output_layer.neurons.iter().map(|n| n.output).collect();
    }

    fn backpropagate(&mut self, targets: &Vec<f64>, learning_rate: f64) {
        // Calculate the error of the output layer neurons
        let output_layer = self.layers.last_mut().unwrap();

        for (i, neuron) in output_layer.neurons.iter_mut().enumerate() {
            let output = neuron.output;
            let target = targets[i];
            let loss_derivative = self.loss_function.derivative(target, output);
            neuron.error = neuron.derivative() * loss_derivative;
        }

        // Calculate the errors of the hidden layer neurons
        for i in (1..self.layers.len() - 1).rev() {
            let prev_layer_errors = self.layers[i - 1]
                .neurons
                .iter()
                .map(|n| n.error)
                .collect::<Vec<_>>();

            let layer = &mut self.layers[i];
            for neuron in &mut layer.neurons {
                let mut sum = 0.0;

                for connection in &neuron.connections {
                    let origin_error = prev_layer_errors[connection.prev_layer_neuron_index];
                    sum += connection.weight * origin_error;
                }

                neuron.error = neuron.derivative() * sum;
            }
        }

        // Update all weights and biases
        for i in 1..self.layers.len() {
            let prev_layer_outputs = self.layers[i - 1]
                .neurons
                .iter()
                .map(|n| n.output)
                .collect::<Vec<_>>();

            let layer = &mut self.layers[i];
            for neuron in &mut layer.neurons {
                for connection in &mut neuron.connections {
                    let prev_layer_neuron_output =
                        prev_layer_outputs[connection.prev_layer_neuron_index];
                    connection.weight += learning_rate * neuron.error * prev_layer_neuron_output;
                }

                neuron.bias += learning_rate * neuron.error;
            }
        }
    }
}
