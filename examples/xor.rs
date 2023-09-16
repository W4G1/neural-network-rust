use std::time::Instant;

#[path = "../src/neural_network.rs"]
mod neural_network;

use crate::neural_network::{
    Activation, Layer, Loss, MeanSquaredError, NeuralNetwork, NeuralNetworkConfig, Sigmoid,
    TrainConfig,
};

fn main() {
    let training_start = Instant::now();

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

    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let config = TrainConfig {
        learning_rate: 0.5,
        steps: -1,
        epochs: -1,
        error_threshold: 0.001,
        eval_frequency: 100_000,
    };

    network.train(&dataset, config);

    let training_end = Instant::now(); // Record the end time

    // Calculate and print the duration
    let duration = training_end - training_start;
    println!("Training completed in {:?}", duration);

    for (ref inputs, _) in &dataset {
        let test = network.predict(inputs)[0];
        println!("{} XOR {} = {}", inputs[0], inputs[1], test);
    }
}
