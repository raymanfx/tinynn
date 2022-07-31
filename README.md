# Tiny Neural Networks (TinyNN)

This project is a crate for training and running neural networks in Rust. The term *tiny* refers to the simple nature of the project: it does not aim for sophisticated hardware acceleration or full model compatibility.

Instead, the code should be easy to read and extend. There are no fancy optimizers or automatic differentiation. The basic building blocks of neural networks such as matrix operations and activations are built from scratch. Higher-level abstractions such as layers are built on top of them.

The goal is to enable people to learn about neural networks and the Rust programming language at the same time. This code is not meant for production (just yet). Instead, it is meant to be hacked on and to provide an educational experience.
