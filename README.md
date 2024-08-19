# xry_1
XPU Array Library

## Overview
xry_1 is a high-performance array library designed for Xtream Processing Units (XPUs). This library aims to provide the fastest array operations for XPU development, with features similar to JAX and PyTorch.

## Features
- High-performance array operations
- Cross-device compatibility
- Optimized for XPU development
- Tensor operations (add, subtract, multiply, divide)
- Enhanced automatic differentiation with complex computation graphs
- Multi-dimensional array support
- SIMD-friendly data layouts
- Just-In-Time (JIT) compilation for improved performance
- Parallel execution of operations using Rayon
- XPU compatibility layer (in progress)

## Installation
To use xry_1 in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
xry_1 = "0.1.0"
```

## Usage
Here are examples demonstrating the key features of xry_1:

### Basic Operations and Automatic Differentiation

```rust
use xry_1::Xry;
use ndarray::Array;

fn main() {
    // Create two 2x2 arrays
    let a = Xry::new(Array::from_elem((2, 2), 1.0));
    let b = Xry::new(Array::from_elem((2, 2), 2.0));

    // Perform addition
    let c = a.add(&b);

    // Perform automatic differentiation
    c.backward();

    // Print the result and gradients
    println!("Result: {:?}", c.grad.data);
    println!("Gradient of a: {:?}", a.grad.grad.as_ref().unwrap().borrow());
    println!("Gradient of b: {:?}", b.grad.grad.as_ref().unwrap().borrow());
}
```

### JIT Compilation

```rust
use xry_1::{Xry, jit_compile};
use ndarray::Array;

fn main() {
    let a = Xry::new(Array::from_elem((100, 100), 1.0));
    let b = Xry::new(Array::from_elem((100, 100), 2.0));

    // Compile the addition operation using JIT
    let jit_add = jit_compile(|x: &Xry<f64, _>, y: &Xry<f64, _>| x.add(y));

    // Perform JIT-compiled addition
    let result = jit_add(&a, &b);

    println!("JIT Result: {:?}", result.grad.data);
}
```

### Parallelism

```rust
use xry_1::{Xry, parallel_execute};
use ndarray::Array;

fn main() {
    let a = Xry::new(Array::from_elem((1000, 1000), 1.0));
    let b = Xry::new(Array::from_elem((1000, 1000), 2.0));

    // Perform parallel addition
    let result = parallel_execute(|x: &Xry<f64, _>, y: &Xry<f64, _>| x.add(y), &a, &b);

    println!("Parallel Result: {:?}", result.grad.data);
}
```

## Contributing
Contributions to improve xry_1's performance and functionality are welcome. Please refer to our contributing guidelines (to be added) for more information.

## License
(License information to be added)

## Roadmap
- Implement comprehensive performance benchmarks
- Optimize JIT compilation for various operation types
- Extend parallelism to more complex tensor operations
- Enhance XPU compatibility layer
- Implement advanced automatic differentiation techniques (e.g., higher-order gradients)
- Develop a user-friendly API for custom operations
- Improve error handling and diagnostics
- Create extensive documentation and tutorials
