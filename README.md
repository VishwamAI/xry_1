# xry_1
XPU Array Library

## Overview
xry_1 is a high-performance array library designed for Xtream Processing Units (XPUs). This library aims to provide the fastest array operations for XPU development, with features similar to JAX and PyTorch.

## Features
- High-performance array operations
- Cross-device compatibility
- Optimized for XPU development
- Tensor operations (add, subtract, multiply, divide)
- Automatic differentiation
- Multi-dimensional array support
- SIMD-friendly data layouts
- Just-In-Time (JIT) compilation capabilities (in progress)
- XPU compatibility layer (in progress)

## Installation
To use xry_1 in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
xry_1 = "0.1.0"
```

## Usage
Here's a basic example of how to use xry_1:

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

    // Print the result
    println!("Result: {:?}", c.grad.data);
}
```

## Contributing
Contributions to improve xry_1's performance and functionality are welcome. Please refer to our contributing guidelines (to be added) for more information.

## License
(License information to be added)

## Roadmap
- Implement performance benchmarks
- Add support for more complex tensor operations
- Enhance automatic differentiation capabilities
- Complete JIT compilation implementation
- Finalize XPU compatibility layer
