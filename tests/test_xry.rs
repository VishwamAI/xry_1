use ndarray::{Array, Dim};
use crate::xry::{Xry, jit_compile, parallel_execute};
use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xry_creation() {
        let data = Array::from_elem(Dim([2, 2]), 1.0);
        let xry = Xry::new(data);
        assert_eq!(xry.grad.data, Array::from_elem(Dim([2, 2]), 1.0));
    }

    #[test]
    fn test_xry_addition() {
        let data1 = Array::from_elem(Dim([2, 2]), 1.0);
        let data2 = Array::from_elem(Dim([2, 2]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let result = xry1.add(&xry2);
        assert_eq!(result.grad.data, Array::from_elem(Dim([2, 2]), 3.0));
    }

    #[test]
    fn test_xry_subtraction() {
        let data1 = Array::from_elem(Dim([2, 2]), 5.0);
        let data2 = Array::from_elem(Dim([2, 2]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let result = xry1.sub(&xry2);
        assert_eq!(result.grad.data, Array::from_elem(Dim([2, 2]), 3.0));
    }

    #[test]
    fn test_xry_multiplication() {
        let data1 = Array::from_elem(Dim([2, 2]), 2.0);
        let data2 = Array::from_elem(Dim([2, 2]), 3.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let result = xry1.mul(&xry2);
        assert_eq!(result.grad.data, Array::from_elem(Dim([2, 2]), 6.0));
    }

    #[test]
    fn test_xry_division() {
        let data1 = Array::from_elem(Dim([2, 2]), 6.0);
        let data2 = Array::from_elem(Dim([2, 2]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let result = xry1.div(&xry2);
        assert_eq!(result.grad.data, Array::from_elem(Dim([2, 2]), 3.0));
    }

    #[test]
    fn test_automatic_differentiation() {
        let data1 = Array::from_elem(Dim([2, 2]), 1.0);
        let data2 = Array::from_elem(Dim([2, 2]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let result = xry1.add(&xry2);

        // Set initial gradient
        *result.grad.grad.as_ref().unwrap().borrow_mut() = Array::from_elem(Dim([2, 2]), 1.0);

        // Perform backward pass
        result.backward();

        // Check gradients
        assert_eq!(*xry1.grad.grad.as_ref().unwrap().borrow(), Array::from_elem(Dim([2, 2]), 1.0));
        assert_eq!(*xry2.grad.grad.as_ref().unwrap().borrow(), Array::from_elem(Dim([2, 2]), 1.0));
    }

    #[test]
    fn test_jit_compilation() {
        let data1 = Array::from_elem(Dim([100, 100]), 1.0);
        let data2 = Array::from_elem(Dim([100, 100]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);

        let start = Instant::now();
        let result = xry1.add(&xry2);
        let normal_duration = start.elapsed();

        let jit_fn = jit_compile(|a: &Xry<f64, Dim<[usize; 2]>>, b: &Xry<f64, Dim<[usize; 2]>>| a.add(b));
        let start = Instant::now();
        let jit_result = jit_fn(&xry1, &xry2);
        let jit_duration = start.elapsed();

        assert_eq!(result.grad.data, jit_result.grad.data);
        assert!(jit_duration < normal_duration, "JIT compilation should be faster");
    }

    #[test]
    fn test_parallelism() {
        let data1 = Array::from_elem(Dim([1000, 1000]), 1.0);
        let data2 = Array::from_elem(Dim([1000, 1000]), 2.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);

        let start = Instant::now();
        let result = xry1.add(&xry2);
        let normal_duration = start.elapsed();

        let start = Instant::now();
        let parallel_result = parallel_execute(|a: &Xry<f64, Dim<[usize; 2]>>, b: &Xry<f64, Dim<[usize; 2]>>| a.add(b), &xry1, &xry2);
        let parallel_duration = start.elapsed();

        assert_eq!(result.grad.data, parallel_result.grad.data);
        assert!(parallel_duration < normal_duration, "Parallel execution should be faster");
    }

    #[test]
    fn test_complex_automatic_differentiation() {
        let data1 = Array::from_elem(Dim([2, 2]), 2.0);
        let data2 = Array::from_elem(Dim([2, 2]), 3.0);
        let data3 = Array::from_elem(Dim([2, 2]), 4.0);
        let xry1 = Xry::new(data1);
        let xry2 = Xry::new(data2);
        let xry3 = Xry::new(data3);

        // (x1 * x2) + (x2 / x3)
        let result = xry1.mul(&xry2).add(&xry2.div(&xry3));

        // Set initial gradient
        *result.grad.grad.as_ref().unwrap().borrow_mut() = Array::from_elem(Dim([2, 2]), 1.0);

        // Perform backward pass
        result.backward();

        // Check gradients (values are approximate due to floating-point arithmetic)
        assert!((*xry1.grad.grad.as_ref().unwrap().borrow() - Array::from_elem(Dim([2, 2]), 3.0)).sum().abs() < 1e-6);
        assert!((*xry2.grad.grad.as_ref().unwrap().borrow() - Array::from_elem(Dim([2, 2]), 2.25)).sum().abs() < 1e-6);
        assert!((*xry3.grad.grad.as_ref().unwrap().borrow() - Array::from_elem(Dim([2, 2]), -0.1875)).sum().abs() < 1e-6);
    }
}
