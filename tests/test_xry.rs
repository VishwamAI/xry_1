use ndarray::{Array, Dim};
use crate::xry::Xry;

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

    // TODO: Add tests for performance benchmarks
    // TODO: Add tests for edge cases (e.g., division by zero)
}
