use ndarray::{Array, Dimension};
use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct XryGrad<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    D: Dimension,
{
    data: Array<T, D>,
    grad: Option<Rc<RefCell<Array<T, D>>>>,
    backward: Option<Rc<dyn Fn()>>,
}

pub struct Xry<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    D: Dimension,
{
    grad: XryGrad<T, D>,
}

impl<T, D> Xry<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Default,
    D: Dimension,
{
    pub fn new(data: Array<T, D>) -> Self {
        Xry {
            grad: XryGrad {
                data,
                grad: None,
                backward: None,
            },
        }
    }

    pub fn add(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = &self.grad.data + &other.grad.data;
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(|| {
                // Implement backward pass for addition
            })),
        };
        Xry { grad }
    }

    pub fn sub(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = &self.grad.data - &other.grad.data;
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(|| {
                // Implement backward pass for subtraction
            })),
        };
        Xry { grad }
    }

    pub fn mul(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = &self.grad.data * &other.grad.data;
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(|| {
                // Implement backward pass for multiplication
            })),
        };
        Xry { grad }
    }

    pub fn div(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = &self.grad.data / &other.grad.data;
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(|| {
                // Implement backward pass for division
            })),
        };
        Xry { grad }
    }

    pub fn backward(&self) {
        if let Some(ref backward) = self.grad.backward {
            backward();
        }
    }
}

// TODO: Implement JIT compilation
// TODO: Implement multi-device support
