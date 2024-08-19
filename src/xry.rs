use ndarray::{Array, Dimension, ArrayView, Zip};
use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use std::cell::RefCell;
use rayon::prelude::*;
use rustc_jit_utils::Jit;

#[derive(Clone)]
pub struct XryGrad<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Send + Sync,
    D: Dimension,
{
    data: Array<T, D>,
    grad: Option<Rc<RefCell<Array<T, D>>>>,
    backward: Option<Rc<dyn Fn() + Send + Sync>>,
}

pub struct Xry<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Send + Sync,
    D: Dimension,
{
    grad: XryGrad<T, D>,
    jit: Option<Jit>,
}

impl<T, D> Xry<T, D>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
{
    pub fn new(data: Array<T, D>) -> Self {
        Xry {
            grad: XryGrad {
                data,
                grad: None,
                backward: None,
            },
            jit: None,
        }
    }

    pub fn add(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = self.parallel_op(&other.grad.data, |a, b| a + b);
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(move || {
                let grad = result.grad.as_ref().unwrap().borrow();
                self.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b);
                other.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b);
            })),
        };
        Xry { grad, jit: self.jit.clone() }
    }

    pub fn sub(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = self.parallel_op(&other.grad.data, |a, b| a - b);
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(move || {
                let grad = result.grad.as_ref().unwrap().borrow();
                self.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b);
                other.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a -= b);
            })),
        };
        Xry { grad, jit: self.jit.clone() }
    }

    pub fn mul(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = self.parallel_op(&other.grad.data, |a, b| a * b);
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(move || {
                let grad = result.grad.as_ref().unwrap().borrow();
                self.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b * other.grad.data);
                other.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b * self.grad.data);
            })),
        };
        Xry { grad, jit: self.jit.clone() }
    }

    pub fn div(&self, other: &Xry<T, D>) -> Xry<T, D> {
        let result = self.parallel_op(&other.grad.data, |a, b| a / b);
        let grad = XryGrad {
            data: result.clone(),
            grad: Some(Rc::new(RefCell::new(Array::default(result.dim())))),
            backward: Some(Rc::new(move || {
                let grad = result.grad.as_ref().unwrap().borrow();
                self.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a += b / other.grad.data);
                other.grad.grad.as_ref().unwrap().borrow_mut().zip_mut_with(&*grad, |a, &b| *a -= b * self.grad.data / (other.grad.data * other.grad.data));
            })),
        };
        Xry { grad, jit: self.jit.clone() }
    }

    pub fn backward(&self) {
        if let Some(ref backward) = self.grad.backward {
            backward();
        }
    }

    fn parallel_op<F>(&self, other: &Array<T, D>, op: F) -> Array<T, D>
    where
        F: Fn(T, T) -> T + Sync + Send,
    {
        if let Some(ref jit) = self.jit {
            // Use JIT compilation for the operation
            jit.compile_and_run(|| {
                Zip::from(&self.grad.data)
                    .and(other)
                    .par_map_collect(|&a, &b| op(a.clone(), b.clone()))
            })
        } else {
            // Use parallel execution without JIT
            Zip::from(&self.grad.data)
                .and(other)
                .par_map_collect(|&a, &b| op(a.clone(), b.clone()))
        }
    }

    pub fn enable_jit(&mut self) {
        self.jit = Some(Jit::new());
    }
}
