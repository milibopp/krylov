extern crate nalgebra;
extern crate alga;
extern crate sprs;
extern crate ndarray;
#[cfg(test)]
#[macro_use]
extern crate approx;

use nalgebra::{norm, zero};
use alga::linear::{InnerSpace, VectorSpace};
use alga::general::{Identity, Multiplicative};

#[derive(Clone, Copy)]
struct State<V: VectorSpace> {
    value: V,
    difference: V,
    rho: V::Field,
    omega: V::Field,
    alpha: V::Field,
    p: V,
    v: V
}

pub fn bicgstab<V, F>(guess: V, vector: V, matrix: F, tolerance: V::Field) -> V
    where V: InnerSpace + Clone,
          V::Field: Identity<Multiplicative> + PartialOrd + Clone,
          F: Fn(&V) -> V,
{
    let matrix = &matrix;
    let initial = initial_state(guess, vector.clone(), matrix);
    let good_enough = |value: &_| norm(&(matrix(value) - vector.clone())) < tolerance;
    let initial_difference = &initial.difference;
    let steps = (0..2).cycle()
        .map(|i| move |state|
            if i == 0 {
                first_step(state, matrix, &initial_difference)
            } else {
                second_step(state, matrix)
            }
        );

    steps.scan(initial.clone(), |state, step| { *state = step(state.clone()); Some(state.clone()) })
        .take(1_000)
        .find(|state| good_enough(&state.value))
        .map(|state| state.value)
        .unwrap()
}

fn initial_state<V, F>(guess: V, vector: V, matrix: F) -> State<V>
    where V: VectorSpace,
          V::Field: Identity<Multiplicative>,
          F: Fn(&V) -> V
{
    State {
        difference: vector - matrix(&guess),
        value: guess,
        rho: Identity::identity(),
        omega: Identity::identity(),
        alpha: Identity::identity(),
        p: zero(),
        v: zero()
    }
}

fn first_step<V, F>(state: State<V>, matrix: F, reference_diff: &V) -> State<V>
    where V: InnerSpace + Clone,
          V::Field: Clone,
          F: Fn(&V) -> V
{
    let rho = reference_diff.inner_product(&state.difference);
    let beta = (rho.clone() / state.rho) * (state.alpha / state.omega.clone());
    let p = state.difference.clone() + (state.p - state.v * state.omega.clone()) * beta;
    let v = matrix(&p);
    let alpha = rho.clone() / reference_diff.inner_product(&v);
    let value = state.value + p.clone() * alpha.clone();
    State { value, rho, p, v, alpha, ..state }
}

fn second_step<V, F>(state: State<V>, matrix: F) -> State<V>
    where V: InnerSpace + Clone,
          V::Field: Clone,
          F: Fn(&V) -> V
{
    let s = state.difference - state.v.clone() * state.alpha.clone();
    let t = matrix(&s);
    let omega = t.inner_product(&s) / t.norm_squared();
    let value = state.value + s.clone() * state.omega.clone();
    let difference = s - t * state.omega;
    State { value, omega, difference, ..state }
}


#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3, Matrix4, Vector4};
    use sprs::CsMat;
    use super::*;

    #[test]
    fn solves_simple_system_of_linear_equations() {
        let matrix = Matrix3::new(
            2., 1., 0.,
            0., 1., 0.,
            0., 3., 1.,
        );
        let solution = Vector3::new(2., -1., -2.);
        let vector = matrix * solution;
        let guess = Vector3::new(1., 0., 1.);
        let result = bicgstab(guess, vector, |v| matrix * v, 1e-8);
        assert_relative_eq!(result, solution, max_relative = 1e-4);
    }

    #[test]
    fn solves_differently_sized_system() {
        let matrix = Matrix4::new_random();
        let solution = Vector4::new_random();
        let vector = matrix * solution;
        let guess = Vector4::new_random();
        let result = bicgstab(guess, vector, |v| matrix * v, 1e-8);
        assert_relative_eq!(result, solution, max_relative = 1e-4);
    }

    #[test]
    fn solves_system_represented_by_sparse_matrix() {
        use nalgebra::VectorN;
        use nalgebra::core::dimension::U10;
        let matrix = CsMat::<f64>::new(
            (10, 10),
            vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19],
            vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            vec![1.; 19]
        );
        let multiply = |vector: &VectorN<f64, U10>| {
            use ndarray::arr1;
            let a = &matrix * &arr1(&vector.data);
            let b = a.to_vec().into_iter();
            VectorN::<f64, U10>::from_iterator(b)
        };
        let solution = VectorN::<f64, U10>::new_random();
        let guess = VectorN::<f64, U10>::new_random();
        let vector = multiply(&solution);
        let result = bicgstab(guess, vector, multiply, 1e-8);
        assert_relative_eq!(result, solution, max_relative = 1e-4);
    }
}
