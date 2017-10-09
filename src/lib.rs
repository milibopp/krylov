extern crate nalgebra;
extern crate alga;
#[cfg(test)]
#[macro_use]
extern crate approx;

use nalgebra::{dot, norm, zero};
use alga::linear::{FiniteDimVectorSpace, VectorSpace, NormedSpace};
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
    where V: FiniteDimVectorSpace + NormedSpace + Clone,
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
    where V: FiniteDimVectorSpace + Clone,
          V::Field: Clone,
          F: Fn(&V) -> V
{
    let rho = dot(reference_diff, &state.difference);
    let beta = (rho.clone() / state.rho) * (state.alpha / state.omega.clone());
    let p = state.difference.clone() + (state.p - state.v * state.omega.clone()) * beta;
    let v = matrix(&p);
    let alpha = rho.clone() / dot(reference_diff, &v);
    let value = state.value + p.clone() * alpha.clone();
    State { value, rho, p, v, alpha, ..state }
}

fn second_step<V, F>(state: State<V>, matrix: F) -> State<V>
    where V: FiniteDimVectorSpace + Clone,
          V::Field: Clone,
          F: Fn(&V) -> V
{
    let s = state.difference - state.v.clone() * state.alpha.clone();
    let t = matrix(&s);
    let omega = dot(&t, &s) / dot(&t, &t);
    let value = state.value + s.clone() * state.omega.clone();
    let difference = s - t * state.omega;
    State { value, omega, difference, ..state }
}


#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3, Matrix4, Vector4};
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
}
