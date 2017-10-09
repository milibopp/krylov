extern crate nalgebra;
#[cfg(test)]
#[macro_use]
extern crate approx;

use nalgebra::{Vector3, Matrix3, dot, norm, zero};

#[derive(Clone, Copy)]
struct State {
    value: Vector3<f64>,
    difference: Vector3<f64>,
    rho: f64,
    omega: f64,
    alpha: f64,
    p: Vector3<f64>,
    v: Vector3<f64>
}

pub fn bicgstab(guess: Vector3<f64>, vector: Vector3<f64>, matrix: Matrix3<f64>) -> Vector3<f64> {
    let initial = initial_state(guess, &vector, &matrix);
    let good_enough = |value: &Vector3<f64>| norm(&(matrix * value - vector)) < 0.01;
    let steps = (0..2).cycle()
        .map(|i| move |state|
            if i == 0 {
                first_step(state, &matrix, &initial.difference)
            } else {
                second_step(state, &matrix)
            }
        );

    steps.scan(initial, |state, step| { *state = step(*state); Some(*state) })
        .find(|state| good_enough(&state.value))
        .map(|state| state.value)
        .unwrap()
}

fn initial_state(guess: Vector3<f64>, vector: &Vector3<f64>, matrix: &Matrix3<f64>) -> State {
    State {
        value: guess,
        difference: vector - matrix * &guess,
        rho: 1.,
        omega: 1.,
        alpha: 1.,
        p: zero(),
        v: zero()
    }
}

fn first_step(state: State, matrix: &Matrix3<f64>, reference_diff: &Vector3<f64>) -> State {
    let rho = dot(reference_diff, &state.difference);
    let beta = (rho / state.rho) * (state.alpha / state.omega);
    let p = state.difference + beta * (state.p - state.omega * state.v);
    let v = matrix * p;
    let alpha = rho / dot(reference_diff, &v);
    let value = state.value + alpha * p;
    State { value, rho, p, v, alpha, ..state }
}

fn second_step(state: State, matrix: &Matrix3<f64>) -> State {
    let s = state.difference - state.alpha * state.v;
    let t = matrix * s;
    let omega = dot(&t, &s) / dot(&t, &t);
    let value = state.value + state.omega * s;
    let difference = s - state.omega * t;
    State { value, omega, difference, ..state }
}


#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3};
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
        let result = bicgstab(guess, vector, matrix);
        assert_relative_eq!(result, solution, max_relative = 1e-4);
    }
}
