//! Benchmarks verifying zero-cost abstraction.
//!
//! Compares sunpou typed operations against raw nalgebra operations
//! to ensure no runtime overhead from the type-level dimension tracking.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix3, SVector, Vector3};
use sunpou::prelude::*;
use sunpou::scalar::Scalar;
use sunpou::unit_vec::UnitVec;
use sunpou::unit_mat::UnitMat;
use sunpou::frame_vec::FrameVec;

struct Eci;

fn bench_scalar_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_mul");

    group.bench_function("raw_f64", |b| {
        let a = black_box(3.0_f64);
        let x = black_box(9.8_f64);
        b.iter(|| a * x)
    });

    group.bench_function("sunpou", |b| {
        let a = black_box(Scalar::<Mass>::from_raw(3.0));
        let x = black_box(Scalar::<Acceleration>::from_raw(9.8));
        b.iter(|| a * x)
    });

    group.finish();
}

fn bench_vec3_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3_dot");

    group.bench_function("raw_nalgebra", |b| {
        let a = black_box(Vector3::new(1.0, 2.0, 3.0));
        let x = black_box(Vector3::new(4.0, 5.0, 6.0));
        b.iter(|| a.dot(&x))
    });

    group.bench_function("sunpou_unitvec", |b| {
        let a = black_box(UnitVec::<Length, 3>::from_raw(
            SVector::from([1.0, 2.0, 3.0]),
        ));
        let x = black_box(UnitVec::<Velocity, 3>::from_raw(
            SVector::from([4.0, 5.0, 6.0]),
        ));
        b.iter(|| a.dot(&x))
    });

    group.bench_function("sunpou_framevec", |b| {
        let a = black_box(FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0));
        let x = black_box(FrameVec::<Eci, Velocity>::new(4.0, 5.0, 6.0));
        b.iter(|| a.dot(&x))
    });

    group.finish();
}

fn bench_mat3_mul_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3_mul_vec");

    group.bench_function("raw_nalgebra", |b| {
        let m = black_box(Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
        let v = black_box(Vector3::new(1.0, 2.0, 3.0));
        b.iter(|| m * v)
    });

    group.bench_function("sunpou", |b| {
        let m = black_box(UnitMat::<Velocity, Length, 3, 3>::from_raw(
            Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        ));
        let v = black_box(UnitVec::<Length, 3>::from_raw(
            SVector::from([1.0, 2.0, 3.0]),
        ));
        b.iter(|| m * v)
    });

    group.finish();
}

fn bench_vec3_cross(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3_cross");

    group.bench_function("raw_nalgebra", |b| {
        let a = black_box(Vector3::new(1.0, 2.0, 3.0));
        let x = black_box(Vector3::new(4.0, 5.0, 6.0));
        b.iter(|| a.cross(&x))
    });

    group.bench_function("sunpou", |b| {
        let a = black_box(UnitVec::<Length, 3>::from_raw(
            SVector::from([1.0, 2.0, 3.0]),
        ));
        let x = black_box(UnitVec::<Velocity, 3>::from_raw(
            SVector::from([4.0, 5.0, 6.0]),
        ));
        b.iter(|| a.cross(&x))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_mul,
    bench_vec3_dot,
    bench_mat3_mul_vec,
    bench_vec3_cross,
);
criterion_main!(benches);
