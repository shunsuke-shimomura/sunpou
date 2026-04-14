# uolgebra

Unit-aware linear algebra library for Rust with compile-time SI dimension checking.

## Features

- **Compile-time unit safety**: SI dimensions tracked at the type level via `typenum`
- **Zero-cost abstraction**: `#[repr(transparent)]` — identical performance to raw `nalgebra`
- **`no_std`**: Works in embedded/FSW environments
- **Scalars, vectors, matrices**: `Scalar<D>`, `UnitVec<D, N>`, `FrameVec<F, D>`, `UnitMat<DR, DC, R, C>`
- **Frame safety**: `FrameVec<F, D>` prevents cross-frame operations at compile time
- **Block matrices**: `BlockMat2x2`, `BlockMat3x3` for state transition matrices and EKF
- **Heterogeneous operations**: Cross-dimension dot/cross products (e.g. Force · Length → Energy)

## Quick Example

```rust
use uolgebra::prelude::*;
use uolgebra::scalar::Scalar;

// F = m * a — dimensions checked at compile time
let mass = Scalar::<Mass>::from_raw_unchecked(100.0);
let accel = Scalar::<Acceleration>::from_raw_unchecked(9.8);
let force: Scalar<Force> = mass * accel; // 980 N

// This would NOT compile:
// let _ = mass + accel; // Mass + Acceleration → type error
```

See `examples/` for more: `basic_scalar`, `vectors`, `frame_transform`, `orbital_stm`, `ekf`.

## License

MIT OR Apache-2.0
