# Claude Code Guidelines for sunpou

## Project Overview

`sunpou` (寸法) is a `no_std` Rust library providing compile-time dimensional
analysis for linear algebra — scalars, vectors, matrices with SI units,
coordinate frames, and prefix scaling.

## Key Design Principles

- **Type-level dimensions**: SI dimensions are `Dim<L,M,T,I,Th,N,J>` using `typenum`
- **Element-dimension matrices**: `ElemMat<E>` / `FrameElemMat<F,E>` — output dim inferred from `E × input_dim`
- **SI prefix tracking**: `Scalar<D, P>`, `UnitVec<D, N, P>` etc. with P = power of 10
- **Zero-cost**: All types are `#[repr(transparent)]` over nalgebra types
- **Generics-first**: Constraints via trait bounds, not macros
- **Frame safety**: `FrameVec<F, D, P>` / `FrameElemMat<F, E, R, C, P>` prevent cross-frame ops

## Build & Test

```bash
cargo test                    # All tests (161+)
cargo clippy -- -D warnings   # Lint
cargo build --examples        # All examples
cargo bench --bench zero_cost # Performance benchmarks
```

## Test Requirements (see docs/development-policy.md)

1. nalgebra cross-validation for every operation
2. uom cross-validation for dimension arithmetic
3. Zero-cost static assertions (size_of/align_of)
4. trybuild compile-fail tests for type safety
5. Criterion benchmarks

## Architecture (current)

- `src/dim.rs` — `Dim`, `DimMul`, `DimDiv` type-level dimension arithmetic
- `src/prefix.rs` — SI prefix types (Nano, Kilo, Mega, etc.)
- `src/scalar.rs` — `Scalar<D, P>` unit+prefix-tagged f64
- `src/unit_vec.rs` — `UnitVec<D, N, P>` N-dimensional vector
- `src/frame_vec.rs` — `FrameVec<F, D, P>` frame+dimension+prefix tagged 3D vector
- `src/elem_mat.rs` — `ElemMat<E, R, C, P>` element-dimension matrix (preferred)
- `src/frame_elem_mat.rs` — `FrameElemMat<F, E, R, C, P>` frame-tagged matrix (preferred)
- `src/rotation.rs` — `Rotation<F1, F2>` frame transformation
- `src/block.rs` — `BlockMat2x2`, `BlockMat3x3`, `BlockVec2`, `BlockVec3`
- `src/aliases.rs` — Common dimension type aliases

### Legacy (doc(hidden), backward compat)

- `src/unit_mat.rs` — `UnitMat<DR, DC, R, C>` (old DR/DC model, use ElemMat instead)
- `src/frame_unit_mat.rs` — `FrameUnitMat<F, DR, DC, R, C>` (old, use FrameElemMat instead)

## Design Decisions

Recorded in `docs/decisions/NNN-<topic>.md` (7 ADRs). Always create an ADR when
making non-trivial design choices, documenting alternatives considered and rationale.
