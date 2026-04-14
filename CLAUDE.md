# Claude Code Guidelines for uolgebra

## Project Overview

`uolgebra` is a `no_std` Rust library providing compile-time SI dimension checking
for linear algebra types (scalars, vectors, matrices, block matrices).

## Key Design Principles

- **Type-level dimensions**: SI dimensions are `Dim<L,M,T,I,Th,N,J>` using `typenum`
- **Zero-cost**: All types are `#[repr(transparent)]` over nalgebra types
- **Generics-first**: Constraints via trait bounds, not macros
- **Frame safety**: `FrameVec<F, D>` tags vectors with coordinate frame + dimension

## Build & Test

```bash
cargo test                    # All tests (43+)
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

## Architecture

- `src/dim.rs` — `Dim`, `DimMul`, `DimDiv` type-level dimension arithmetic
- `src/scalar.rs` — `Scalar<D>` unit-tagged f64
- `src/unit_vec.rs` — `UnitVec<D, N>` N-dimensional unit-tagged vector
- `src/frame_vec.rs` — `FrameVec<F, D>` frame+dimension tagged 3D vector
- `src/unit_mat.rs` — `UnitMat<DR, DC, R, C>` unit-tagged matrix
- `src/rotation.rs` — `Rotation<F1, F2>` frame transformation
- `src/block.rs` — `BlockMat2x2`, `BlockMat3x3`, `BlockVec2`, `BlockVec3`
- `src/aliases.rs` — Common dimension type aliases
- `docs/decisions/` — Architecture Decision Records

## Design Decisions

Recorded in `docs/decisions/NNN-<topic>.md`. Always create an ADR when making
non-trivial design choices, documenting alternatives considered and rationale.
