# ADR-005: FrameUnitMat — フレーム付き行列型

## 判断内容

`FrameUnitMat<F, DR, DC, R, C>` を導入し、行列にフレーム情報を持たせる。
`FrameUnitMat<F, ...> * FrameVec<F, ...>` のように、同一フレームでなければ
乗算できない制約をコンパイル時に保証する。

## 背景

`UnitMat` は単位次元を追跡するが、座標系フレームを持たない。そのため、
ECI で計算した状態遷移行列を ECEF の状態ベクトルに適用しても型エラーにならない。

物理的には STM、慣性テンソル、制御ゲイン行列はすべて特定のフレーム内で
定義されるため、フレームの不一致は常にバグである。

## 選択肢

### A. UnitMat に Frame を追加 (棄却)

既存の `UnitMat<DR, DC, R, C>` を `UnitMat<F, DR, DC, R, C>` に変更する。

- **Cons**: フレームが不要な純粋数学の場面で冗長。既存 API の破壊的変更。

### B. UnitMat * FrameVec の Mul を追加 (棄却)

`UnitMat` は frame-less のまま、`UnitMat * FrameVec → FrameVec` を実装。

- **Cons**: 行列側のフレームが検証されない。ECI の STM × ECEF のベクトルが通る。

### C. 新型 FrameUnitMat を追加 (採用)

`UnitMat` を維持しつつ、`FrameUnitMat<F, DR, DC, R, C>` を新規追加。

- **Pros**: 後方互換。フレーム不要な場面は `UnitMat` を使用。フレーム安全が
  必要な場面は `FrameUnitMat` を使用。`BlockMat2x2` の型パラメータが
  ジェネリックなので、`FrameUnitMat` のブロック行列もそのまま動く。
- **Cons**: 2種類の行列型が存在する。

## 制御ゲインの扱い

「ゲインには単位がない」という直感は誤り。ゲイン行列の次元は
`output_dim / input_dim` である:

| ゲイン | 物理的意味 | DR (出力) | DC (入力) |
|--------|-----------|-----------|-----------|
| Kp (姿勢) | N·m / rad | Torque | Dimensionless |
| Kv (姿勢) | N·m·s / rad | Torque | AngularVelocity |
| Kp (位置) | N / m | Force | Length |
| Kv (位置) | N·s / m | Force | Velocity |

uolgebra はこれを自然に表現する:
```rust
let kp = FrameUnitMat::<Body, Torque, Dimensionless, 3, 3>::from_raw_unchecked(...);
let kv = FrameUnitMat::<Body, Torque, AngularVelocity, 3, 3>::from_raw_unchecked(...);
let torque: FrameVec<Body, Torque> = -(kp * theta_err) - (kv * omega_err);
```

## 慣性テンソルの二重解釈

同一の数値行列 I が2つの異なる型を持つ:

1. `I · ω = L`: `FrameUnitMat<Body, AngularMomentum, AngularVelocity>`
2. `I · ω̇ = τ`: `FrameUnitMat<Body, Torque, AngularAcceleration>`

これは物理的に正しい — 同じ数値でも文脈によって次元が異なる。
`from_raw_unchecked` で同じ値から2つの型付き行列を作成する。

## 結論

C を採用。`UnitMat` (frame-less) と `FrameUnitMat` (frame-safe) の
二層構造により、柔軟性と安全性を両立する。
