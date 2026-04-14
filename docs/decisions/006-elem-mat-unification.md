# ADR-006: ElemMat — 要素次元モデルへの一本化

## 判断内容

行列の型パラメータを「行次元 DR / 列次元 DC」から「要素次元 E」に変更する。

- `ElemMat<E, R, C>` が `UnitMat<DR, DC, R, C>` を置き換える
- `FrameElemMat<F, E, R, C>` が `FrameUnitMat<F, DR, DC, R, C>` を置き換える

## 背景

`UnitMat<DR, DC>` / `FrameUnitMat<F, DR, DC>` モデルでは、行列が特定の入力次元 DC に
固定される。同じ物理行列（慣性テンソル等）を異なる入力に使うには `rescale_dims` が必要だった。

## 選択肢

### A. DR/DC モデルのまま rescale_dims を使う (棄却)

```rust
let i_vel: FrameUnitMat<Body, AngularMomentum, AngularVelocity, 3, 3> = ...;
let i_acc = i_vel.rescale_dims::<InvTime>(); // 手動変換が必要
```

- **Cons**: 明示的な変換が冗長。同じ行列から2つのオブジェクトを管理する必要がある。

### B. 要素次元 E モデル (採用)

```rust
let inertia = FrameElemMat::<Body, MomentOfInertia, 3, 3>::from_raw_unchecked(i_raw);
let ang_mom = inertia * omega;      // 自動: MomentOfInertia × AngularVelocity = AngularMomentum
let torque  = inertia * omega_dot;  // 自動: MomentOfInertia × AngularAcceleration = Torque
let omega   = inertia.try_inverse().unwrap() * ang_mom;  // 自動推論
```

- **Pros**: 同一オブジェクトが任意の入力に対応。出力次元は型推論で自動解決。
  `rescale_dims` 不要。API が直感的。型パラメータが1つ減り simpler。
- **Cons**: 入力次元を制約しないため、誤った入力でも乗算は通る（ただし出力次元が
  意図と異なるため、下流の演算で型エラーになる）。

## 演算規則

| 操作 | 型 |
|------|-----|
| mat × vec | `ElemMat<E> × UnitVec<D> → UnitVec<E×D>` |
| mat × mat | `ElemMat<E1> × ElemMat<E2> → ElemMat<E1×E2>` |
| 転置 | `ElemMat<E> → ElemMat<1/E>` |
| 逆行列 | `ElemMat<E> → ElemMat<1/E>` |
| 加算 | `ElemMat<E> + ElemMat<E>` (同一 E のみ) |
| 恒等行列 | `ElemMat<Dimensionless>` |

## 結論

B を採用。型推論による自動次元解決が最大の利点。DR/DC モデルの旧型は
`doc(hidden)` として残し、後方互換性を維持する。
