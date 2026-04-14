# ADR-003: UnitMat の次元モデル

## 判断内容

`UnitMat<DR, DC, R, C>` で行次元 `DR` と列次元 `DC` を持ち、
行列-ベクトル乗算を `UnitMat<DR, DC, R, C> * UnitVec<DC, C> → UnitVec<DR, R>` とする。

## 背景

行列に単位を付ける方法は複数考えられる。行列 A がベクトル x に作用して y = Ax を
計算するとき、y[i] = Σ_j A[i][j] * x[j] であり、各 j で A[i][j] * x[j] が
同じ次元を持つ必要がある。

## 選択肢

### A. 行次元 + 列次元 (DR, DC) モデル (採用)

```rust
pub struct UnitMat<DR, DC, const R: usize, const C: usize>;
// A * x: UnitMat<DR, DC> * UnitVec<DC> → UnitVec<DR>
```

- **Pros**: シンプル。行列-ベクトル乗算の型チェックが `DC` の一致だけで済む。
  行列-行列乗算では中間次元がキャンセルする（`UnitMat<DR, DM> * UnitMat<DM, DC> → UnitMat<DR, DC>`）。
  転置は `UnitMat<DR, DC> → UnitMat<DC, DR>` で自然。
  逆行列は `UnitMat<DR, DC>⁻¹ → UnitMat<DC, DR>` で正しい。
- **Cons**: 各要素の次元は暗黙的に `DR / DC`。行ごと・列ごとに異なる次元を持つ
  行列（状態遷移行列等）は表現できない → ブロック行列で対応。

### B. 要素次元 (DE) のみ

```rust
pub struct UnitMat<DE, const R: usize, const C: usize>;
```

- **Pros**: さらにシンプル。
- **Cons**: 入力/出力の次元関係が表現できない。

### C. 行ごと・列ごとの次元ベクトル

各行、各列に個別の次元を持たせる。

- **Pros**: 最も表現力が高い。
- **Cons**: 型パラメータが爆発する。実用的でない。→ ブロック行列で対応。

## 結論

Aを採用。均質行列には DR/DC モデルが最適。異種次元はブロック行列
(`BlockMat2x2` 等) で各ブロックを個別の `UnitMat<DR, DC>` として構成する。
