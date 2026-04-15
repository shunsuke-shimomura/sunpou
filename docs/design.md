# sunpou — 単位付き線形代数ライブラリ

物理単位 (SI 次元) をコンパイル時に追跡し、スカラー・ベクトル・行列・ブロック行列を
統一的に扱う `no_std` Rust ライブラリ。

## 動機

### 現状の課題

- **s7e**: シミュレーションフレームワークだが、物理量はすべて `f64` で扱われており、
  単位の整合性はプログラマの注意に依存している
- **orts/arika**: 座標系 (Frame) と時刻系 (TimeScale) を phantom type でコンパイル時に
  区別する優れた設計を持つが、物理単位 (m, kg, s, ...) の追跡は行わない
- **uom**: SI 次元をコンパイル時に検査するが、スカラー値のみが対象。
  ベクトル・行列・座標系フレームには非対応

### sunpou が解決すること

1. スカラー・ベクトル・行列のすべてで、物理単位の整合性をコンパイル時に保証する
2. 座標系フレームの型安全性を単位と直交する軸として統合する
3. ブロック行列による異種単位の構造的な合成を可能にする（状態遷移行列、ヤコビアン、EKF）
4. nalgebra や生 `f64` との相互変換を明示的な境界で提供する

## 設計方針

### 原則

- **コンパイル時検査**: 単位の不整合は型エラーとして検出される。ランタイムコストはゼロ
- **`no_std`**: 組込み環境・FSW (flight software) で使用可能。`alloc` も不要（固定サイズ）
- **nalgebra との共存**: 内部ストレージは nalgebra 型を活用しつつ、型レベルで単位を付加する。
  最終的に nalgebra に「剥がす」操作は明示的 API で行う
- **段階的採用**: 既存コードベースに一括導入ではなく、境界を明確にして段階的に導入できる
- **完全な汎用性**: 数学的・物理的に正当な演算であれば、表現できない行列はなく、
  できない演算はない状態を目指す。ジェネリックな型パラメータと trait bound で制約を
  表現し、マクロは利便性のために提供するが、制約の実現はジェネリクスで行う

### uom との差異

| | uom | sunpou |
|---|---|---|
| スカラー | `Quantity<D, U, V>` | `Scalar<D>` |
| ベクトル | 非対応 | `UnitVec<D, N>`, `FrameVec<F, D>` |
| 行列 | 非対応 | `UnitMat<DR, DC, R, C>` |
| ブロック行列 | 非対応 | `BlockMat` (型レベル構造) |
| フレーム | 非対応 | 型パラメータ `F` |
| ストレージ | 任意の `num::Num` | `f64` (初期), nalgebra ベース |
| 単位系 | 7 次元 ISQ + Kind | 7 次元 SI (typenum) |
| 単位変換 | 自動 (autoconvert) | 基本単位のみ (SI base), 変換は明示的 |

## 型システム設計

### SI 次元の型レベル表現

uom と同様に `typenum` クレートを利用し、SI 基本量の指数をコンパイル時整数で表現する:

```rust
use typenum::{Z0, P1, P2, N1, N2, Integer};

/// SI 次元を表す型。各パラメータは基本量の指数。
/// L: 長さ, M: 質量, T: 時間, I: 電流, Th: 温度, N: 物質量, J: 光度
pub struct Dim<L, M, T, I = Z0, Th = Z0, N = Z0, J = Z0>;
```

頻出する次元の型エイリアス:

```rust
pub type Dimensionless = Dim<Z0, Z0, Z0>;
pub type Length        = Dim<P1, Z0, Z0>;         // m
pub type Mass          = Dim<Z0, P1, Z0>;         // kg
pub type Time          = Dim<Z0, Z0, P1>;         // s
pub type Velocity      = Dim<P1, Z0, N1>;         // m/s
pub type Acceleration  = Dim<P1, Z0, N2>;         // m/s²
pub type Force         = Dim<P1, P1, N2>;         // kg·m/s² = N
pub type Momentum      = Dim<P1, P1, N1>;         // kg·m/s
pub type Energy        = Dim<P2, P1, N2>;         // kg·m²/s² = J
pub type AngularVelocity = Dim<Z0, Z0, N1>;       // rad/s (角度は無次元)
```

### 次元の演算

乗除算は指数の加減算として型レベルで表現される:

```rust
/// D1 * D2: 各指数を加算
type DimMul<D1, D2> = ...;

/// D1 / D2: 各指数を減算
type DimDiv<D1, D2> = ...;

// 例: Velocity = Length / Time
//   Dim<P1,Z0,Z0> / Dim<Z0,Z0,P1> = Dim<P1,Z0,N1> ✓
```

### スカラー: `Scalar<D>`

最も基本的な型。物理次元 `D` を持つ単一の `f64` 値。

```rust
#[repr(transparent)]
pub struct Scalar<D> {
    value: f64,
    _dim: PhantomData<D>,
}
```

演算規則:
- `Scalar<D> + Scalar<D> → Scalar<D>` (同次元のみ加算可能)
- `Scalar<D> - Scalar<D> → Scalar<D>`
- `Scalar<D1> * Scalar<D2> → Scalar<DimMul<D1, D2>>`
- `Scalar<D1> / Scalar<D2> → Scalar<DimDiv<D1, D2>>`
- `Scalar<D> * f64 → Scalar<D>` (無次元スカラー倍)

### N 次元ベクトル: `UnitVec<D, const N: usize>`

同一次元の成分を持つ N 次元ベクトル。フレーム情報を持たない。

```rust
#[repr(transparent)]
pub struct UnitVec<D, const N: usize> {
    value: SVector<f64, N>,  // nalgebra の固定サイズベクトル
    _dim: PhantomData<D>,
}
```

演算規則:
- `UnitVec<D, N> + UnitVec<D, N> → UnitVec<D, N>`
- `Scalar<D1> * UnitVec<D2, N> → UnitVec<DimMul<D1, D2>, N>`
- `UnitVec<D1, N>.dot(UnitVec<D2, N>) → Scalar<DimMul<D1, D2>>` (異種次元対応)
- `UnitVec<D1, 3>.cross(UnitVec<D2, 3>) → UnitVec<DimMul<D1, D2>, 3>` (異種次元対応)

内積・外積は異種次元間でも物理的に意味がある。例:
- 力 (N) と変位 (m) の内積 → エネルギー (J): `UnitVec<Force>.dot(UnitVec<Length>) → Scalar<Energy>`
- 位置 (m) と速度 (m/s) の外積 → 角運動量/質量: `UnitVec<Length>.cross(UnitVec<Velocity>) → UnitVec<DimMul<Length, Velocity>>`

同種次元の場合は `D1 = D2` の特殊ケースとして自然に含まれる。

### 3D フレーム付きベクトル: `FrameVec<F, D>`

座標系フレーム `F` と物理次元 `D` の両方を型パラメータに持つ 3 次元ベクトル。
arika の `Vec3<F>` に単位を追加した上位概念。

```rust
#[repr(transparent)]
pub struct FrameVec<F, D> {
    value: Vector3<f64>,  // nalgebra
    _marker: PhantomData<(F, D)>,
}
```

演算規則:
- `FrameVec<F, D> + FrameVec<F, D> → FrameVec<F, D>` (同フレーム・同次元のみ)
- `FrameVec<F1, D> + FrameVec<F2, D>` → **コンパイルエラー** (異フレーム加算禁止)
- `Scalar<D1> * FrameVec<F, D2> → FrameVec<F, DimMul<D1, D2>>`
- フレーム変換: `Rotation<F1, F2>.transform(FrameVec<F1, D>) → FrameVec<F2, D>`
  (回転は次元を変えない)

フレームマーカーの例:

```rust
pub struct Eci;    // Earth-Centered Inertial
pub struct Ecef;   // Earth-Centered Earth-Fixed
pub struct Body;   // Spacecraft body-fixed
pub struct Rsw;    // Radial / Along-track / Cross-track
```

**arika との関係**: sunpou はフレームマーカーを trait bound として定義し、
arika の具体型 (`SimpleEci`, `Gcrs` 等) をマーカーとして使えるように設計する。
sunpou 自体は arika に依存せず、ユーザーが任意のフレーム型を定義できる。

### 行列: `UnitMat<DR, DC, const R: usize, const C: usize>`

行ごとの次元 `DR` と列ごとの次元 `DC` を持つ行列。行列の要素 `(i, j)` の次元は
`DR[i] / DC[j]` (行の次元を列の次元で割ったもの) として解釈される。

**背景**: 行列 `A` がベクトル `x` に作用して `y = A * x` を計算するとき、
`y[i]` の次元は `A[i][j] * x[j]` の次元の和であり、これが well-defined であるためには
各行 `i` について `dim(A[i][j]) * dim(x[j])` がすべての `j` で同じ次元を返す必要がある。

これは `dim(A[i][j]) = dim(y[i]) / dim(x[j])` を意味する。

#### 同次元行列 (均質行列)

すべての要素が同じ次元を持つ最も単純なケース:

```rust
// 回転行列: 無次元 3×3
type RotationMatrix = UnitMat<Dimensionless, Dimensionless, 3, 3>;

// 慣性テンソル: kg·m² の 3×3
type InertiaTensor = UnitMat<Dim<P2, P1, Z0>, Dimensionless, 3, 3>;
```

#### ブロック行列 (異種次元)

状態遷移行列やヤコビアンなど、ブロックごとに異なる次元を持つ行列を
型レベルで構成する。

**具体例: 軌道力学の状態遷移行列 (STM)**

状態ベクトル `[r, v]` (位置 + 速度) に対する 6×6 状態遷移行列:

```
Φ = | ∂r/∂r₀  ∂r/∂v₀ |   =  | 無次元      時間        |
    | ∂v/∂r₀  ∂v/∂v₀ |      | 時間⁻¹      無次元      |
```

(∂r/∂v₀ の次元: m / (m/s) = s、∂v/∂r₀ の次元: (m/s) / m = s⁻¹)

型レベルでの表現:

```rust
use typenum::consts::*;

/// 状態ベクトルの型: 上3要素が位置 (Length)、下3要素が速度 (Velocity)
type OrbitalState = BlockVec!(
    UnitVec<Length, 3>,
    UnitVec<Velocity, 3>,
);

/// 状態遷移行列: 4ブロックの 6×6 行列
type OrbitalStm = BlockMat!(
    [UnitMat<Dimensionless, Dimensionless, 3, 3>,  UnitMat<Time, Dimensionless, 3, 3>],
    [UnitMat<DimDiv<Velocity, Length>, Dimensionless, 3, 3>, UnitMat<Dimensionless, Dimensionless, 3, 3>],
);

// Φ * x₀ = x  の型検査:
// 上3要素: Dimensionless * Length + Time * Velocity = Length + Length = Length ✓
// 下3要素: (Velocity/Length) * Length + Dimensionless * Velocity = Velocity + Velocity = Velocity ✓
```

**具体例: EKF の共分散行列**

```
P = | P_rr  P_rv |   =  | m²      m·(m/s)   |
    | P_vr  P_vv |      | (m/s)·m  (m/s)²    |
```

```rust
type CovarianceMatrix = BlockMat!(
    [UnitMat<DimMul<Length, Length>, Dimensionless, 3, 3>,
     UnitMat<DimMul<Length, Velocity>, Dimensionless, 3, 3>],
    [UnitMat<DimMul<Velocity, Length>, Dimensionless, 3, 3>,
     UnitMat<DimMul<Velocity, Velocity>, Dimensionless, 3, 3>],
);
```

#### ブロック行列の設計戦略

**基本原則: 数学的・物理的に正当な演算はすべて表現できなければならない。**

制約の実現はジェネリクスと trait bound で行う。マクロはボイラープレート削減のために
提供するが、マクロなしでも同等のことがジェネリクスで書ける。

##### ジェネリックなブロック構造

ブロック行列・ブロックベクトルは、各ブロックの次元とサイズを型パラメータで持つ
ジェネリックな構造として表現する。行列-ベクトル乗算の型検査は trait bound で
「各ブロックの次元が整合する」ことを要求する:

```rust
/// 2×2 ブロック行列 (ジェネリック)
pub struct BlockMat2x2<A, B, C, D> {
    pub a: A,  // 左上
    pub b: B,  // 右上
    pub c: C,  // 左下
    pub d: D,  // 右下
}

/// 2 ブロックベクトル (ジェネリック)
pub struct BlockVec2<U, L> {
    pub upper: U,
    pub lower: L,
}
```

乗算の型制約は trait bound で表現する。各ブロックの乗算結果が加算可能であること
(= 同じ次元を持つこと) をコンパイル時に検証する:

```rust
impl<A, B, C, D, U, L> Mul<BlockVec2<U, L>> for BlockMat2x2<A, B, C, D>
where
    A: Mul<U>,                             // A * U
    B: Mul<L>,                             // B * L
    C: Mul<U>,                             // C * U
    D: Mul<L>,                             // D * L
    <A as Mul<U>>::Output: Add<<B as Mul<L>>::Output>,  // A*U + B*L が可能 (同次元)
    <C as Mul<U>>::Output: Add<<D as Mul<L>>::Output>,  // C*U + D*L が可能 (同次元)
{
    type Output = BlockVec2<
        <<A as Mul<U>>::Output as Add<<B as Mul<L>>::Output>>::Output,
        <<C as Mul<U>>::Output as Add<<D as Mul<L>>::Output>>::Output,
    >;
    // ...
}
```

この設計により:
- **任意の次元の組み合わせ** が型パラメータとして受け入れられる
- **整合しない次元** は trait bound を満たせず、コンパイルエラーになる
- **新しいブロック構造** をユーザーが自由に定義できる (3×3, 4×4, 非正方形, ...)
- マクロ (`BlockMat!`, `BlockVec!`) は `BlockMat2x2` 等のボイラープレートを
  生成するが、ジェネリクスの表現力を制限しない

##### 具体例: 軌道状態遷移行列

```rust
/// 軌道状態遷移行列
type OrbitalStm = BlockMat2x2<
    UnitMat<Dimensionless, Dimensionless, 3, 3>,  // ∂r/∂r₀
    UnitMat<Time, Dimensionless, 3, 3>,            // ∂r/∂v₀
    UnitMat<InvTime, Dimensionless, 3, 3>,         // ∂v/∂r₀
    UnitMat<Dimensionless, Dimensionless, 3, 3>,   // ∂v/∂v₀
>;

/// 軌道状態ベクトル
type OrbitalState = BlockVec2<
    UnitVec<Length, 3>,
    UnitVec<Velocity, 3>,
>;

// stm * state の型検査:
//   上: Dimensionless * Length + Time * Velocity = Length + Length = Length  ✓
//   下: InvTime * Length + Dimensionless * Velocity = Velocity + Velocity  ✓
let x1: OrbitalState = stm * x0;
```

##### 拡張性

同じ原則で任意のブロックサイズに拡張できる。例えば 3×3 ブロック行列
(位置+速度+バイアス推定の EKF) は `BlockMat3x3<A,B,C,D,E,F,G,H,I>` として
同様に定義し、trait bound で制約する。将来的にはリスト型による可変長ブロックも検討する。

### フレーム変換

```rust
pub struct Rotation<From, To> {
    quat: UnitQuaternion<f64>,
    _marker: PhantomData<(From, To)>,
}

impl<F1, F2> Rotation<F1, F2> {
    /// ベクトルをフレーム F1 から F2 に変換する。次元は保存される。
    pub fn transform<D>(&self, v: &FrameVec<F1, D>) -> FrameVec<F2, D> { ... }

    /// 逆変換
    pub fn inverse(&self) -> Rotation<F2, F1> { ... }

    /// 合成: F1→F2 then F2→F3 = F1→F3
    pub fn then<F3>(&self, other: &Rotation<F2, F3>) -> Rotation<F1, F3> { ... }
}
```

### nalgebra / f64 との相互変換

単位付き型と生の数値型との境界を明示的に管理する:

```rust
impl<D> Scalar<D> {
    /// 単位を付与する (unchecked: 呼び出し側が次元の正しさを保証)
    pub fn from_raw_unchecked(value: f64) -> Self { ... }

    /// 単位を剥がして生の f64 を取得
    pub fn into_raw(self) -> f64 { ... }
}

impl<D, const N: usize> UnitVec<D, N> {
    /// nalgebra ベクトルから変換 (unchecked)
    pub fn from_raw_unchecked(value: SVector<f64, N>) -> Self { ... }

    /// nalgebra ベクトルに変換
    pub fn into_raw(self) -> SVector<f64, N> { ... }

    /// 内部参照を取得 (nalgebra の読み取り専用操作に使用)
    pub fn as_raw(&self) -> &SVector<f64, N> { ... }
}
```

**マーカー付き変換** (型安全な境界):

```rust
/// 「この値は SI 基本単位で表現されている」ことを示すマーカー trait
pub trait SiBaseUnit {}

/// SI 基本単位であることを保証した上での変換
impl<D: SiBaseUnit> Scalar<D> {
    pub fn from_si(value: f64) -> Self { ... }
    pub fn to_si(self) -> f64 { ... }
}
```

## ユースケース

### 1. 基本的な物理計算

```rust
use sunpou::prelude::*;

let mass = Scalar::<Mass>::from_si(100.0);        // 100 kg
let accel = Scalar::<Acceleration>::from_si(9.8);  // 9.8 m/s²
let force: Scalar<Force> = mass * accel;           // 980 N ✓

// let bad: Scalar<Force> = mass + accel;  // コンパイルエラー: Mass ≠ Acceleration
```

### 2. フレーム付き軌道計算

```rust
let pos_eci = FrameVec::<Eci, Length>::new(7000e3, 0.0, 0.0);
let vel_eci = FrameVec::<Eci, Velocity>::new(0.0, 7.5e3, 0.0);

// フレーム変換
let rot = Rotation::<Eci, Ecef>::from_angle(era);
let pos_ecef = rot.transform(&pos_eci);  // FrameVec<Ecef, Length>

// フレーム不一致は型エラー
// let bad = pos_eci + pos_ecef;  // コンパイルエラー: Eci ≠ Ecef
```

### 3. 状態遷移行列と EKF

```rust
// 状態遷移行列を作成 (Phase 1: 手動構造体)
let stm = OrbitalStm {
    rr: UnitMat::identity(),
    rv: UnitMat::from_raw_unchecked(dt * Matrix3::identity()),
    vr: UnitMat::from_raw_unchecked(gravity_gradient),
    vv: UnitMat::identity(),
};

// 状態を伝搬
let x0 = OrbitalState { r: pos, v: vel };
let x1 = stm.mul(&x0);  // 型検査: 各ブロックの次元が整合

// 共分散伝搬: P₁ = Φ P₀ Φᵀ + Q
// (行列の転置でも次元が正しく追跡される)
```

### 4. nalgebra ライブラリとの連携

```rust
// 既存の nalgebra コードとの境界
let raw_pos: Vector3<f64> = some_legacy_function();
let pos = FrameVec::<Eci, Length>::from_raw_unchecked(raw_pos);  // 明示的な境界

// nalgebra に戻す
let raw = pos.into_raw();
legacy_function(raw);
```

## 実装ロードマップ

### Phase 1: 基盤 (MVP)

- [ ] `Dim<L, M, T, I, Th, N, J>` 型と次元演算 (`DimMul`, `DimDiv`)
- [ ] `Scalar<D>` と四則演算
- [ ] `UnitVec<D, N>` と基本演算 (加減算、スカラー倍、異種次元内積・外積)
- [ ] `FrameVec<F, D>` と基本演算 (異種次元内積・外積含む)
- [ ] `Rotation<F1, F2>` によるフレーム変換
- [ ] `from_raw_unchecked` / `into_raw` による nalgebra 相互変換
- [ ] 頻出する次元の型エイリアス

### Phase 2: 行列とブロック構造

- [ ] `UnitMat<DR, DC, R, C>` と行列-ベクトル乗算
- [ ] ジェネリックなブロック構造体 (`BlockMat2x2`, `BlockVec2`, ...)
- [ ] ブロック行列の乗算・転置・加算の trait bound による型検査
- [ ] `BlockMat!` / `BlockVec!` ボイラープレート削減マクロ
- [ ] 転置の次元追跡
- [ ] 逆行列の次元追跡
- [ ] 行列-行列乗算 (ジェネリック)

### Phase 3: 実用化

- [ ] arika のフレーム型との統合例
- [ ] s7e での採用例
- [ ] EKF の完全な型付き実装例 (共分散伝搬含む)
- [ ] コンパイルエラーメッセージの改善
- [ ] ドキュメントと examples

### Phase 4: 拡張 (将来)

- [ ] 単位変換 (km ↔ m 等) のサポート
- [ ] 可変長ブロック行列 (型レベルリストによる N×M ブロック)
- [ ] `serde` 対応
- [ ] テンソル (2階以上) への拡張

## 依存クレート

- `typenum` — 型レベル整数演算 (no_std)
- `nalgebra` — 線形代数の内部ストレージ (no_std, `default-features = false`)

## 未決定事項

- [ ] 単位変換 (km, mm 等) をどのレベルで扱うか (Phase 1 では SI 基本単位のみ)
- [ ] `Debug` / `Display` 表示での単位文字列
- [ ] 異種次元ベクトル (位置+速度の6次元状態ベクトルを単一型で表現するか、BlockVec で構造体に分けるか)
- [ ] `PartialEq` / `PartialOrd` の次元チェック方針
- [ ] クォータニオンの単位的な扱い (回転は無次元だが区別したい可能性)
- [ ] N×M ブロック行列の汎用表現 (型レベルリスト vs 固定サイズの BlockMatNxM 系列)
