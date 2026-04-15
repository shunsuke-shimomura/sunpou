# sunpou 開発方針

## テスト方針

本プロジェクトではテストを最重視する。すべての実装は以下のテスト基準を満たすこと。

### 1. nalgebra とのクロスバリデーション

すべての線形代数演算 (加減算、スカラー倍、内積、外積、行列-ベクトル乗算、
行列-行列乗算、転置、逆行列等) について、同じ数値を nalgebra の生の型で計算した
結果と sunpou の単位付き型で計算した結果が完全に一致することを検証する。

```rust
// 例: 内積のクロスバリデーション
let raw_a = Vector3::new(1.0, 2.0, 3.0);
let raw_b = Vector3::new(4.0, 5.0, 6.0);
let expected = raw_a.dot(&raw_b);

let a = UnitVec::<Length, 3>::from_raw_unchecked(raw_a);
let b = UnitVec::<Velocity, 3>::from_raw_unchecked(raw_b);
let result: Scalar<DimMul<Length, Velocity>> = a.dot(&b);
assert_eq!(result.into_raw(), expected);
```

### 2. ランタイム性能の検証

型レベルの単位追跡はゼロコスト抽象化であることを保証する。

- **メモリ**: `Scalar<D>` は `f64` と、`UnitVec<D, N>` は `SVector<f64, N>` と、
  `FrameVec<F, D>` は `Vector3<f64>` と、それぞれ `size_of` / `align_of` が
  同一であることをコンパイル時アサーション (`const_assert!`) で保証する
- **計算時間**: ベンチマーク (`criterion`) で nalgebra の生の演算と sunpou の
  単位付き演算の実行時間を比較し、有意な劣化がないことを確認する
- `#[repr(transparent)]` を全構造体に適用し、ABI レベルでの等価性を保証する

### 3. uom とのクロスバリデーション

単位次元の演算 (乗算、除算) の正しさを uom の型システムと照合する。
uom は dev-dependency として導入し、以下を検証する:

- 乗算の次元: `Length * Time = ?` の結果が uom の型と一致すること
- 除算の次元: `Length / Time = Velocity` が uom の型と一致すること
- 加算の次元制約: 同次元のみ加算可能であることをコンパイルテスト (trybuild) で検証

```rust
// 例: uom との次元クロスバリデーション
use uom::si::f64::{Length as UomLength, Velocity as UomVelocity, Time as UomTime};

let uom_len = UomLength::new::<meter>(10.0);
let uom_time = UomTime::new::<second>(2.0);
let uom_vel: UomVelocity = uom_len / uom_time;

let our_len = Scalar::<Length>::from_raw_unchecked(10.0);
let our_time = Scalar::<Time>::from_raw_unchecked(2.0);
let our_vel: Scalar<Velocity> = our_len / our_time;

assert_eq!(uom_vel.get::<meter_per_second>(), our_vel.into_raw());
```

### 4. Example による使用法の提示

`examples/` ディレクトリに実践的なサンプルコードを配置する。各 Phase の完了時に
対応する example を追加すること:

- `examples/basic_scalar.rs` — スカラーの単位付き計算
- `examples/vectors.rs` — ベクトル演算 (内積・外積・異種次元)
- `examples/frame_transform.rs` — フレーム付きベクトルと座標変換
- `examples/orbital_stm.rs` — 軌道状態遷移行列 (ブロック行列)
- `examples/ekf.rs` — EKF での型付き共分散伝搬

Example は `cargo run --example <name>` で実行可能であり、コメントで
物理的な意味と型の役割を説明すること。

### 5. コンパイルテスト (trybuild)

「コンパイルできてはいけないコード」を trybuild で検証する:

- 異なる次元同士の加算がコンパイルエラーになること
- 異なるフレーム同士の加算がコンパイルエラーになること
- ブロック行列の次元不整合がコンパイルエラーになること

## 開発プロセス

### 6. Issue 駆動開発

すべての実装タスクは GitHub Issue として起票し、対応する PR で実装する。

- Issue には目的、スコープ、受け入れ条件 (テスト基準) を明記する
- PR は対応する Issue を参照し、レビュー可能な単位で分割する
- CI ですべてのテスト (単体テスト、クロスバリデーション、コンパイルテスト、
  ベンチマーク、example のビルド) が通ることを確認してからマージする

### 7. 設計判断の記録

設計判断を行った箇所については、`docs/decisions/` ディレクトリに ADR
(Architecture Decision Record) として記録する:

- **判断内容**: 何を決めたか
- **背景**: なぜその判断が必要だったか
- **選択肢**: 検討した alternative とその pros/cons
- **結論**: 選択した理由

ファイル名は `docs/decisions/NNN-<topic>.md` の形式とする。

例:
- `docs/decisions/001-dim-representation.md` — 次元の型表現方法
- `docs/decisions/002-block-matrix-strategy.md` — ブロック行列の実装戦略

### 8. 完了報告

すべての実装タスク (Phase 1〜3) が完了し、以下の条件を満たした時点で完了報告を行う:

- 全テストが通ること
- nalgebra クロスバリデーションが完了していること
- uom クロスバリデーションが完了していること
- ランタイム性能ベンチマークで劣化がないこと
- 全 example が実行可能であること
- 設計判断が docs に記録されていること

完了までの間、こちらへの確認は不要。自律的に設計判断を行い、記録すること。
完成後にフィードバックを提供する。
