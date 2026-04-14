# ADR-001: SI次元の型レベル表現

## 判断内容

SI次元を `Dim<L, M, T, I, Th, N, J>` として typenum の整数型パラメータで表現する。
乗算は指数の加算 (`DimMul`)、除算は指数の減算 (`DimDiv`) として trait で実装する。

## 背景

コンパイル時に物理次元の整合性を検証するために、次元を型レベルで表現する必要がある。

## 選択肢

### A. typenum による整数指数 (採用)

```rust
pub struct Dim<L, M, T, I = Z0, Th = Z0, N = Z0, J = Z0>;
```

- **Pros**: 乗除算が typenum の Add/Sub で自然に表現できる。uom と同じアプローチで実績がある。
  7次元すべてを表現可能。
- **Cons**: 型名が長くなる（`Dim<PInt<UInt<UTerm, B1>>, Z0, NInt<...>>` 等）。
  コンパイルエラーメッセージが読みにくい。

### B. マーカー trait ベース

各次元を個別の trait で表現:
```rust
trait HasLength { type L: Integer; }
```

- **Pros**: 拡張性が高い。
- **Cons**: 7つの trait を束ねるのが煩雑。演算の実装が複雑。

### C. const generics (i8 指数)

```rust
pub struct Dim<const L: i8, const M: i8, const T: i8, ...>;
```

- **Pros**: 型名が直感的 (`Dim<1, 0, -1>`)。
- **Cons**: Rust の const generics ではまだ `const L: i8` の算術がコンパイル時に
  型レベルで検証できない（`where` 節で const 式の等価性を要求できない）。
  将来的には可能になるかもしれないが、現時点では typenum が安定。

## 結論

typenum を採用。コンパイルエラーの可読性は型エイリアスで緩和する。
将来 const generics が成熟したら移行を検討する。
