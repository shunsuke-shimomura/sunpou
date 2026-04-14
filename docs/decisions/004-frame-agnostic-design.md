# ADR-004: フレームマーカーはユーザー定義の ZST

## 判断内容

`FrameVec<F, D>` と `Rotation<F1, F2>` のフレーム型パラメータ `F` は、
ユーザーが自由に定義するゼロサイズ型 (ZST) とする。uolgebra はフレームの
具体型を提供せず、trait bound も課さない。

## 背景

arika では `SimpleEci`, `Gcrs`, `Ecef` 等の具体的なフレームマーカーと
sealed trait (`Eci`, `Ecef` 等) を提供している。uolgebra がこれらに
依存するか、独自に定義するか、あるいは完全にユーザーに委ねるかを決める必要がある。

## 選択肢

### A. 完全にユーザー定義 (採用)

```rust
// ユーザーコード
struct Eci;
struct Ecef;
let v = FrameVec::<Eci, Length>::new(1.0, 0.0, 0.0);
```

- **Pros**: arika に依存しない。任意のフレームライブラリと組み合わせ可能。
  arika の具体型をそのまま `F` として使える。最も柔軟。
- **Cons**: フレーム間の変換パス (ECI→ECEF→Body 等) をライブラリ側で検証できない。

### B. arika に依存

- **Pros**: arika のフレーム型と sealed trait をそのまま活用。型安全性が高い。
- **Cons**: arika への強依存。他のフレームライブラリと組み合わせ不可。
  no_std 対応に影響する可能性。

### C. uolgebra 独自の Frame trait を定義

```rust
pub trait Frame: 'static {}
```

- **Pros**: 最低限の contract を定義できる。
- **Cons**: 実質的に `struct Eci;` に `impl Frame for Eci {}` を書かせるだけの
  ボイラープレート。現時点で trait に有用なメソッドがない。

## 結論

A を採用。フレームマーカーに trait bound を課す利点が現時点でないため、
ユーザーの任意の ZST を受け入れる。arika との統合は、arika の具体型を
そのまま `F` として渡すことで自然に実現できる。
将来 Frame trait が必要になった場合は後方互換的に追加可能。
