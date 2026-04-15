# sunpou の型エラーの読み方

## 問題

sunpou は typenum を使ってSI次元をコンパイル時に表現するため、
型エラーに `PInt<UInt<UTerm, B1>>` のような暗号的な型名が出ます。

## typenum の型名の読み方

| typenum 型 | 値 | sunpou エイリアス |
|---|---|---|
| `Z0` | 0 | (次元の指数が0) |
| `PInt<UInt<UTerm, B1>>` | +1 | P1 |
| `PInt<UInt<UInt<UTerm, B1>, B0>>` | +2 | P2 |
| `PInt<UInt<UInt<UTerm, B1>, B1>>` | +3 | P3 |
| `NInt<UInt<UTerm, B1>>` | -1 | N1 |
| `NInt<UInt<UInt<UTerm, B1>, B0>>` | -2 | N2 |

## 代表的な次元の型

| 次元 | `Dim<L, M, T>` | typenum 表現 |
|------|----------------|-------------|
| Length | `Dim<P1, Z0, Z0>` | `Dim<PInt<UInt<UTerm, B1>>>` |
| Mass | `Dim<Z0, P1, Z0>` | `Dim<Z0, PInt<UInt<UTerm, B1>>>` |
| Time | `Dim<Z0, Z0, P1>` | `Dim<Z0, Z0, PInt<UInt<UTerm, B1>>>` |
| Velocity | `Dim<P1, Z0, N1>` | `Dim<PInt<...>, Z0, NInt<...>>` |
| Force | `Dim<P1, P1, N2>` | `Dim<PInt<...>, PInt<...>, NInt<...>>` |

## エラーメッセージの例

```
error[E0308]: mismatched types
  expected `Scalar<Dim<PInt<UInt<UTerm, B1>>>>`,
     found `Scalar<Dim<Z0, PInt<UInt<UTerm, B1>>>>`
```

これは「`Scalar<Length>` を期待したが `Scalar<Mass>` が渡された」という意味です。

## ヒント

- `Dim<PInt<...>>` = L指数が正 → Length 関連
- `Dim<Z0, PInt<...>>` = M指数が正 → Mass 関連
- `Dim<..., ..., NInt<...>>` = T指数が負 → 時間の逆数 (速度、加速度等)
- 明示的に型アノテーションを付けると、エラー箇所が特定しやすくなります
