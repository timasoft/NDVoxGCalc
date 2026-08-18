# hypervox_expr

[![crates.io](https://img.shields.io/crates/v/hypervox_expr.svg)](https://crates.io/crates/hypervox_expr)
[![Documentation](https://docs.rs/hypervox_expr/badge.svg)](https://docs.rs/hypervox_expr)

High-performance mathematical expression parser, AST optimizer, and closure compiler for N-dimensional evaluation.

## Features

- **Pratt parser** -- parses string expressions into an AST with standard operator precedence and right-associative `^`/`**`
- **Constant folding and algebraic simplification** (`pre_eval`) -- evaluates constant sub-expressions, identity elimination, zero propagation, negation rewrites, inverse composition, constant reassociation
- **Common subexpression elimination** (`cse`) -- extracts repeated subtrees into shared slots evaluated once
- **Closure compilation** (`compile`) -- emits fused MulAdd/MulSub/NegMulAdd/NegMul patterns when detected
- **Multi-level invariant hoisting** (`compile_multi`) -- extracts dimension-invariant sub-expressions into hierarchical pre-computed closures for efficient N-dimensional grid evaluation
- **Extensible** -- add custom constants and functions via `ExtF0`/`ExtF1`/`ExtF2` traits and `define_ext_f{0,1,2}!` macros
- **Variable resolution** -- implement `VarMap` to define spatial aliases (`x`, `y`, `z`) and indexed variables (`x0`, `x1`, ...)

## Built-in functions

- **Constants**: `PI`, `E`
- **Unary**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sqrt`, `cbrt`, `exp`, `ln`, `log10`, `log2`, `floor`, `ceil`, `round`, `trunc`, `abs`
- **Binary**: `atan2`, `pow`
- **Operators**: `+`, `-`, `*`, `/`, `^`/`**` (power, right-associative), `%` (modulo), `|x|` (pipe-abs)

`0^0` evaluates to `1`.

## Quick start

```rust
use hypervox_expr::{parse, VarMap};

struct MyVars;
impl VarMap for MyVars {
    fn ndim(&self) -> usize { 3 }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None }
    }
    fn primary_prefix(&self) -> &str { "x" }
}

let mut node = parse("sin(x) * cos(y) + z", &MyVars).unwrap();
let (f, slots) = node.prepare(&[]);
let mut cache = vec![0.0; slots];
let result = f(&[1.0, 2.0, 3.0], &mut cache);
```

## Multi-level evaluation

```rust
let mut node = parse("cos(z)*tan(y) + x", &MyVars).unwrap();
let multi = node.prepare_multi(&[], &[0, 1, 2]);
let mut cache = vec![0.0; multi.cse_slots];
for g in &multi.groups {
    (g.combined)(&[1.0, 2.0, 3.0], &mut cache);
}
let result = (multi.main)(&[1.0, 2.0, 3.0], &mut cache);
```

## Extending with custom functions

```rust
use hypervox_expr::{parse_with_ext, NoExtF};
hypervox_expr::define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);

let mut node = parse_with_ext::<_, NoExtF, MyF1, NoExtF>("cube(x)", &MyVars).unwrap();
let (f, slots) = node.prepare(&[]);
assert_eq!(f(&[3.0], &mut vec![0.0; slots]), 27.0);
```

## Performance

The `compile` strategy emits a single flat closure; `compile_multi` hoists
dimension-invariant sub-expressions into pre-computed groups, trading longer
compile time for faster evaluation on large grids:

| Benchmark  | Parse   | Pre-eval | Compile | Compile multi | CSE     | Prepare | Prepare multi |
|------------|---------|----------|---------|---------------|---------|---------|---------------|
| simple     |  0.48µs | 0.03µs   | 0.11µs  |   0.77µs      |  0.07µs |  0.19µs |   0.86µs      |
| medium     |  1.52µs | 0.15µs   | 0.47µs  |   2.27µs      |  0.39µs |  1.12µs |   3.01µs      |
| heavy      |  2.98µs | 0.24µs   | 0.98µs  |   4.72µs      |  1.50µs |  2.90µs |   7.70µs      |
| repeated   |  1.95µs | 0.14µs   | 0.61µs  |   2.83µs      |  1.16µs |  1.96µs |   4.24µs      |
| very_heavy | 31.58µs | 2.77µs   | 9.80µs  | 139.85µs      | 69.93µs | 89.66µs | 260.45µs      |

Runtime evaluation on a 128^3 grid comparing compilation strategies:

| Benchmark  | direct  | flat      | cse       | multi     |
|------------|---------|-----------|-----------|-----------|
| simple     |   7.9ms |     9.6ms |     9.8ms |     8.2ms |
| medium     |  67.3ms |    81.7ms |    81.5ms |    47.2ms |
| heavy      | 121.3ms |   186.6ms |   190.8ms |   130.6ms |
| repeated   |  33.9ms |    70.2ms |    68.2ms |    56.4ms |
| very_heavy | 783.5ms | 2,594.6ms | 1,810.4ms | 1,561.7ms |

vs [`evalexpr`](https://crates.io/crates/evalexpr) on a 64^3 grid:

| Benchmark  | hypervox_expr | evalexpr  | speedup  |
|------------|---------------|-----------|----------|
| simple     |   1.2ms       |   104.9ms | **~87x** |
| medium     |  10.2ms       |   296.7ms | **~29x** |
| heavy      |  24.2ms       |   623.2ms | **~26x** |
| repeated   |   8.7ms       |   378.5ms | **~44x** |
| very_heavy | 236.3ms       | 6,562.0ms | **~28x** |

Measurements from [`criterion`] benchmarks on GitHub Actions
(ubuntu-latest) at commit `33631f6`, run with the `fma` cargo feature
and `RUSTFLAGS="-C target-feature=+fma"`. [View live dashboard][bencher]

[`criterion`]: https://github.com/criterion-rs/criterion.rs
[bencher]: https://bencher.dev/perf/hypervox-expr

## Cargo features

- `fma` -- compiles the fused `a*b+c` (MulAdd/MulSub/NegMulAdd/NegMul) patterns to `f64::mul_add`. Only beneficial when the target also enables the `+fma` target feature (e.g. `-C target-feature=+fma` in `.cargo/config.toml` for native x86-64/ARM) (disabled by default).
- `slow-benches` -- enables comparison with `evalexpr` in benchmarks (disabled by default)

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/license/mit)

at your option.
