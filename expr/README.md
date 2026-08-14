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
| simple     |  0.65µs | 0.02µs   | 0.11µs  |   0.74µs      |  0.05µs |  0.19µs |   0.81µs      |
| medium     |  2.05µs | 0.15µs   | 0.47µs  |   2.14µs      |  0.38µs |  1.09µs |   2.77µs      |
| heavy      |  4.07µs | 0.28µs   | 0.92µs  |   4.45µs      |  1.54µs |  2.91µs |   6.79µs      |
| repeated   |  2.66µs | 0.14µs   | 0.60µs  |   2.70µs      |  1.13µs |  1.91µs |   3.92µs      |
| very_heavy | 44.02µs | 2.74µs   | 9.49µs  | 124.68µs      | 54.88µs | 72.64µs | 231.65µs      |

Runtime evaluation on a 128^3 grid comparing compilation strategies:

| Benchmark  | direct  | flat      | cse       | multi     |
|------------|---------|-----------|-----------|-----------|
| simple     |   9.0ms |     9.8ms |     9.8ms |     9.0ms |
| medium     |  66.5ms |    76.2ms |    76.1ms |    50.2ms |
| heavy      | 119.8ms |   184.0ms |   186.6ms |   130.6ms |
| repeated   |  30.9ms |    63.1ms |    61.8ms |    61.4ms |
| very_heavy | 830.5ms | 2,508.1ms | 1,833.7ms | 1,497.0ms |

vs [`evalexpr`](https://crates.io/crates/evalexpr) on a 64^3 grid:

| Benchmark  | hypervox_expr | evalexpr  | speedup  |
|------------|---------------|-----------|----------|
| simple     |   1.2ms       |   100.8ms | **~84x** |
| medium     |  10.7ms       |   287.3ms | **~27x** |
| heavy      |  24.0ms       |   583.2ms | **~24x** |
| repeated   |   7.6ms       |   385.3ms | **~51x** |
| very_heavy | 233.5ms       | 6,281.2ms | **~27x** |

Measurements from [`criterion`] benchmarks on GitHub Actions
(ubuntu-latest) at commit `bd3acb2`, run with the `fma` cargo feature
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
