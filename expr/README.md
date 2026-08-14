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
| simple     |  0.51µs | 0.03µs   | 0.11µs  |   0.70µs      |  0.07µs |  0.22µs |   0.81µs      |
| medium     |  1.58µs | 0.15µs   | 0.47µs  |   2.16µs      |  0.42µs |  1.15µs |   2.89µs      |
| heavy      |  3.02µs | 0.24µs   | 1.02µs  |   4.55µs      |  1.54µs |  3.00µs |   7.38µs      |
| repeated   |  2.05µs | 0.15µs   | 0.59µs  |   2.70µs      |  1.13µs |  1.95µs |   4.02µs      |
| very_heavy | 33.62µs | 2.79µs   | 9.65µs  | 134.72µs      | 68.95µs | 87.15µs | 243.82µs      |

Runtime evaluation on a 128^3 grid comparing compilation strategies:

| Benchmark  | direct  | flat      | cse       | multi     |
|------------|---------|-----------|-----------|-----------|
| simple     |   8.0ms |    11.9ms |    12.0ms |     8.4ms |
| medium     |  52.8ms |    83.3ms |    84.2ms |    48.1ms |
| heavy      | 122.8ms |   189.7ms |   190.7ms |   140.7ms |
| repeated   |  35.1ms |    76.7ms |    68.0ms |    58.6ms |
| very_heavy | 729.8ms | 2,755.1ms | 1,831.1ms | 1,562.2ms |

vs [`evalexpr`](https://crates.io/crates/evalexpr) on a 64^3 grid:

| Benchmark  | hypervox_expr | evalexpr  | speedup  |
|------------|---------------|-----------|----------|
| simple     |   1.5ms       |   100.4ms | **~67x** |
| medium     |  10.4ms       |   298.7ms | **~29x** |
| heavy      |  24.2ms       |   616.2ms | **~25x** |
| repeated   |   8.3ms       |   384.6ms | **~46x** |
| very_heavy | 236.2ms       | 6,800.4ms | **~29x** |

Measurements from [`criterion`] benchmarks on GitHub Actions
(ubuntu-latest) at commit `3fe33ad`. [View live dashboard][bencher]

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
