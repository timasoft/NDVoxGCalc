use criterion::{Criterion, criterion_group, criterion_main};
use hypervox_expr::{VarMap, parse};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct TestVars {
    ndim: usize,
}

impl VarMap for TestVars {
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name {
            "x" => Some(0),
            "y" => Some(usize::from(self.ndim > 1)),
            "z" => Some(if self.ndim > 2 { 2 } else { 0 }),
            _ => None,
        }
    }
    fn primary_prefix(&self) -> &'static str {
        "x"
    }
}

const D3: TestVars = TestVars { ndim: 3 };

const FAST_BENCH_SIDE: usize = 128;
const SLOW_BENCH_SIDE: usize = 64;

#[inline(never)]
const fn direct_simple(x: f64, y: f64, z: f64) -> f64 {
    y.mul_add(z, x)
}

#[inline(never)]
fn direct_medium(x: f64, y: f64, z: f64) -> f64 {
    y.cos().mul_add(z * z, x.sin()) + x.hypot(y)
}

#[inline(never)]
fn direct_heavy(x: f64, y: f64, z: f64) -> f64 {
    (x.sin() * y.cos()).exp()
        + z.mul_add(z, 1.0).ln()
        + (z.mul_add(z, y.mul_add(y, x * x)).sqrt()).atan2(black_box(1.0))
        + (x + y).abs().sqrt()
}

#[inline(never)]
fn direct_repeated(x: f64, y: f64, _z: f64) -> f64 {
    y.mul_add(y, x * x)
        .mul_add(y.mul_add(y, x * x), y.mul_add(y, x * x).sin())
}

#[inline(never)]
fn direct_very_heavy(x: f64, y: f64, z: f64) -> f64 {
    let r2 = z.mul_add(z, y.mul_add(y, x * x));
    let sqrt_r2 = r2.sqrt();
    ((x.sin() * y.cos()).mul_add(
        z.tanh(),
        (z.tanh() * z.tanh()).mul_add(
            z.tanh(),
            (y.cos() * y.cos()).mul_add(y.cos(), x.sin() * x.sin() * x.sin()),
        ),
    ) / (1.0 + (x.sin() * y.cos() * z.tanh()).abs()))
    .mul_add(
        z.tanh()
            .mul_add(
                z.tanh(),
                y.cos().mul_add(y.cos(), x.sin().mul_add(x.sin(), 1.0)),
            )
            .sqrt(),
        (1.0 + r2.sin().mul_add(r2.cos(), -(r2 / (1.0 + r2)).tanh()).abs())
            .cbrt()
            .mul_add(
                ((1.0 + r2).sqrt() / (1.0 + sqrt_r2)).atan(),
                (x.sin() * y.sin())
                    .mul_add(
                        z.sin(),
                        z.sin()
                            .mul_add(x.cos(), y.sin().mul_add(z.cos(), x.sin() * y.cos())),
                    )
                    .abs()
                    .ln_1p()
                    .mul_add(
                        (r2.sin().mul_add(r2.cos(), 1.0 + r2).sqrt()).atan2(
                            1.0 + y
                                .cos()
                                .mul_add(
                                    z.cos(),
                                    x.sin().mul_add(
                                        z.sin(),
                                        z.tanh().mul_add(
                                            z.tanh(),
                                            y.cos().mul_add(y.cos(), x.sin() * x.sin()),
                                        ),
                                    ),
                                )
                                .abs(),
                        ),
                        (-sqrt_r2 / (1.0 + sqrt_r2)).exp()
                            * sqrt_r2
                                .cos()
                                .mul_add((sqrt_r2 / (1.0 + sqrt_r2)).tanh(), sqrt_r2)
                                .sin(),
                    )
                    + (y.mul_add(y, x * x).cos() * z.mul_add(z, y * y).sin())
                        .mul_add(
                            x.mul_add(x, z * z).cos(),
                            y.mul_add(y, x * x).sin()
                                * z.mul_add(z, y * y).cos()
                                * x.mul_add(x, z * z).sin(),
                        )
                        .tanh()
                        / (1.0 + r2.sin().mul_add(r2.sin(), r2.cos()).abs()),
            ),
    )
}

struct Case {
    name: &'static str,
    hx: &'static str,
    ee: &'static str,
    direct: fn(f64, f64, f64) -> f64,
}

const CASES: &[Case] = &[
    Case {
        name: "simple",
        hx: "x + y * z",
        ee: "x + y * z",
        direct: direct_simple,
    },
    Case {
        name: "medium",
        hx: "sin(x) + cos(y) * z^2 + sqrt(x*x + y*y)",
        ee: "math::sin(x) + math::cos(y) * z^2 + math::sqrt(x*x + y*y)",
        direct: direct_medium,
    },
    Case {
        name: "heavy",
        hx: "exp(sin(x) * cos(y)) + ln(z*z + 1) + atan2(sqrt(x*x + y*y + z*z), 1) + sqrt(abs(x + y))",
        ee: "math::exp(math::sin(x) * math::cos(y)) + math::ln(z*z + 1) + math::atan2(math::sqrt(x*x + y*y + z*z), 1) + math::sqrt(math::abs(x + y))",
        direct: direct_heavy,
    },
    Case {
        name: "repeated",
        hx: "(x*x + y*y)*(x*x + y*y) + sin(x*x + y*y)",
        ee: "(x*x + y*y)*(x*x + y*y) + math::sin(x*x + y*y)",
        direct: direct_repeated,
    },
    Case {
        name: "very_heavy",
        hx: concat!(
            "exp(-sqrt(x*x + y*y + z*z) / (1 + sqrt(x*x + y*y + z*z)))",
            " * sin(sqrt(x*x + y*y + z*z) + cos(sqrt(x*x + y*y + z*z))",
            " * tanh(sqrt(x*x + y*y + z*z) / (1 + sqrt(x*x + y*y + z*z)))) + ",
            "ln(1 + abs(sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
            " + sin(x)*sin(y)*sin(z)))",
            " * atan2(sqrt(1 + x*x + y*y + z*z",
            " + sin(x*x + y*y + z*z)*cos(x*x + y*y + z*z)),",
            " 1 + abs(sin(x)^2 + cos(y)^2 + tanh(z)^2",
            " + sin(x)*sin(z) + cos(y)*cos(z))) + ",
            "tanh(sin(x*x + y*y)*cos(y*y + z*z)*sin(z*z + x*x)",
            " + cos(x*x + y*y)*sin(y*y + z*z)*cos(z*z + x*x))",
            " / (1 + abs(cos(x*x + y*y + z*z)",
            " + sin(x*x + y*y + z*z)^2)) + ",
            "cbrt(1 + abs(sin(x*x + y*y + z*z)*cos(x*x + y*y + z*z)",
            " - tanh((x*x + y*y + z*z) / (1 + x*x + y*y + z*z))))",
            " * atan(sqrt(1 + x*x + y*y + z*z)",
            " / (1 + sqrt(x*x + y*y + z*z))) + ",
            "(sin(x)^3 + cos(y)^3 + tanh(z)^3",
            " + sin(x)*cos(y)*tanh(z))",
            " / (1 + abs(sin(x)*cos(y)*tanh(z)))",
            " * sqrt(1 + sin(x)^2 + cos(y)^2 + tanh(z)^2)",
        ),
        ee: concat!(
            "math::exp(-math::sqrt(x*x + y*y + z*z)",
            " / (1 + math::sqrt(x*x + y*y + z*z)))",
            " * math::sin(math::sqrt(x*x + y*y + z*z)",
            " + math::cos(math::sqrt(x*x + y*y + z*z))",
            " * math::tanh(math::sqrt(x*x + y*y + z*z)",
            " / (1 + math::sqrt(x*x + y*y + z*z)))) + ",
            "math::ln(1 + math::abs(math::sin(x)*math::cos(y)",
            " + math::sin(y)*math::cos(z)",
            " + math::sin(z)*math::cos(x)",
            " + math::sin(x)*math::sin(y)*math::sin(z)))",
            " * math::atan2(math::sqrt(1 + x*x + y*y + z*z",
            " + math::sin(x*x + y*y + z*z)",
            " * math::cos(x*x + y*y + z*z)),",
            " 1 + math::abs(math::sin(x)^2 + math::cos(y)^2",
            " + math::tanh(z)^2 + math::sin(x)*math::sin(z)",
            " + math::cos(y)*math::cos(z))) + ",
            "math::tanh(math::sin(x*x + y*y)*math::cos(y*y + z*z)",
            " * math::sin(z*z + x*x)",
            " + math::cos(x*x + y*y)*math::sin(y*y + z*z)",
            " * math::cos(z*z + x*x))",
            " / (1 + math::abs(math::cos(x*x + y*y + z*z)",
            " + math::sin(x*x + y*y + z*z)^2)) + ",
            "math::cbrt(1 + math::abs(math::sin(x*x + y*y + z*z)",
            " * math::cos(x*x + y*y + z*z)",
            " - math::tanh((x*x + y*y + z*z)",
            " / (1 + x*x + y*y + z*z))))",
            " * math::atan(math::sqrt(1 + x*x + y*y + z*z)",
            " / (1 + math::sqrt(x*x + y*y + z*z))) + ",
            "(math::sin(x)^3 + math::cos(y)^3 + math::tanh(z)^3",
            " + math::sin(x)*math::cos(y)*math::tanh(z))",
            " / (1 + math::abs(math::sin(x)*math::cos(y)",
            " * math::tanh(z)))",
            " * math::sqrt(1 + math::sin(x)^2 + math::cos(y)^2",
            " + math::tanh(z)^2)",
        ),
        direct: direct_very_heavy,
    },
];

/// A simple triple-nested loop over a `side³` grid, pushing `x, y, z` through
/// `f` and accumulating the results.  Both inputs and output are `black_box`ed
/// to prevent the compiler from constant-folding the entire computation.
fn run_triple_loop<F>(side: usize, scale: f64, mut f: F) -> f64
where
    F: FnMut(f64, f64, f64) -> f64,
{
    let mut sum = 0.0_f64;
    for nz in 0..side {
        #[expect(clippy::cast_precision_loss)]
        let zv = (nz as f64).mul_add(scale, -5.0);
        for ny in 0..side {
            #[expect(clippy::cast_precision_loss)]
            let yv = (ny as f64).mul_add(scale, -5.0);
            for nx in 0..side {
                #[expect(clippy::cast_precision_loss)]
                let xv = (nx as f64).mul_add(scale, -5.0);
                sum += black_box(f(black_box(xv), black_box(yv), black_box(zv)));
            }
        }
    }
    black_box(sum)
}

fn bench_compile_time(c: &mut Criterion, name: &str, expr_src: &str) {
    let mut group = c.benchmark_group(format!("compile/{name}"));

    group.bench_function("parse", |b| {
        b.iter_batched(
            || (),
            |()| black_box(parse(expr_src, &D3).expect("expression should be valid")),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("pre_eval", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).expect("expression should be valid"),
            |mut node| {
                node.pre_eval(&[]);
                black_box(node)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("cse", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).expect("expression should be valid");
                node.pre_eval(&[]);
                node
            },
            |mut node| {
                node.cse();
                black_box(node)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("compile", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).expect("expression should be valid");
                node.pre_eval(&[]);
                node
            },
            |node| black_box(node.compile()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("compile_multi", |b| {
        b.iter_batched(
            || {
                let mut node = parse(expr_src, &D3).expect("expression should be valid");
                node.pre_eval(&[]);
                node
            },
            |mut node| black_box(node.compile_multi(&[0, 1, 2])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("prepare", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).expect("expression should be valid"),
            |mut node| black_box(node.prepare(&[])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("prepare_multi", |b| {
        b.iter_batched(
            || parse(expr_src, &D3).expect("expression should be valid"),
            |mut node| black_box(node.prepare_multi(&[], &[0, 1, 2])),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_expression(
    c: &mut Criterion,
    name: &str,
    expr_src: &str,
    direct_fn: fn(f64, f64, f64) -> f64,
) {
    let side = FAST_BENCH_SIDE;
    #[expect(clippy::cast_precision_loss)]
    let scale = 10.0_f64 / side as f64;

    let mut node_flat = parse(expr_src, &D3).expect("parse failed");
    node_flat.pre_eval(&[]);
    let flat_expr = node_flat.compile();

    let mut node_cse = parse(expr_src, &D3).expect("parse failed");
    let (cse_expr, cse_slots) = node_cse.prepare(&[]);

    let mut node_multi = parse(expr_src, &D3).expect("parse failed");
    let multi = node_multi.prepare_multi(&[], &[0, 1, 2]);

    // Sanity check
    {
        let vars = [1.5_f64, 2.5_f64, -3.0_f64];
        let mut cache = vec![0.0_f64; multi.cse_slots];
        for g in &multi.groups {
            (g.combined)(&vars, &mut cache);
        }
        let compiled_val = (multi.main)(&vars, &mut cache);
        let direct_val = direct_fn(vars[0], vars[1], vars[2]);
        assert!(
            (compiled_val - direct_val).abs() < 1e-12_f64,
            "sanity fail for {name}: compiled={compiled_val}, direct={direct_val}"
        );
    }

    let mut group = c.benchmark_group(name);
    group.sample_size(25);

    group.bench_function("direct", |b| {
        b.iter(|| run_triple_loop(side, scale, direct_fn));
    });

    let mut cache_flat = vec![0.0_f64; 0];
    group.bench_function("flat", |b| {
        let cache = &mut cache_flat;
        b.iter(|| run_triple_loop(side, scale, |x, y, z| flat_expr(&[x, y, z], cache)));
    });

    let mut cache_cse = vec![0.0_f64; cse_slots];
    group.bench_function("cse", |b| {
        let cache = &mut cache_cse;
        b.iter(|| run_triple_loop(side, scale, |x, y, z| cse_expr(&[x, y, z], cache)));
    });

    let mut cache_multi = vec![0.0_f64; multi.cse_slots];
    group.bench_function("multi", |b| {
        let cache = &mut cache_multi;
        b.iter(|| {
            let mut sum = 0.0_f64;
            let mut vars = [0.0_f64; 3];

            #[expect(clippy::cast_precision_loss)]
            for nz in 0..side {
                vars[2] = black_box((nz as f64).mul_add(scale, -5.0));
                for g in &multi.groups {
                    if g.level == 2 {
                        (g.combined)(&vars, cache);
                    }
                }

                for ny in 0..side {
                    vars[1] = black_box((ny as f64).mul_add(scale, -5.0));
                    for g in &multi.groups {
                        if g.level == 1 {
                            (g.combined)(&vars, cache);
                        }
                    }

                    for nx in 0..side {
                        vars[0] = black_box((nx as f64).mul_add(scale, -5.0));
                        sum += black_box((multi.main)(&vars, cache));
                    }
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn register_fast(c: &mut Criterion) {
    for case in CASES {
        bench_expression(c, case.name, case.hx, case.direct);
    }
}

fn register_compile(c: &mut Criterion) {
    for case in CASES {
        bench_compile_time(c, case.name, case.hx);
    }
}

#[cfg_attr(not(feature = "slow-benches"), expect(dead_code))]
mod evalexpr_bench {
    use super::{CASES, Criterion, D3, SLOW_BENCH_SIDE, black_box, parse, run_triple_loop};
    use evalexpr::{
        ContextWithMutableVariables as _, DefaultNumericTypes, HashMapContext, Value,
        build_operator_tree,
    };

    fn bench_vs_evalexpr(
        c: &mut Criterion,
        name: &str,
        hx_expr: &str,
        ee_expr: &str,
        direct_fn: fn(f64, f64, f64) -> f64,
    ) {
        let side = SLOW_BENCH_SIDE;
        #[expect(clippy::cast_precision_loss)]
        let scale = 10.0_f64 / side as f64;

        let mut node = parse(hx_expr, &D3).expect("parse failed");
        let (compiled, slots) = node.prepare(&[]);
        let mut cache = vec![0.0_f64; slots];

        let ee_node = build_operator_tree::<DefaultNumericTypes>(ee_expr).expect("evalexpr parse");

        {
            let v = [1.5_f64, 2.5_f64, -3.0_f64];
            let hx = compiled(&v, &mut cache);

            let mut ctx = HashMapContext::new();
            ctx.set_value("x".into(), Value::from_float(v[0]))
                .expect("should not fail");
            ctx.set_value("y".into(), Value::from_float(v[1]))
                .expect("should not fail");
            ctx.set_value("z".into(), Value::from_float(v[2]))
                .expect("should not fail");
            let ee = ee_node
                .eval_with_context(&ctx)
                .expect("eval should not fail")
                .as_number()
                .expect("convertion should not fail");

            let expected = direct_fn(v[0], v[1], v[2]);
            assert!(
                (hx - expected).abs() < 1e-12_f64,
                "HX sanity fail for {name}"
            );
            assert!(
                (ee - expected).abs() < 1e-6_f64,
                "EE sanity fail for {name}"
            );
        }

        let mut group = c.benchmark_group(format!("vs_evalexpr/{name}"));
        group.sample_size(25);

        group.bench_function("hypervox", |b| {
            b.iter(|| run_triple_loop(side, scale, |x, y, z| compiled(&[x, y, z], &mut cache)));
        });

        group.bench_function("evalexpr", |b| {
            let mut ctx = HashMapContext::new();
            b.iter(|| {
                let mut sum = 0.0_f64;
                for nz in 0..side {
                    for ny in 0..side {
                        for nx in 0..side {
                            #[expect(clippy::cast_precision_loss)]
                            let xv = black_box((nx as f64).mul_add(scale, -5.0));
                            #[expect(clippy::cast_precision_loss)]
                            let yv = black_box((ny as f64).mul_add(scale, -5.0));
                            #[expect(clippy::cast_precision_loss)]
                            let zv = black_box((nz as f64).mul_add(scale, -5.0));
                            ctx.set_value("x".into(), Value::from_float(xv))
                                .expect("should not fail");
                            ctx.set_value("y".into(), Value::from_float(yv))
                                .expect("should not fail");
                            ctx.set_value("z".into(), Value::from_float(zv))
                                .expect("should not fail");
                            sum += black_box(
                                ee_node
                                    .eval_with_context_mut(&mut ctx)
                                    .expect("eval should not fail")
                                    .as_number()
                                    .expect("convertion should not fail"),
                            );
                        }
                    }
                }
                black_box(sum)
            });
        });

        group.finish();
    }

    pub fn register(c: &mut Criterion) {
        for case in CASES {
            bench_vs_evalexpr(c, case.name, case.hx, case.ee, case.direct);
        }
    }
}

criterion_group!(fast_benches, register_fast, register_compile);

#[cfg(feature = "slow-benches")]
criterion_group!(slow_benches, evalexpr_bench::register);

#[cfg(feature = "slow-benches")]
criterion_main!(fast_benches, slow_benches);

#[cfg(not(feature = "slow-benches"))]
criterion_main!(fast_benches);
