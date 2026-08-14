#![expect(clippy::indexing_slicing)]

use hypervox_expr::*;

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

#[test]
fn eval_basic_ops() {
    let dim = D3;
    let e = |s: &str| {
        parse(s, &dim)
            .expect("expression should be valid")
            .compile()(&[], &mut [])
    };
    assert_eq!(e("3 + 5"), 8.0_f64);
    assert_eq!(e("10 - 7"), 3.0_f64);
    assert_eq!(e("4 * 6"), 24.0_f64);
    assert_eq!(e("15 / 3"), 5.0_f64);
    assert_eq!(e("2 ^ 3"), 8.0_f64);
    assert_eq!(e("--5"), 5.0_f64);
    assert_eq!(e("(1 + 2) * 3"), 9.0_f64);
    assert_eq!(e("2 ** 3"), 8.0_f64);
}

#[test]
fn eval_vars_funcs_consts() {
    let dim = D3;
    let e = |s: &str, v: &[f64]| {
        parse(s, &dim)
            .expect("expression should be valid")
            .compile()(v, &mut [])
    };
    assert_eq!(e("x + y + z", &[1.0_f64, 2.0_f64, 3.0_f64]), 6.0_f64);
    assert_eq!(e("2 * x", &[5.0_f64, 0.0_f64, 0.0_f64]), 10.0_f64);
    assert_eq!(e("sqrt(4)", &[]), 2.0_f64);
    assert_eq!(e("abs(-3)", &[]), 3.0_f64);
    assert_eq!(e("atan2(1, 0)", &[]), std::f64::consts::FRAC_PI_2);
    assert_eq!(e("pow(2, 3)", &[]), 8.0_f64);
    assert_eq!(e("pow(0, 0)", &[]), 1.0_f64);
    assert_eq!(e("PI", &[]), std::f64::consts::PI);
    assert_eq!(e("E", &[]), std::f64::consts::E);
}

#[test]
fn pre_eval_simplify() {
    let dim = D3;
    let pe = |s: &str, v: &[Option<f64>]| -> Node {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(v);
        n
    };

    // Full constant folding
    assert_eq!(pe("3 + 5", &[]), Node::Num(8.0_f64));
    assert_eq!(pe("4 * 6", &[]), Node::Num(24.0_f64));
    assert_eq!(pe("2 ^ 3", &[]), Node::Num(8.0_f64));

    // Identity / zero elimination
    assert_eq!(pe("0 + x", &[]), Node::Var(0));
    assert_eq!(pe("x + 0", &[]), Node::Var(0));
    assert_eq!(pe("x - 0", &[]), Node::Var(0));
    assert_eq!(pe("1 * x", &[]), Node::Var(0));
    assert_eq!(pe("x * 1", &[]), Node::Var(0));
    assert_eq!(pe("x / 1", &[]), Node::Var(0));
    assert_eq!(pe("x ^ 1", &[]), Node::Var(0));
    assert_eq!(pe("0 * x", &[]), Node::Num(0.0_f64));
    assert_eq!(pe("x * 0", &[]), Node::Num(0.0_f64));
    assert_eq!(pe("0 / x", &[]), Node::Num(0.0_f64));
    assert_eq!(pe("x ^ 0", &[]), Node::Num(1.0_f64));
    assert_eq!(pe("1 ^ x", &[]), Node::Num(1.0_f64));
    assert_eq!(pe("0 - x", &[]), Node::Neg(Box::new(Node::Var(0))));

    // Negation rewrites via -1 factor
    assert_eq!(pe("-1 * x", &[]), Node::Neg(Box::new(Node::Var(0))));
    assert_eq!(pe("x * -1", &[]), Node::Neg(Box::new(Node::Var(0))));
    assert_eq!(pe("x / -1", &[]), Node::Neg(Box::new(Node::Var(0))));

    // (-a) + (-b) = -(a + b)
    assert_eq!(
        pe("-x + -y", &[]),
        Node::Neg(Box::new(Node::Add(
            Box::new(Node::Var(0)),
            Box::new(Node::Var(1)),
        )))
    );

    // a - (-b) = a + b
    assert_eq!(
        pe("x - -y", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Var(1)))
    );

    // (-a) * (-b) = a * b
    assert_eq!(
        pe("-x * -y", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Var(1)))
    );

    // (-a) / (-b) = a / b
    assert_eq!(
        pe("-x / -y", &[]),
        Node::Div(Box::new(Node::Var(0)), Box::new(Node::Var(1)))
    );

    // ln(exp(x)) = x
    assert_eq!(pe("ln(exp(x))", &[]), Node::Var(0));

    // idempotent: abs(abs(x)) = abs(x)
    assert_eq!(
        pe("abs(abs(x))", &[]),
        Node::F1(F1::Abs, Box::new(Node::Var(0)))
    );

    // Partial substitution: x + y with x=2
    assert_eq!(
        pe("x + y", &[Some(2.0_f64), None, None]),
        Node::Add(Box::new(Node::Num(2.0_f64)), Box::new(Node::Var(1)))
    );

    // 0^0 = 1 through pre_eval
    assert_eq!(pe("0^0", &[]), Node::Num(1.0_f64));
    assert_eq!(pe("pow(0, 0)", &[]), Node::Num(1.0_f64));

    // x - x => 0
    assert_eq!(pe("x - x", &[]), Node::Num(0.0_f64));
    assert_eq!(
        pe("x - x", &[None, Some(3.0_f64), None]),
        Node::Num(0.0_f64)
    );

    // x / x => 1
    assert_eq!(pe("x / x", &[]), Node::Num(1.0_f64));

    // a + a => 2*a
    assert_eq!(
        pe("x + x", &[]),
        Node::Mul(Box::new(Node::Num(2.0_f64)), Box::new(Node::Var(0)))
    );

    // (-a) - b = -(a + b)
    assert_eq!(
        pe("-x - y", &[]),
        Node::Neg(Box::new(Node::Add(
            Box::new(Node::Var(0)),
            Box::new(Node::Var(1)),
        )))
    );

    // a / (-b) = -(a / b)
    assert_eq!(
        pe("x / -y", &[]),
        Node::Neg(Box::new(Node::Div(
            Box::new(Node::Var(0)),
            Box::new(Node::Var(1)),
        )))
    );

    // (-x)^n => x^n for even n  (then x^2 => x*x, x^4 stays as pow)
    assert_eq!(
        pe("(-x)^2", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Var(0)))
    );
    assert_eq!(
        pe("(-x)^4", &[]),
        Node::Pow(Box::new(Node::Var(0)), Box::new(Node::Num(4.0_f64)))
    );
    assert_eq!(
        pe("(-x)^(-2)", &[]),
        Node::Pow(Box::new(Node::Var(0)), Box::new(Node::Num(-2.0_f64)))
    );
    // (-x)^3 stays as Pow(Neg(x), 3) — odd n
    assert_eq!(
        pe("(-x)^3", &[]),
        Node::Pow(
            Box::new(Node::Neg(Box::new(Node::Var(0)))),
            Box::new(Node::Num(3.0_f64))
        )
    );

    // x / c => x * (1/c)
    assert_eq!(
        pe("x / 3", &[]),
        Node::Mul(
            Box::new(Node::Var(0)),
            Box::new(Node::Num(1.0_f64 / 3.0_f64))
        )
    );

    // (x + c1) + c2 => x + (c1 + c2)
    assert_eq!(
        pe("(x + 3) + 2", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0_f64)))
    );
    // (c1 + x) + c2 => x + (c1 + c2)
    assert_eq!(
        pe("(3 + x) + 2", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0_f64)))
    );
    // c1 + (x + c2) => x + (c1 + c2)
    assert_eq!(
        pe("3 + (x + 2)", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0_f64)))
    );
    // c1 + (c2 + x) => x + (c1 + c2)
    assert_eq!(
        pe("3 + (2 + x)", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(5.0_f64)))
    );
    // deep nesting: (((x+1)+2)+3) => x + 6
    assert_eq!(
        pe("(((x + 1) + 2) + 3)", &[]),
        Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(6.0_f64)))
    );

    // (x * c1) * c2 => x * (c1 * c2)
    assert_eq!(
        pe("(x * 3) * 2", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0_f64)))
    );
    // (c1 * x) * c2 => x * (c1 * c2)
    assert_eq!(
        pe("(3 * x) * 2", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0_f64)))
    );
    // c1 * (x * c2) => x * (c1 * c2)
    assert_eq!(
        pe("3 * (x * 2)", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0_f64)))
    );
    // c1 * (c2 * x) => x * (c1 * c2)
    assert_eq!(
        pe("3 * (2 * x)", &[]),
        Node::Mul(Box::new(Node::Var(0)), Box::new(Node::Num(6.0_f64)))
    );

    // exp(ln(x)) = x
    assert_eq!(pe("exp(ln(x))", &[]), Node::Var(0));

    // sin(asin(x)) = x
    assert_eq!(pe("sin(asin(x))", &[]), Node::Var(0));

    // cos(acos(x)) = x
    assert_eq!(pe("cos(acos(x))", &[]), Node::Var(0));

    // tan(atan(x)) = x
    assert_eq!(pe("tan(atan(x))", &[]), Node::Var(0));

    // Runtime verification: all optimizations produce correct values
    let e = |s: &str, v: &[f64]| {
        let n = pe(s, &v.iter().copied().map(Some).collect::<Vec<_>>());
        n.compile()(v, &mut [])
    };
    assert_eq!(e("x - x", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    assert_eq!(e("x / x", &[5.0_f64, 0.0_f64, 0.0_f64]), 1.0_f64);
    assert_eq!(e("x + x", &[3.0_f64, 0.0_f64, 0.0_f64]), 6.0_f64);
    assert_eq!(e("-x - y", &[2.0_f64, 3.0_f64, 0.0_f64]), -5.0_f64);
    assert_eq!(e("x / -y", &[6.0_f64, 2.0_f64, 0.0_f64]), -3.0_f64);
    assert_eq!(e("x^(-1)", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.2_f64);
    assert_eq!(e("(-x)^2", &[5.0_f64, 0.0_f64, 0.0_f64]), 25.0_f64);
    assert_eq!(e("exp(ln(x))", &[2.0_f64, 0.0_f64, 0.0_f64]), 2.0_f64);
    assert_eq!(e("sin(asin(x))", &[0.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    assert_eq!(e("cos(acos(x))", &[1.0_f64, 0.0_f64, 0.0_f64]), 1.0_f64);
    assert_eq!(e("tan(atan(x))", &[0.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    assert_eq!(e("-x + x", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    assert_eq!(e("x + -x", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    // (-a) + a = 0  and  a + (-a) = 0
    assert_eq!(pe("-x + x", &[]), Node::Num(0.0_f64));
    assert_eq!(pe("x + -x", &[]), Node::Num(0.0_f64));
}

#[test]
fn cse_basic() {
    let dim = D3;
    let cse_eval = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        let (f, slots) = n.prepare(&[]);
        let mut cache = vec![0.0_f64; slots];
        f(v, &mut cache)
    };

    // (x+1)*(x+1)
    assert_eq!(
        cse_eval("(x+1)*(x+1)", &[5.0_f64, 0.0_f64, 0.0_f64]),
        36.0_f64
    );
    // x*x + x*x
    assert_eq!(
        cse_eval("x*x + x*x", &[3.0_f64, 0.0_f64, 0.0_f64]),
        18.0_f64
    );
    // sin(x)*cos(y) + sin(x)*cos(y)
    assert_eq!(
        cse_eval(
            "sin(x)*cos(y) + sin(x)*cos(y)",
            &[1.0_f64, 2.0_f64, 0.0_f64]
        ),
        2.0_f64 * (1.0_f64.sin() * 2.0_f64.cos())
    );

    let no_cse = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&[]);
        n.compile()(v, &mut [])
    };
    for &(expr, vars) in &[
        ("x*x + y*y + x*x", &[2.0_f64, 3.0_f64, 0.0_f64]),
        ("x*x*x + x*x*x", &[2.0_f64, 0.0_f64, 0.0_f64]),
        ("(x+y)*(x+y)", &[1.0_f64, 2.0_f64, 0.0_f64]),
        (
            "sqrt(x*x + y*y) + sqrt(x*x + y*y)",
            &[3.0_f64, 4.0_f64, 0.0_f64],
        ),
    ] {
        assert_eq!(
            cse_eval(expr, vars),
            no_cse(expr, vars),
            "CSE result mismatch for '{expr}'"
        );
    }
}

#[test]
fn cse_nested() {
    let dim = D3;
    let cse_eval = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        let (f, slots) = n.prepare(&[]);
        let mut cache = vec![0.0_f64; slots];
        f(v, &mut cache)
    };
    let no_cse = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&[]);
        n.compile()(v, &mut [])
    };

    for &(expr, vars) in &[
        (
            "sin(x*x + y*y) + cos(x*x + y*y)",
            &[1.0_f64, 2.0_f64, 0.0_f64],
        ),
        (
            "sin(x*x + y*y) + cos(x*x + y*y) + x*x",
            &[1.0_f64, 2.0_f64, 0.0_f64],
        ),
    ] {
        assert_eq!(
            cse_eval(expr, vars),
            no_cse(expr, vars),
            "CSE nested result mismatch for '{expr}'"
        );
    }
}

#[test]
fn cse_multiple_extractions() {
    let dim = D3;
    let cse_eval = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        let (f, slots) = n.prepare(&[]);
        let mut cache = vec![0.0_f64; slots];
        f(v, &mut cache)
    };
    let no_cse = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&[]);
        n.compile()(v, &mut [])
    };

    for &(expr, vars) in &[
        ("x*x + x*x + y*y + y*y", &[2.0_f64, 3.0_f64, 0.0_f64]),
        ("(x+1)*(x+1) + (x+1)", &[5.0_f64, 0.0_f64, 0.0_f64]),
        ("x*x*y + x*x*z + x*x", &[2.0_f64, 3.0_f64, 4.0_f64]),
    ] {
        assert_eq!(
            cse_eval(expr, vars),
            no_cse(expr, vars),
            "CSE multi mismatch for '{expr}'"
        );
    }
}

#[test]
fn cse_no_duplicates() {
    let dim = D3;
    let mut n = parse("x + y + z", &dim).expect("expression should be valid");
    n.pre_eval(&[]);
    n.cse();
    assert_eq!(
        n,
        parse("x + y + z", &dim).expect("expression should be valid")
    );
}

#[test]
fn let_chain_order() {
    let pieces: Vec<(usize, Node)> = vec![
        (
            1_usize,
            Node::Mul(Box::new(Node::Var(2)), Box::new(Node::Var(2))),
        ),
        (0_usize, Node::CseRef(1, IndexSet::singleton(2))),
        (
            2_usize,
            Node::F1(
                F1::Ln,
                Box::new(Node::Add(
                    Box::new(Node::CseRef(1, IndexSet::singleton(2))),
                    Box::new(Node::Num(1.0_f64)),
                )),
            ),
        ),
    ];
    let vars = [1.5_f64, 2.5_f64, -3.0_f64];

    let mut chain = Node::Num(0.0_f64);
    for (slot, node) in pieces.into_iter().rev() {
        chain = Node::Let(slot, Box::new(node), Box::new(chain));
    }
    let mut cse = vec![0.0_f64; 10];
    let _: f64 = chain.compile()(&vars, &mut cse);
    assert!((cse[1] - 9.0_f64).abs() < 1e-12_f64, "cse[1]");
    assert!(
        (cse[0] - 9.0_f64).abs() < 1e-12_f64,
        "cse[0] should be 9.0_f64, got {}",
        cse[0]
    );
    assert!(
        (cse[2] - (10.0_f64).ln()).abs() < 1e-12_f64,
        "cse[2] should be ln(10), got {}",
        cse[2]
    );
}

#[test]
fn cse_vs_nocse() {
    let dim = D3;
    let exprs = [
        ("simple", "x + y * z"),
        ("medium", "sin(x) + cos(y) * z^2 + sqrt(x*x + y*y)"),
        (
            "heavy",
            "exp(sin(x) * cos(y)) + ln(z*z + 1) + atan2(sqrt(x*x + y*y + z*z), 1) + sqrt(abs(x + y))",
        ),
        ("repeated", "(x*x + y*y)*(x*x + y*y) + sin(x*x + y*y)"),
        (
            "very_heavy",
            concat!(
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
        ),
    ];
    let side = 32_u32;
    let scale = 10.0_f64 / f64::from(side);

    for (name, expr_str) in &exprs {
        let mut node1 = parse(expr_str, &dim).expect("expression should be valid");
        node1.pre_eval(&[]);
        let multi1 = node1.compile_multi(&[0, 1, 2]);

        let mut node2 = parse(expr_str, &dim).expect("expression should be valid");
        let multi2 = node2.prepare_multi(&[], &[0, 1, 2]);

        let mut sum1 = 0.0_f64;
        let mut sum2 = 0.0_f64;

        for nz in 0..side {
            let zv = f64::from(nz).mul_add(scale, -5.0_f64);
            for ny in 0..side {
                let yv = f64::from(ny).mul_add(scale, -5.0_f64);
                for nx in 0..side {
                    let xv = f64::from(nx).mul_add(scale, -5.0_f64);
                    let vars = [xv, yv, zv];

                    let mut cache1 = vec![0.0_f64; multi1.cse_slots];
                    for g in &multi1.groups {
                        (g.combined)(&vars, &mut cache1);
                    }
                    sum1 += (multi1.main)(&vars, &mut cache1);

                    let mut cache2 = vec![0.0_f64; multi2.cse_slots];
                    for g in &multi2.groups {
                        (g.combined)(&vars, &mut cache2);
                    }
                    sum2 += (multi2.main)(&vars, &mut cache2);
                }
            }
        }

        let diff = (sum1 - sum2).abs();
        let tol = 1e-12_f64;
        assert!(
            diff < tol,
            "CSE divergence in '{name}': diff={diff:.6e} > tol={tol:.6e}"
        );
    }
}

#[test]
fn compile_patterns_fused() {
    let dim = D3;

    let cases = [
        // MulAdd: a*b + c
        ("x*y + z", [2.0_f64, 3.0_f64, 4.0_f64], 10.0_f64),
        ("z + x*y", [2.0_f64, 3.0_f64, 4.0_f64], 10.0_f64),
        ("x*y + z", [-1.5_f64, 2.0_f64, 3.0_f64], 0.0_f64),
        ("x*y + z", [0.0_f64, 5.0_f64, 1.0_f64], 1.0_f64),
        // MulSub: a*b - c
        ("x*y - z", [2.0_f64, 3.0_f64, 4.0_f64], 2.0_f64),
        ("x*y - z", [1.0_f64, 1.0_f64, 0.5_f64], 0.5_f64),
        // NegMulAdd: c - a*b
        ("z - x*y", [2.0_f64, 3.0_f64, 4.0_f64], -2.0_f64),
        ("z - x*y", [1.0_f64, 1.0_f64, 5.0_f64], 4.0_f64),
        // NegMul: -(a*b)
        ("-(x*y)", [2.0_f64, 3.0_f64, 0.0_f64], -6.0_f64),
        ("-(x*y)", [-2.0_f64, 3.0_f64, 0.0_f64], 6.0_f64),
    ];

    for &(expr, vars, expected) in &cases {
        let n = parse(expr, &dim).expect("expression should be valid");
        let fused = n.compile()(&vars, &mut []);

        let mut n_ref = parse(expr, &dim).expect("expression should be valid");
        n_ref.pre_eval(&[]);
        let ref_val = n_ref.compile()(&vars, &mut []);

        let diff = (fused - ref_val).abs();
        assert!(
            diff < 1e-12_f64,
            "fused mismatch for '{expr}': fused={fused}, ref={ref_val}, diff={diff:.6e}"
        );
        let exp_diff = (fused - expected).abs();
        assert!(
            exp_diff < 1e-12_f64,
            "wrong result for '{expr}': fused={fused}, expected={expected}, diff={exp_diff:.6e}"
        );
    }

    // Fused path (direct compile)
    // Ref path (pre_eval + compile)
    let fuzz_cases: &[(&str, &[f64])] = &[
        ("x*y + z", &[1.5, 2.5, 3.5]),
        ("z + x*y", &[1.5, 2.5, 3.5]),
        ("x*y - z", &[1.5, 2.5, 3.5]),
        ("z - x*y", &[1.5, 2.5, 3.5]),
        ("-(x*y)", &[1.5, 2.5, 0.0_f64]),
    ];
    // Fuzz: fused vs full pipeline (pre_eval + CSE + compile)
    for &(expr, vars) in fuzz_cases {
        let n = parse(expr, &dim).expect("expression should be valid");
        let fused = n.compile()(vars, &mut []);

        let mut n_ref = parse(expr, &dim).expect("expression should be valid");
        let (f_prep, slots) = n_ref.prepare(&[]);
        let mut c = vec![0.0_f64; slots];
        let prep = f_prep(vars, &mut c);

        let diff = (fused - prep).abs();
        assert!(
            diff < 1e-12_f64,
            "fused vs pipeline mismatch for '{expr}': fused={fused}, prep={prep}, diff={diff:.6e}"
        );
    }
}

#[test]
fn parse_errors() {
    let dim = D3;
    assert!(parse("", &dim).is_err());
    assert!(parse("x + ", &dim).is_err());
    assert!(parse("(x + y", &dim).is_err());
    assert!(parse("unknown", &dim).is_err());
    assert!(parse("sin()", &dim).is_err());
    assert!(parse("atan2(x)", &dim).is_err());
}

#[test]
fn percent_modulo() {
    let dim = D3;
    let e = |s: &str, v: &[f64]| {
        parse(s, &dim)
            .expect("expression should be valid")
            .compile()(v, &mut [])
    };
    assert_eq!(e("5 % 3", &[]), 2.0_f64);
    assert_eq!(e("7 % 3.5", &[]), 0.0_f64);
    assert_eq!(e("10 % 4", &[]), 2.0_f64);
    assert_eq!(e("x % y", &[10.0_f64, 3.0_f64, 0.0_f64]), 1.0_f64);
    assert_eq!(e("x % 2", &[5.0_f64, 0.0_f64, 0.0_f64]), 1.0_f64);
}

#[test]
fn pipe_abs() {
    let dim = D3;
    let e = |s: &str, v: &[f64]| {
        parse(s, &dim)
            .expect("expression should be valid")
            .compile()(v, &mut [])
    };
    assert_eq!(e("|x|", &[-5.0_f64, 0.0_f64, 0.0_f64]), 5.0_f64);
    assert_eq!(e("|x|", &[3.0_f64, 0.0_f64, 0.0_f64]), 3.0_f64);
    assert_eq!(e("|x + y|", &[2.0_f64, -3.0_f64, 0.0_f64]), 1.0_f64);
    assert_eq!(e("|x| * |y|", &[-2.0_f64, -3.0_f64, 0.0_f64]), 6.0_f64);
    assert_eq!(e("-|x|", &[-5.0_f64, 0.0_f64, 0.0_f64]), -5.0_f64);
    assert_eq!(e("|x|^2", &[-3.0_f64, 0.0_f64, 0.0_f64]), 9.0_f64);
    assert_eq!(e("||x||", &[-5.0_f64, 0.0_f64, 0.0_f64]), 5.0_f64);
}

#[test]
fn pipe_abs_pre_eval() {
    let dim = D3;
    let pe = |s: &str| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&[]);
        n
    };
    // |5| = 5
    assert_eq!(pe("|5|"), Node::Num(5.0_f64));
    // |-3| = 3
    assert_eq!(pe("|-3|"), Node::Num(3.0_f64));
    // abs(abs(x)) = abs(x) (idempotent)
    assert_eq!(pe("||x||"), Node::F1(F1::Abs, Box::new(Node::Var(0))));
}

#[test]
fn percent_pre_eval() {
    let dim = D3;
    let pe = |s: &str| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&[]);
        n
    };
    assert_eq!(pe("10 % 3"), Node::Num(1.0_f64));
    // x % x => 0
    let n = pe("x % x");
    assert_eq!(n, Node::Num(0.0_f64));
    let n = pe("(x+y) % (x+y)");
    assert_eq!(n, Node::Num(0.0_f64));
    // 0 % x => 0
    let n = pe("0 % x");
    assert_eq!(n, Node::Num(0.0_f64));
    // a % (-b) => a % b
    let n = pe("x % -y");
    assert_eq!(n, Node::Mod(Box::new(Node::Var(0)), Box::new(Node::Var(1))));
    // (-a) % b => -(a % b)
    let n = pe("-x % y");
    assert_eq!(
        n,
        Node::Neg(Box::new(Node::Mod(
            Box::new(Node::Var(0)),
            Box::new(Node::Var(1)),
        )))
    );
    // (-a) % (-b) => -(a % b)
    let n = pe("-x % -y");
    assert_eq!(
        n,
        Node::Neg(Box::new(Node::Mod(
            Box::new(Node::Var(0)),
            Box::new(Node::Var(1)),
        )))
    );
}

#[test]
fn percent_runtime_optimizations() {
    let dim = D3;
    let e = |s: &str, v: &[f64]| {
        let mut n = parse(s, &dim).expect("expression should be valid");
        n.pre_eval(&v.iter().copied().map(Some).collect::<Vec<_>>());
        n.compile()(v, &mut [])
    };
    // x % x = 0
    assert_eq!(e("x % x", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    // 0 % x = 0
    assert_eq!(e("0 % x", &[5.0_f64, 0.0_f64, 0.0_f64]), 0.0_f64);
    // (-a) % b = -(a % b)
    assert_eq!(
        e("-x % y", &[10.0_f64, 3.0_f64, 0.0_f64]),
        -(10.0_f64 % 3.0_f64)
    );
    // a % (-b) = a % b
    assert_eq!(
        e("x % -y", &[10.0_f64, 3.0_f64, 0.0_f64]),
        10.0_f64 % 3.0_f64
    );
    // (-a) % (-b) = -(a % b)
    // compile_multi with 10 spatial dims
    assert_eq!(
        e("-x % -y", &[10.0_f64, 3.0_f64, 0.0_f64]),
        -(10.0_f64 % 3.0_f64)
    );
}

#[test]
fn nd_variables() {
    let dim = TestVars { ndim: 4 };
    let e = |s: &str, v: &[f64]| {
        parse(s, &dim)
            .expect("expression should be valid")
            .compile()(v, &mut [])
    };
    assert_eq!(e("x0", &[5.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]), 5.0_f64);
    assert_eq!(e("x3", &[0.0_f64, 0.0_f64, 0.0_f64, 5.0_f64]), 5.0_f64);
    assert!(parse("x4", &dim).is_err());
    assert!(parse("xabc", &dim).is_err());
}

#[test]
fn error_structure() {
    let dim = D3;
    let err = parse("(x + y", &dim).expect_err("expression should be invalid");
    assert!(matches!(
        err,
        Error::Parser {
            col: _,
            kind: ParseErrorKind::ExpectedRParen(_)
        }
    ));
}

#[test]
fn compile_multi_10_dim() {
    let dim = TestVars { ndim: 10 };
    let expr = "x0*x0 + x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5 + x6*x6 + x7*x7 + x8*x8 + x9*x9";
    let vars = [
        1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64, 10.0_f64,
    ];
    let expected: f64 = vars.iter().map(|v| v * v).sum();
    let tol = 1e-12_f64;

    let mut node = parse(expr, &dim).expect("expression should be valid");
    node.pre_eval(&[]);
    let multi = node.compile_multi(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let mut cache = vec![0.0_f64; multi.cse_slots];
    for g in &multi.groups {
        (g.combined)(&vars, &mut cache);
    }
    let result = (multi.main)(&vars, &mut cache);

    // Reference: direct compile (no multi)
    let node_ref = parse(expr, &dim).expect("expression should be valid");
    let reference = node_ref.compile()(&vars, &mut []);

    assert!(
        (result - reference).abs() < tol,
        "compile_multi(10-dim)={result} != reference={reference}"
    );
    assert!(
        (result - expected).abs() < tol,
        "compile_multi(10-dim)={result} != expected={expected}"
    );
}
