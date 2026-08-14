use hypervox_expr::*;
use std::str::FromStr as _;

// Empty enums

define_ext_f0!(EmptyF0);
define_ext_f1!(EmptyF1);
define_ext_f2!(EmptyF2);

#[test]
fn empty_f0() {
    assert!(EmptyF0::from_str("anything").is_err());
}

#[test]
fn empty_f1() {
    assert!(EmptyF1::from_str("anything").is_err());
}

#[test]
fn empty_f2() {
    assert!(EmptyF2::from_str("anything").is_err());
}

// TestF0

define_ext_f0!(
    TestF0,
    Zero => "zero" = 0_f64,
    One => "one" = 1_f64,
);

#[test]
fn f0_from_str() {
    assert_eq!(
        TestF0::from_str("zero").expect("should be valid"),
        TestF0::Zero
    );
    assert_eq!(
        TestF0::from_str("one").expect("should be valid"),
        TestF0::One
    );
    assert!(TestF0::from_str("unknown").is_err());
}

#[test]
fn f0_to_num() {
    assert_eq!(TestF0::Zero.to_num(), 0.0_f64);
    assert_eq!(TestF0::One.to_num(), 1.0_f64);
}

#[test]
fn f0_names() {
    assert_eq!(TestF0::Zero.names(), &["zero", "one"]);
    assert_eq!(TestF0::One.names(), &["zero", "one"]);
}

#[test]
fn f0_list() {
    let list = TestF0::Zero.list();
    assert_eq!(list, "zero, one");
    assert!(list.contains("zero"));
    assert!(list.contains("one"));
}

// TestF1

define_ext_f1!(
    TestF1,
    Double => "double" = |x| x * 2_f64,
    Square => "square" = |x| x * x,
);

#[test]
fn f1_from_str() {
    assert_eq!(
        TestF1::from_str("double").expect("should be valid"),
        TestF1::Double
    );
    assert_eq!(
        TestF1::from_str("square").expect("should be valid"),
        TestF1::Square
    );
    assert!(TestF1::from_str("unknown").is_err());
}

#[test]
fn f1_to_fn() {
    let f = TestF1::Double.to_fn();
    assert_eq!(f(0.0_f64), 0.0_f64);
    assert_eq!(f(1.0_f64), 2.0_f64);
    assert_eq!(f(-3.0_f64), -6.0_f64);

    let g = TestF1::Square.to_fn();
    assert_eq!(g(0.0_f64), 0.0_f64);
    assert_eq!(g(3.0_f64), 9.0_f64);
    assert_eq!(g(-4.0_f64), 16.0_f64);
}

#[test]
fn f1_names() {
    assert_eq!(TestF1::Double.names(), &["double", "square"]);
}

#[test]
fn f1_list() {
    assert_eq!(TestF1::Double.list(), "double, square");
}

// TestF2

define_ext_f2!(
    TestF2,
    Sum => "sum" = |x, y| x + y,
    Product => "product" = |x, y| x * y,
);

#[test]
fn f2_from_str() {
    assert_eq!(
        TestF2::from_str("sum").expect("should be valid"),
        TestF2::Sum
    );
    assert_eq!(
        TestF2::from_str("product").expect("should be valid"),
        TestF2::Product
    );
    assert!(TestF2::from_str("unknown").is_err());
}

#[test]
fn f2_to_fn() {
    let f = TestF2::Sum.to_fn();
    assert_eq!(f(0.0_f64, 0.0_f64), 0.0_f64);
    assert_eq!(f(1.0_f64, 2.0_f64), 3.0_f64);
    assert_eq!(f(-1.0_f64, 1.0_f64), 0.0_f64);

    let g = TestF2::Product.to_fn();
    assert_eq!(g(0.0_f64, 5.0_f64), 0.0_f64);
    assert_eq!(g(2.0_f64, 3.0_f64), 6.0_f64);
    assert_eq!(g(-2.0_f64, 4.0_f64), -8.0_f64);
}

#[test]
fn f2_names() {
    assert_eq!(TestF2::Sum.names(), &["sum", "product"]);
}

#[test]
fn f2_list() {
    assert_eq!(TestF2::Sum.list(), "sum, product");
}

// Integration: parse_with_ext + compile

#[derive(Clone, Copy)]
struct TestVars;

impl VarMap for TestVars {
    fn ndim(&self) -> usize {
        3
    }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name {
            "x" => Some(0),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        }
    }
    fn primary_prefix(&self) -> &'static str {
        "x"
    }
}

define_ext_f0!(MyF0, Answer => "answer" = 42.0_f64);

define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);

define_ext_f2!(MyF2, Hypot => "hypot" = |x, y| x.hypot(y));

#[test]
fn ext_f0_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("answer", &TestVars)
        .expect("expression should be valid");
    assert_eq!(node.compile()(&[], &mut []), 42.0_f64);
}

#[test]
fn ext_f1_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("cube(x)", &TestVars)
        .expect("expression should be valid");
    assert_eq!(
        node.compile()(&[3.0_f64, 0.0_f64, 0.0_f64], &mut []),
        27.0_f64
    );
}

#[test]
fn ext_f2_in_expression() {
    let node = parse_with_ext::<TestVars, MyF0, MyF1, MyF2>("hypot(3, 4)", &TestVars)
        .expect("expression should be valid");
    assert_eq!(node.compile()(&[], &mut []), 5.0_f64);
}

#[test]
fn ext_combined_expression() {
    let node = parse_with_ext::<_, MyF0, MyF1, MyF2>("answer + cube(x) + hypot(y, z)", &TestVars)
        .expect("expression should be valid");
    let result = node.compile()(&[2.0_f64, 3.0_f64, 4.0_f64], &mut []);
    // 42 + 2^3 + sqrt(3^2 + 4^2) = 42 + 8 + 5 = 55
    assert_eq!(result, 55.0_f64);
}

#[test]
fn ext_nested_expression() {
    let node = parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(answer + x) - hypot(y, z)", &TestVars)
        .expect("expression should be valid");
    let result = node.compile()(&[1.0_f64, 3.0_f64, 4.0_f64], &mut []);
    // cube(42 + 1) - sqrt(3^2 + 4^2) = 43^3 - 5 = 79507 - 5 = 79502
    assert_eq!(result, 79507.0_f64 - 5.0_f64);
}

#[test]
fn ext_f1_in_pre_eval() {
    let mut node = parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(2)", &TestVars)
        .expect("expression should be valid");
    node.pre_eval(&[]);
    // 2^3 = 8, should fold to constant
    assert_eq!(node, Node::Num(8.0_f64));
}

#[test]
fn ext_validate() {
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("answer + x", &TestVars).is_ok());
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("unknown", &TestVars).is_err());
    assert!(validate_with_ext::<_, MyF0, MyF1, MyF2>("", &TestVars).is_err());
}

#[test]
fn ext_parse_error() {
    let err = parse_with_ext::<_, MyF0, MyF1, MyF2>("cube(1", &TestVars)
        .expect_err("expression should be invalid");
    assert!(matches!(
        err,
        Error::Parser {
            kind: ParseErrorKind::ExpectedRParenOrComma(_),
            ..
        }
    ));
}
