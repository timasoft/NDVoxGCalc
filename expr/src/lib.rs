//! Math expression parser, compiler, and CSE engine.
//!
//! Parses string expressions (e.g. `"sin(x)*cos(y)"`) into an AST ([`Node`]),
//! applies constant folding ([`Node::pre_eval`]),
//! common-subexpression elimination ([`Node::cse`]),
//! and compiles to closures ([`Node::compile`] / [`Node::compile_multi`])
//! for fast repeated evaluation over N-dimensional grids.
//! Variable names are resolved through the [`VarMap`] trait.

use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

mod errors;
mod index_set;
mod node;
mod parse;

pub use errors::{ArithIndexSetTryFromError, Error, LexerErrorKind, ParseErrorKind};
pub use index_set::{ArithIndexSet, ArithRangeFrom, ArithRangeIter, IndexSet, IndexSetIter};
pub use node::{CompiledExpr, CompiledExprMulti, InvariantGroup, Node};
pub use parse::{parse, parse_with_ext, validate, validate_with_ext};

/// Trait for resolving variable names in expressions.
/// Implementors define variable aliases and the indexed-variable naming scheme.
pub trait VarMap {
    /// Number of dimensions.
    fn ndim(&self) -> usize;

    /// Resolve a variable alias (e.g. "x", "y", "z") to a dimension index.
    fn resolve_alias(&self, name: &str) -> Option<usize>;

    /// Primary variable prefix used for indexed variables (e.g. "x0", "x1", ..).
    fn primary_prefix(&self) -> &str;
}

macro_rules! define_f0 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Constants (PI, E).
        #[derive(Debug, Clone, Copy)]
        pub enum F0 {
            $($variant,)*
        }
        impl F0 {
            /// Evaluate the constant as an f64.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F0;
            /// let pi = F0::PI.to_num();
            /// assert!((pi - std::f64::consts::PI).abs() < 1e-15);
            /// ```
            pub const fn to_num(self) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All constant names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F0 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown const '{s}'")),
                }
            }
        }
    };
}

macro_rules! define_f1 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Single-argument math functions.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum F1 {
            $($variant,)*
        }
        impl F1 {
            /// Resolve to a function pointer.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F1;
            /// let f = F1::Sin.to_fn();
            /// assert!((f(1.0) - 1.0_f64.sin()).abs() < 1e-15);
            /// ```
            #[inline]
            pub const fn to_fn(self) -> fn(f64) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F1 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

macro_rules! define_f2 {
    ($($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Two-argument math functions (atan2, pow).
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum F2 {
            $($variant,)*
        }
        impl F2 {
            /// Resolve to a function pointer.
            ///
            /// # Examples
            /// ```
            /// # use hypervox_expr::F2;
            /// let atan2 = F2::Atan2.to_fn();
            /// assert_eq!(atan2(0.0, 1.0), 0.0);
            /// ```
            #[inline]
            pub const fn to_fn(self) -> fn(f64, f64) -> f64 {
                match self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            pub const NAMES: &'static [&'static str] = &[$($str,)*];
        }
        impl FromStr for F2 {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

define_f0! {
    PI => "PI" = std::f64::consts::PI,
    E => "E" = std::f64::consts::E,
}

define_f1! {
    Sin => "sin" = f64::sin,
    Cos => "cos" = f64::cos,
    Tan => "tan" = f64::tan,
    Asin => "asin" = f64::asin,
    Acos => "acos" = f64::acos,
    Atan => "atan" = f64::atan,
    Sinh => "sinh" = f64::sinh,
    Cosh => "cosh" = f64::cosh,
    Tanh => "tanh" = f64::tanh,
    Sqrt => "sqrt" = f64::sqrt,
    Cbrt => "cbrt" = f64::cbrt,
    Exp => "exp" = f64::exp,
    Ln => "ln" = f64::ln,
    Log10 => "log10" = f64::log10,
    Log2 => "log2" = f64::log2,
    Floor => "floor" = f64::floor,
    Ceil => "ceil" = f64::ceil,
    Round => "round" = f64::round,
    Trunc => "trunc" = f64::trunc,
    Abs => "abs" = f64::abs,
}

define_f2! {
    Atan2 => "atan2" = f64::atan2,
    Pow => "pow" = |a, b| {
        if a == 0.0_f64 && b == 0.0_f64 {
            1.0_f64
        } else {
            let exp_int = b as i32;
            if f64::from(exp_int) == b {
                a.powi(exp_int)
            } else {
                a.powf(b)
            }
        }
    },
}

/// Return a comma-separated list of available constant names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f0_list;
/// let list = f0_list();
/// assert!(list.contains("PI"));
/// ```
#[must_use]
pub fn f0_list() -> String {
    F0::NAMES.join(", ")
}

/// Return a comma-separated list of available single-argument function names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f1_list;
/// let list = f1_list();
/// assert!(list.contains("sin"));
/// ```
#[must_use]
pub fn f1_list() -> String {
    F1::NAMES.join(", ")
}

/// Return a comma-separated list of available two-argument function names.
///
/// # Examples
/// ```
/// # use hypervox_expr::f2_list;
/// let list = f2_list();
/// assert!(list.contains("atan2"));
/// ```
#[must_use]
pub fn f2_list() -> String {
    F2::NAMES.join(", ")
}

/// Trait for custom constants.
pub trait ExtF0: Debug + Clone + PartialEq {
    /// Evaluate the constant as an f64.
    fn to_num(&self) -> f64;

    /// All constant names for display.
    fn names(&self) -> &[&str];

    /// List all constant names.
    ///
    /// The default implementation returns the elements of [`Self::names`] separated by commas.
    fn list(&self) -> String {
        self.names().join(", ")
    }
}

/// Trait for custom single-argument functions.
pub trait ExtF1: Debug + Clone + PartialEq {
    /// Resolve to a function pointer.
    fn to_fn(&self) -> fn(f64) -> f64;

    /// All function names for display.
    fn names(&self) -> &[&str];

    /// List all function names.
    ///
    /// The default implementation returns the elements of [`Self::names`] separated by commas.
    fn list(&self) -> String {
        self.names().join(", ")
    }
}

/// Trait for custom two-argument functions.
pub trait ExtF2: Debug + Clone + PartialEq {
    /// Resolve to a function pointer.
    fn to_fn(&self) -> fn(f64, f64) -> f64;

    /// All function names for display.
    fn names(&self) -> &[&str];

    /// List all function names.
    ///
    /// The default implementation returns the elements of [`Self::names`] separated by commas.
    fn list(&self) -> String {
        self.names().join(", ")
    }
}

/// Define a custom-constants enum implementing [`ExtF0`] and [`FromStr`].
///
/// # Examples
/// ```
/// use hypervox_expr::ExtF0;
/// hypervox_expr::define_ext_f0!(MyF0, MyConst => "my_const" = 42.0);
/// assert_eq!(MyF0::MyConst.to_num(), 42.0);
/// ```
#[macro_export]
macro_rules! define_ext_f0 {
    ($enum_name:ident) => {
        define_ext_f0!($enum_name,);
    };
    ($enum_name:ident, $($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Constants.
        #[expect(clippy::derive_partial_eq_without_eq)]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum $enum_name {
            $($variant,)*
        }
        impl hypervox_expr::ExtF0 for $enum_name {
            /// Evaluate the constant as an f64.
            fn to_num(&self) -> f64 {
                match *self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All constant names for display.
            fn names(&self) -> &[&str] {
                &[$($str,)*]
            }
        }
        impl std::str::FromStr for $enum_name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown const '{s}'")),
                }
            }
        }
    };
}

/// Define custom single-argument functions implementing [`ExtF1`] and [`FromStr`].
///
/// # Examples
/// ```
/// use hypervox_expr::ExtF1;
/// hypervox_expr::define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);
/// assert_eq!(MyF1::Cube.to_fn()(3.0), 27.0);
/// ```
#[macro_export]
macro_rules! define_ext_f1 {
    ($enum_name:ident) => {
        define_ext_f1!($enum_name,);
    };
    ($enum_name:ident, $($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Single-argument math functions.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $($variant,)*
        }
        impl hypervox_expr::ExtF1 for $enum_name {
            /// Resolve to a function pointer.
            #[inline]
            fn to_fn(&self) -> fn(f64) -> f64 {
                match *self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            fn names(&self) -> &[&str] {
                &[$($str,)*]
            }
        }
        impl std::str::FromStr for $enum_name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

/// Define custom two-argument functions implementing [`ExtF2`] and [`FromStr`].
///
/// # Examples
/// ```
/// use hypervox_expr::ExtF2;
/// hypervox_expr::define_ext_f2!(MyF2, Hypot => "hypot" = |x, y| x.hypot(y));
/// assert_eq!(MyF2::Hypot.to_fn()(3.0, 4.0), 5.0);
/// ```
#[macro_export]
macro_rules! define_ext_f2 {
    ($enum_name:ident) => {
        define_ext_f2!($enum_name,);
    };
    ($enum_name:ident, $($variant:ident => $str:literal = $body:expr),* $(,)?) => {
        /// Two-argument math functions (atan2, pow).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $($variant,)*
        }
        impl hypervox_expr::ExtF2 for $enum_name {
            /// Resolve to a function pointer.
            #[inline]
            fn to_fn(&self) -> fn(f64, f64) -> f64 {
                match *self {
                    $(Self::$variant => $body,)*
                }
            }
            /// All function names for display.
            fn names(&self) -> &[&str] {
                &[$($str,)*]
            }
        }
        impl std::str::FromStr for $enum_name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($str => Ok(Self::$variant),)*
                    _ => Err(format!("unknown function '{s}'")),
                }
            }
        }
    };
}

/// Zero-variant placeholder type for when no external functions/constants are used.
///
/// Acts as the default type parameter for [`Node`], [`parse`], and [`validate`]
/// so that the regular API requires no generics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoExtF {}

impl ExtF0 for NoExtF {
    fn to_num(&self) -> f64 {
        unreachable!("NoExtF should never be called")
    }
    fn names(&self) -> &[&str] {
        unreachable!("NoExtF should never be called")
    }
}

impl ExtF1 for NoExtF {
    fn to_fn(&self) -> fn(f64) -> f64 {
        unreachable!("NoExtF should never be called")
    }
    fn names(&self) -> &[&str] {
        unreachable!("NoExtF should never be called")
    }
}

impl ExtF2 for NoExtF {
    fn to_fn(&self) -> fn(f64, f64) -> f64 {
        unreachable!("NoExtF should never be called")
    }
    fn names(&self) -> &[&str] {
        unreachable!("NoExtF should never be called")
    }
}

impl FromStr for NoExtF {
    type Err = String;
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err(String::from("no external constants or functions defined"))
    }
}

/// Token produced by the lexer and consumed by the Pratt parser.
#[derive(Debug, Clone, PartialEq)]
pub enum LexerToken {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,
    Pipe,
    LParen,
    RParen,
    Comma,
    Eof,
}

impl Display for LexerToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Num(v) => write!(f, "{v}"),
            Self::Ident(s) => write!(f, "'{s}'"),
            Self::Plus => write!(f, "'+'"),
            Self::Minus => write!(f, "'-'"),
            Self::Star => write!(f, "'*'"),
            Self::Slash => write!(f, "'/'"),
            Self::Caret => write!(f, "'^'"),
            Self::Percent => write!(f, "'%'"),
            Self::Pipe => write!(f, "'|'"),
            Self::LParen => write!(f, "'('"),
            Self::RParen => write!(f, "')'"),
            Self::Comma => write!(f, "','"),
            Self::Eof => write!(f, "end of expression"),
        }
    }
}

/// Fused multiply-add that respects the `fma` feature flag.
///
/// Computes `self * b + c`. When the `fma` feature is enabled this maps to the
/// hardware fused multiply-add (`mul_add`), preserving a single rounding step;
/// otherwise it falls back to the separate multiply-then-add operations.
///
/// Implemented for `f32` and `f64`
pub trait CondMulAdd {
    #[expect(clippy::min_ident_chars)]
    #[must_use]
    fn cond_mul_add(self, b: Self, c: Self) -> Self;
}

#[cfg(feature = "fma")]
impl CondMulAdd for f64 {
    #[inline]
    #[expect(clippy::disallowed_methods)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(not(feature = "fma"))]
impl CondMulAdd for f64 {
    #[inline]
    #[expect(clippy::suboptimal_flops)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}

#[cfg(feature = "fma")]
impl CondMulAdd for f32 {
    #[inline]
    #[expect(clippy::disallowed_methods)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(not(feature = "fma"))]
impl CondMulAdd for f32 {
    #[inline]
    #[expect(clippy::suboptimal_flops)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
