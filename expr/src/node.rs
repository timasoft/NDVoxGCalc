use crate::index_set::{ArithIndexSet, IndexSet};
use crate::{CondMulAdd, ExtF1, ExtF2, F1, F2, NoExtF};

/// A compiled expression closure: `(vars, cse_cache) -> result`.
pub type CompiledExpr = Box<dyn Fn(&[f64], &mut [f64]) -> f64 + Send + Sync>;

/// AST node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Node<EF1: ExtF1 = NoExtF, EF2: ExtF2 = NoExtF> {
    Num(f64),
    Var(usize),
    Neg(Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Pow(Box<Self>, Box<Self>),
    Mod(Box<Self>, Box<Self>),
    F1(F1, Box<Self>),
    F2(F2, Box<Self>, Box<Self>),
    /// External single-argument function call.
    ExtF1(EF1, Box<Self>),
    /// External two-argument function call.
    ExtF2(EF2, Box<Self>, Box<Self>),
    #[expect(clippy::doc_markdown)]
    /// let slot_i = expr in body
    Let(usize, Box<Self>, Box<Self>),
    /// reference to cached CSE slot
    CseRef(usize, IndexSet),
}

/// Leaf-inlined child classification
enum InlinedChild {
    Num(f64),
    Var(usize),
    Cse(usize),
    Compound(CompiledExpr),
}

/// A group of dimension-invariant sub-expressions at a given nesting level.
pub struct InvariantGroup {
    /// Nesting level: higher = more invariant (evaluated sooner).
    pub level: usize,
    /// Combined closure populating CSE slots for all invariants at this level.
    pub combined: CompiledExpr,
}

/// Multi-level compiled expression with invariant groups and main expression.
pub struct CompiledExprMulti {
    /// Invariant groups.
    pub groups: Vec<InvariantGroup>,
    /// Main expression after extracting all invariants.
    pub main: CompiledExpr,
    /// Total number of CSE slots required.
    pub cse_slots: usize,
}

impl<EF1: ExtF1, EF2: ExtF2> Node<EF1, EF2> {
    /// Evaluate constant sub-expressions and apply algebraic simplifications at compile time.
    ///
    /// When `vars[i]` is `Some(v)`, variable `i` is replaced with `v` before folding.
    ///
    /// Simplifications include:
    /// identity elimination (`x+0`, `x*1`, `x^1`),
    /// zero propagation (`x*0`),
    /// negation rewrites (`-(-x)` -> `x`, `x/-1` -> `-x`),
    /// inverse composition (`ln(exp(x))` -> `x`),
    /// constant reassociation,
    /// and `x-x` -> `0`, `x/x` -> `1`, `x+x` -> `2*x`.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let mut node: Node = Node::Add(Box::new(Node::Num(2.0_f64)), Box::new(Node::Num(3.0_f64)));
    /// node.pre_eval(&[]);
    /// assert_eq!(node, Node::Num(5.0_f64));
    /// ```
    pub fn pre_eval(&mut self, vars: &[Option<f64>]) {
        match self {
            Self::Num(_) | Self::CseRef(_, _) => {}
            Self::Var(i) => {
                if let Some(Some(v)) = vars.get(*i) {
                    *self = Self::Num(*v);
                }
            }
            Self::Neg(a) => {
                a.pre_eval(vars);
                if let Self::Num(x) = a.as_ref() {
                    *self = Self::Num(-*x);
                } else if let Self::Neg(inner) = a.as_ref() {
                    // -(-x) => x
                    *self = *inner.clone();
                }
            }
            Self::Add(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Self::Num(x), Self::Num(y)) => *self = Self::Num(x + y),
                    (Self::Num(x), _) if *x == 0.0_f64 => *self = *b.clone(),
                    (_, Self::Num(y)) if *y == 0.0_f64 => *self = *a.clone(),
                    (Self::Neg(a_inner), Self::Neg(b_inner)) => {
                        // (-a) + (-b) = -(a + b)
                        let mut new =
                            Self::Neg(Box::new(Self::Add(a_inner.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Self::Neg(a_inner), _) if *a_inner == *b => {
                        // (-a) + a = 0
                        *self = Self::Num(0.0_f64);
                    }
                    (_, Self::Neg(b_inner)) if *a == *b_inner => {
                        // a + (-a) = 0
                        *self = Self::Num(0.0_f64);
                    }
                    (Self::Neg(a_inner), _) => {
                        // (-a) + b = b - a
                        let mut new = Self::Sub(b.clone(), a_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Self::Neg(b_inner)) => {
                        // a + (-b) = a - b
                        let mut new = Self::Sub(a.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ if *a == *b => {
                        // a + a => 2*a
                        let mut new = Self::Mul(Box::new(Self::Num(2.0_f64)), a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    // reassociate: (x + c1) + c2 / (c1 + x) + c2 => x + (c1 + c2)
                    (Self::Add(left, right), Self::Num(c2)) => {
                        if let Self::Num(c1) = right.as_ref() {
                            let mut new = Self::Add(left.clone(), Box::new(Self::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Self::Num(c1) = left.as_ref() {
                            let mut new = Self::Add(right.clone(), Box::new(Self::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    // reassociate: c1 + (x + c2) / c1 + (c2 + x) => x + (c1 + c2)
                    (Self::Num(c1), Self::Add(left, right)) => {
                        if let Self::Num(c2) = right.as_ref() {
                            let mut new = Self::Add(left.clone(), Box::new(Self::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Self::Num(c2) = left.as_ref() {
                            let mut new = Self::Add(right.clone(), Box::new(Self::Num(*c1 + *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    _ => {}
                }
            }
            Self::Sub(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                if let (Self::Num(x), Self::Num(y)) = (a.as_ref(), b.as_ref()) {
                    *self = Self::Num(x - y);
                } else if let Self::Neg(b_inner) = b.as_ref() {
                    // a - (-b) => a + b
                    let mut new = Self::Add(a.clone(), b_inner.clone());
                    new.pre_eval(vars);
                    *self = new;
                } else if *a == *b {
                    // x - x => 0
                    *self = Self::Num(0.0_f64);
                } else if let Self::Neg(a_inner) = a.as_ref() {
                    // (-a) - b = -(a + b)
                    let mut new = Self::Neg(Box::new(Self::Add(a_inner.clone(), b.clone())));
                    new.pre_eval(vars);
                    *self = new;
                } else if let Self::Num(y) = b.as_ref()
                    && *y == 0.0_f64
                {
                    *self = *a.clone();
                } else if let Self::Num(x) = a.as_ref()
                    && *x == 0.0_f64
                {
                    let mut new = Self::Neg(b.clone());
                    new.pre_eval(vars);
                    *self = new;
                }
            }
            Self::Mul(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Self::Num(x), Self::Num(y)) => *self = Self::Num(x * y),
                    (Self::Num(x), _) if *x == 0.0_f64 => *self = Self::Num(0.0_f64),
                    (_, Self::Num(y)) if *y == 0.0_f64 => *self = Self::Num(0.0_f64),
                    (Self::Num(x), _) if *x == 1.0_f64 => *self = *b.clone(),
                    (_, Self::Num(y)) if *y == 1.0_f64 => *self = *a.clone(),
                    (Self::Num(x), _) if *x == -1.0_f64 => {
                        let mut new = Self::Neg(b.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Self::Num(y)) if *y == -1.0_f64 => {
                        let mut new = Self::Neg(a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Self::Neg(a_inner), Self::Neg(b_inner)) => {
                        // (-a) * (-b) = a * b
                        let mut new = Self::Mul(a_inner.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    // reassociate: (x * c1) * c2 / (c1 * x) * c2 => x * (c1 * c2)
                    (Self::Mul(left, right), Self::Num(c2)) => {
                        if let Self::Num(c1) = right.as_ref() {
                            let mut new = Self::Mul(left.clone(), Box::new(Self::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Self::Num(c1) = left.as_ref() {
                            let mut new = Self::Mul(right.clone(), Box::new(Self::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    // reassociate: c1 * (x * c2) / c1 * (c2 * x) => x * (c1 * c2)
                    (Self::Num(c1), Self::Mul(left, right)) => {
                        if let Self::Num(c2) = right.as_ref() {
                            let mut new = Self::Mul(left.clone(), Box::new(Self::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        } else if let Self::Num(c2) = left.as_ref() {
                            let mut new = Self::Mul(right.clone(), Box::new(Self::Num(*c1 * *c2)));
                            new.pre_eval(vars);
                            *self = new;
                        }
                    }
                    _ => {}
                }
            }
            Self::Div(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    _ if *a == *b => {
                        // x / x => 1
                        *self = Self::Num(1.0_f64);
                    }
                    (Self::Num(x), Self::Num(y)) => *self = Self::Num(x / y),
                    (_, Self::Num(y)) if *y == 1.0_f64 => *self = *a.clone(),
                    (Self::Num(x), _) if *x == 0.0_f64 => *self = Self::Num(0.0_f64),
                    (_, Self::Num(y)) if *y == -1.0_f64 => {
                        let mut new = Self::Neg(a.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Self::Num(y)) => {
                        // x / c => x * (1/c)
                        let mut new = Self::Mul(a.clone(), Box::new(Self::Num(1.0_f64 / *y)));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Self::Neg(a_inner), Self::Neg(b_inner)) => {
                        // (-a) / (-b) = a / b
                        let mut new = Self::Div(a_inner.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Self::Neg(b_inner)) => {
                        // a / (-b) = -(a / b)
                        let mut new = Self::Neg(Box::new(Self::Div(a.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ => {}
                }
            }
            Self::Pow(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Self::Num(x), Self::Num(y)) => {
                        *self = Self::Num(if *x == 0.0_f64 && *y == 0.0_f64 {
                            1.0_f64
                        } else {
                            x.powf(*y)
                        });
                    }
                    (_, Self::Num(y)) if *y == 0.0_f64 => *self = Self::Num(1.0_f64),
                    (_, Self::Num(y)) if *y == 1.0_f64 => *self = *a.clone(),
                    (Self::Num(x), _) if *x == 1.0_f64 => *self = Self::Num(1.0_f64),
                    (Self::Neg(a_inner), Self::Num(y))
                        if *y == f64::from(*y as i32) && (*y as i32) % 2_i32 == 0_i32 =>
                    {
                        // (-x)^n => x^n  for even integer n
                        let mut new = Self::Pow(a_inner.clone(), b.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Self::Var(_), Self::Num(y)) if *y == 2.0_f64 => {
                        let mut new_node = Self::Mul(a.clone(), a.clone());
                        new_node.pre_eval(vars);
                        *self = new_node;
                    }
                    _ => {}
                }
            }
            Self::Mod(a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                match (a.as_ref(), b.as_ref()) {
                    (Self::Num(x), Self::Num(y)) => *self = Self::Num(x % y),
                    _ if *a == *b => {
                        // x % x => 0
                        *self = Self::Num(0.0_f64);
                    }
                    (Self::Num(x), _) if *x == 0.0_f64 => {
                        // 0 % x => 0
                        *self = Self::Num(0.0_f64);
                    }
                    (Self::Neg(a_inner), Self::Neg(b_inner)) => {
                        // (-a) % (-b) = -(a % b)
                        let mut new =
                            Self::Neg(Box::new(Self::Mod(a_inner.clone(), b_inner.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (_, Self::Neg(b_inner)) => {
                        // a % (-b) = a % b  (sign of divisor doesn't affect result)
                        let mut new = Self::Mod(a.clone(), b_inner.clone());
                        new.pre_eval(vars);
                        *self = new;
                    }
                    (Self::Neg(a_inner), _) => {
                        // (-a) % b = -(a % b)
                        let mut new = Self::Neg(Box::new(Self::Mod(a_inner.clone(), b.clone())));
                        new.pre_eval(vars);
                        *self = new;
                    }
                    _ => {}
                }
            }
            Self::F1(f, a) => {
                a.pre_eval(vars);
                if let Self::Num(x) = a.as_ref() {
                    *self = Self::Num(f.to_fn()(*x));
                } else if let Self::F1(g, inner) = a.as_ref() {
                    match (f, g) {
                        // inverse compositions: f(g(x)) = x
                        (F1::Ln, F1::Exp)
                        | (F1::Exp, F1::Ln)
                        | (F1::Sin, F1::Asin)
                        | (F1::Cos, F1::Acos)
                        | (F1::Tan, F1::Atan) => *self = *inner.clone(),
                        // idempotent: f(f(x)) = f(x)
                        (F1::Abs, F1::Abs)
                        | (F1::Floor, F1::Floor)
                        | (F1::Ceil, F1::Ceil)
                        | (F1::Round, F1::Round)
                        | (F1::Trunc, F1::Trunc) => *self = *a.clone(),
                        _ => {}
                    }
                }
            }
            Self::F2(f, a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                if let (Self::Num(x), Self::Num(y)) = (a.as_ref(), b.as_ref()) {
                    *self = Self::Num(f.to_fn()(*x, *y));
                }
            }
            Self::ExtF1(f, a) => {
                a.pre_eval(vars);
                if let Self::Num(x) = a.as_ref() {
                    *self = Self::Num(f.to_fn()(*x));
                }
            }
            Self::ExtF2(f, a, b) => {
                a.pre_eval(vars);
                b.pre_eval(vars);
                if let (Self::Num(x), Self::Num(y)) = (a.as_ref(), b.as_ref()) {
                    *self = Self::Num(f.to_fn()(*x, *y));
                }
            }
            Self::Let(_slot, expr, body) => {
                expr.pre_eval(vars);
                body.pre_eval(vars);
            }
        }
    }

    /// Return the number of CSE slots required (max slot index + 1).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node: Node = Node::Let(0, Box::new(Node::Num(1.0_f64)), Box::new(Node::Var(0)));
    /// assert_eq!(node.cse_slots(), 1);
    /// ```
    pub fn cse_slots(&self) -> usize {
        match self {
            Self::Let(slot, expr, body) => slot
                .saturating_add(1)
                .max(expr.cse_slots())
                .max(body.cse_slots()),
            Self::CseRef(slot, _) => slot.saturating_add(1),
            Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => a.cse_slots(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b)
            | Self::Mod(a, b)
            | Self::F2(_, a, b)
            | Self::ExtF2(_, a, b) => a.cse_slots().max(b.cse_slots()),
            Self::Num(_) | Self::Var(_) => 0,
        }
    }

    /// Apply common-subexpression elimination to the AST in-place.
    ///
    /// Repeated subtrees are extracted into `Let` bindings and evaluated once.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let mut node: Node = Node::Mul(
    ///     Box::new(Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(1.0_f64)))),
    ///     Box::new(Node::Add(Box::new(Node::Var(0)), Box::new(Node::Num(1.0_f64)))),
    /// );
    /// node.cse();
    /// assert!(node.cse_slots() > 0); // (x+1) extracted into a CSE slot
    /// let mut cache = vec![0.0_f64; node.cse_slots()];
    /// let f = node.compile();
    /// assert_eq!(f(&[4.0_f64], &mut cache), 25.0_f64); // (4+1)^2
    /// ```
    pub fn cse(&mut self) {
        let mut slot = 0_usize;
        while self.cse_one_pass(slot) {
            slot = slot.wrapping_add(1);
        }
    }

    /// Preparation pipeline: `pre_eval`, `CSE`, `compile`.
    /// Returns (`compiled_expr`, `cse_slots`).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::{parse, VarMap};
    /// # struct V;
    /// # impl VarMap for V {
    /// #     fn ndim(&self) -> usize { 3 }
    /// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
    /// #     fn primary_prefix(&self) -> &str { "x" }
    /// # }
    /// let mut node = parse("x*x + x*x", &V).unwrap();
    /// let (f, slots) = node.prepare(&[]);
    /// let mut cache = vec![0.0_f64; slots];
    /// assert_eq!(f(&[3.0_f64], &mut cache), 18.0_f64);
    /// ```
    pub fn prepare(&mut self, vars: &[Option<f64>]) -> (CompiledExpr, usize) {
        self.pre_eval(vars);
        self.cse();
        (self.compile(), self.cse_slots())
    }

    /// Preparation pipeline: `pre_eval`, `CSE`, `compile_multi`.
    /// Returns `compiled_expr_multi`.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::{parse, VarMap};
    /// # struct V;
    /// # impl VarMap for V {
    /// #     fn ndim(&self) -> usize { 3 }
    /// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
    /// #     fn primary_prefix(&self) -> &str { "x" }
    /// # }
    /// let mut node = parse("x*y + z", &V).unwrap();
    /// let multi = node.prepare_multi(&[], &[0, 1, 2]);
    /// let mut cache = vec![0.0_f64; multi.cse_slots];
    /// for g in &multi.groups {
    ///     (g.combined)(&[1.0_f64, 2.0_f64, 3.0_f64], &mut cache);
    /// }
    /// assert_eq!((multi.main)(&[1.0_f64, 2.0_f64, 3.0_f64], &mut cache), 5.0_f64);
    /// ```
    pub fn prepare_multi(
        &mut self,
        vars: &[Option<f64>],
        spatial_dims: &[usize],
    ) -> CompiledExprMulti {
        self.pre_eval(vars);
        self.cse();
        self.compile_multi(spatial_dims)
    }

    /// Single-pass: find one repeated subtree, extract into Let.
    fn cse_one_pass(&mut self, slot: usize) -> bool {
        let pattern_option = {
            let mut nodes: Vec<&Self> = Vec::new();
            self.cse_collect_extractable_candidates(&mut nodes);

            let mut found = None::<Self>;
            'outer: for (i, a) in nodes.iter().enumerate() {
                #[expect(clippy::indexing_slicing)]
                for b in &nodes[i + 1..] {
                    if a == b {
                        found = Some((*a).clone());
                        break 'outer;
                    }
                }
            }
            found
        };

        pattern_option.is_some_and(|pattern| {
            self.cse_replace_all(&pattern, slot);

            let old_self = std::mem::replace(self, Self::Num(0.0_f64));
            *self = Self::Let(slot, Box::new(pattern), Box::new(old_self));

            true
        })
    }

    /// Collects AST nodes that can be safely extracted into a CSE `Let` binding.
    ///
    /// Returns an `IndexSet` of unbound `CseRef` slots this node depends on.
    fn cse_collect_extractable_candidates<'a>(&'a self, out: &mut Vec<&'a Self>) -> IndexSet {
        match self {
            Self::CseRef(slot, _) => IndexSet::singleton(*slot),
            Self::Let(slot, expr, body) => {
                let sa = expr.cse_collect_extractable_candidates(out);
                let sb = body.cse_collect_extractable_candidates(out);
                let mut s = sa | sb;
                s.insert(*slot, false);
                s
            }
            Self::Num(_) | Self::Var(_) => IndexSet::default(),
            Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => {
                let s = a.cse_collect_extractable_candidates(out);
                if s.is_empty() {
                    out.push(self);
                }
                s
            }
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b)
            | Self::Mod(a, b)
            | Self::F2(_, a, b)
            | Self::ExtF2(_, a, b) => {
                let sa = a.cse_collect_extractable_candidates(out);
                let sb = b.cse_collect_extractable_candidates(out);
                let s = sa | sb;
                if s.is_empty() {
                    out.push(self);
                }
                s
            }
        }
    }

    fn cse_replace_all(&mut self, pattern: &Self, slot: usize) {
        if *self == *pattern {
            *self = Self::CseRef(slot, pattern.depends_on());
            return;
        }
        match self {
            Self::Num(_) | Self::Var(_) | Self::CseRef(_, _) => {}
            Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => a.cse_replace_all(pattern, slot),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b)
            | Self::Mod(a, b)
            | Self::F2(_, a, b)
            | Self::ExtF2(_, a, b) => {
                a.cse_replace_all(pattern, slot);
                b.cse_replace_all(pattern, slot);
            }
            Self::Let(_, expr, body) => {
                expr.cse_replace_all(pattern, slot);
                body.cse_replace_all(pattern, slot);
            }
        }
    }

    /// Return the set of variable indices this node depends on.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node: Node = Node::Add(Box::new(Node::Var(0)), Box::new(Node::Var(2)));
    /// let deps = node.depends_on();
    /// assert!(deps.contains(0));
    /// assert!(!deps.contains(1));
    /// assert!(deps.contains(2));
    /// ```
    pub fn depends_on(&self) -> IndexSet {
        match self {
            Self::Num(_) => IndexSet::default(),
            Self::Var(i) => IndexSet::singleton(*i),
            Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => a.depends_on(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b)
            | Self::Mod(a, b)
            | Self::F2(_, a, b)
            | Self::ExtF2(_, a, b) => a.depends_on() | b.depends_on(),
            Self::Let(_, expr, body) => expr.depends_on() | body.depends_on(),
            Self::CseRef(_, deps) => deps.clone(),
        }
    }

    /// Compile the AST into a closure for repeated evaluation.
    ///
    /// Leaf nodes (`Var`, `Num`, `CseRef`) are inlined directly into the parent closure
    /// body, eliminating sub-closure vtbl dispatch for the most common case.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::Node;
    /// let node: Node = Node::Add(Box::new(Node::Num(3.0_f64)), Box::new(Node::Num(4.0_f64)));
    /// let f = node.compile();
    /// assert_eq!(f(&[], &mut []), 7.0_f64);
    /// ```
    pub fn compile(&self) -> CompiledExpr {
        match self {
            Self::Num(v) => {
                let v = *v;
                Box::new(move |_, _| v)
            }
            Self::Var(i) => {
                let i = *i;
                Box::new(move |vars: &[f64], _| *vars.get(i).unwrap_or(&0.0_f64))
            }
            Self::Neg(a) => a.compile_unop(|x| -x),
            #[expect(clippy::shadow_reuse)]
            Self::Add(a, b) => {
                if let (Self::Mul(a, b), c) | (c, Self::Mul(b, a)) = (a.as_ref(), b.as_ref()) {
                    a.compile_terop(b, c, CondMulAdd::cond_mul_add)
                } else {
                    a.compile_binop(b, |x, y| x + y)
                }
            }
            #[expect(clippy::shadow_reuse)]
            Self::Sub(a, b) => {
                if let (Self::Mul(a, b), c) = (a.as_ref(), b.as_ref()) {
                    a.compile_terop(b, c, |x, y, z| x.cond_mul_add(y, -z))
                } else if let (c, Self::Mul(b, a)) = (a.as_ref(), b.as_ref()) {
                    a.compile_terop(b, c, |x, y, z| x.cond_mul_add(-y, z))
                } else {
                    a.compile_binop(b, |x, y| x - y)
                }
            }
            Self::Mul(a, b) => a.compile_binop(b, |x, y| x * y),
            Self::Div(a, b) => a.compile_binop(b, |x, y| x / y),
            Self::Mod(a, b) => a.compile_binop(b, |x, y| x % y),
            Self::Pow(a, b) => a.compile_binop(b, |base, exp| {
                let exp_int = exp as i32;
                if base == 0.0_f64 && exp == 0.0_f64 {
                    1.0_f64
                } else if f64::from(exp_int) == exp {
                    base.powi(exp_int)
                } else {
                    base.powf(exp)
                }
            }),
            Self::F1(f, a) => a.compile_unop(f.to_fn()),
            Self::F2(f, a, b) => a.compile_binop(b, f.to_fn()),
            Self::ExtF1(f, a) => a.compile_unop(f.to_fn()),
            Self::ExtF2(f, a, b) => a.compile_binop(b, f.to_fn()),
            Self::Let(slot, expr, body) => {
                let slot = *slot;
                let expr_fn = expr.compile();
                let body_fn = body.compile();
                Box::new(move |vars: &[f64], cse: &mut [f64]| {
                    *cse.get_mut(slot).unwrap_or(&mut 0.0_f64) = expr_fn(vars, cse);
                    body_fn(vars, cse)
                })
            }
            Self::CseRef(slot, _) => {
                let slot = *slot;
                Box::new(move |_: &[f64], cse: &mut [f64]| *cse.get(slot).unwrap_or(&0.0_f64))
            }
        }
    }

    /// Extract invariant sub-expressions into pre-computed closures.
    ///
    /// `invariant_mask` indicates which variables are invariant at the
    /// current nesting level. Returns `None` if no invariants found.
    pub fn compile_invariants_combined(
        &mut self,
        invariant_mask: &IndexSet,
        slot: &mut usize,
    ) -> Option<CompiledExpr> {
        let mut pieces: Vec<(usize, Self)> = Vec::new();

        self.collect_invariants(invariant_mask, slot, &mut pieces);

        if pieces.is_empty() {
            return None;
        }

        let mut chain = Self::Num(0.0_f64);
        for (slot, node) in pieces.into_iter().rev() {
            chain = Self::Let(slot, Box::new(node), Box::new(chain));
        }

        Some(chain.compile())
    }

    fn collect_invariants(
        &mut self,
        invariant_mask: &IndexSet,
        slot: &mut usize,
        pieces: &mut Vec<(usize, Self)>,
    ) {
        let deps = self.depends_on();
        if deps.is_disjoint(invariant_mask)
            && !matches!(self, Self::Num(_) | Self::Var(_) | Self::CseRef(_, _))
        {
            let node = std::mem::replace(self, Self::CseRef(*slot, deps));
            pieces.push((*slot, node));
            *slot = slot.wrapping_add(1);
        } else {
            match self {
                Self::Num(_) | Self::Var(_) | Self::CseRef(_, _) => {}
                Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => {
                    a.collect_invariants(invariant_mask, slot, pieces);
                }
                Self::Add(a, b)
                | Self::Sub(a, b)
                | Self::Mul(a, b)
                | Self::Div(a, b)
                | Self::Pow(a, b)
                | Self::Mod(a, b)
                | Self::F2(_, a, b)
                | Self::ExtF2(_, a, b) => {
                    a.collect_invariants(invariant_mask, slot, pieces);
                    b.collect_invariants(invariant_mask, slot, pieces);
                }
                Self::Let(ls, expr, body) => {
                    expr.collect_invariants(invariant_mask, slot, pieces);

                    let alias = if let Self::CseRef(rs, deps) = expr.as_ref()
                        && deps.is_disjoint(invariant_mask)
                        && *rs != *ls
                    {
                        Some((*ls, *rs))
                    } else {
                        None
                    };

                    if let Some((ls, rs)) = alias {
                        let mut folded = std::mem::replace(body.as_mut(), Self::Num(0.0_f64));
                        folded.remap_cse_slot(ls, rs);
                        *self = folded;
                        self.collect_invariants(invariant_mask, slot, pieces);
                        return;
                    }

                    body.collect_invariants(invariant_mask, slot, pieces);
                }
            }
        }
    }

    fn fold_cse_aliases(&mut self) {
        if let Self::Let(slot, expr, _) = self
            && let Self::CseRef(other, _) = expr.as_ref()
            && *slot != *other
        {
            let old_slot = *slot;
            let new_slot = *other;
            if let Self::Let(_, _, body) = std::mem::replace(self, Self::Num(0.0_f64)) {
                let mut mapped = *body;
                mapped.remap_cse_slot(old_slot, new_slot);
                mapped.fold_cse_aliases();
                *self = mapped;
            }
        } else {
            match self {
                Self::Num(_) | Self::Var(_) | Self::CseRef(_, _) => {}
                Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => a.fold_cse_aliases(),
                Self::Add(a, b)
                | Self::Sub(a, b)
                | Self::Mul(a, b)
                | Self::Div(a, b)
                | Self::Pow(a, b)
                | Self::Mod(a, b)
                | Self::F2(_, a, b)
                | Self::ExtF2(_, a, b) => {
                    a.fold_cse_aliases();
                    b.fold_cse_aliases();
                }
                Self::Let(_, expr, body) => {
                    expr.fold_cse_aliases();
                    body.fold_cse_aliases();
                }
            }
        }
    }

    fn remap_cse_slot(&mut self, old_slot: usize, new_slot: usize) {
        match self {
            Self::CseRef(slot, _) if *slot == old_slot => *slot = new_slot,
            Self::Num(_) | Self::Var(_) | Self::CseRef(_, _) => {}
            Self::Neg(a) | Self::F1(_, a) | Self::ExtF1(_, a) => {
                a.remap_cse_slot(old_slot, new_slot);
            }
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Pow(a, b)
            | Self::Mod(a, b)
            | Self::F2(_, a, b)
            | Self::ExtF2(_, a, b) => {
                a.remap_cse_slot(old_slot, new_slot);
                b.remap_cse_slot(old_slot, new_slot);
            }
            Self::Let(_, expr, body) => {
                expr.remap_cse_slot(old_slot, new_slot);
                body.remap_cse_slot(old_slot, new_slot);
            }
        }
    }

    /// Compile with multi-level invariant extraction for spatial dimensions.
    pub fn compile_multi(&mut self, spatial_dims: &[usize]) -> CompiledExprMulti {
        if spatial_dims.is_empty() {
            let main = self.compile();
            let cse_slots = self.cse_slots();
            return CompiledExprMulti {
                groups: Vec::new(),
                main,
                cse_slots,
            };
        }
        let masks: Vec<IndexSet> = {
            #[expect(clippy::indexing_slicing)]
            let rest = &spatial_dims[1..];
            let n = rest.len();
            let total = ArithIndexSet(IndexSet::singleton(n));
            let mut masks_by_popcount: Vec<Vec<IndexSet>> = vec![Vec::new(); n + 2];
            // generate only masks containing spatial_dims[0], skipping full set
            for bits in total.range_to().rev().skip(1) {
                #[expect(clippy::indexing_slicing)]
                let mut msk = IndexSet::singleton(spatial_dims[0]);
                for (i, &dim) in rest.iter().enumerate() {
                    if bits.contains(i) {
                        msk.insert(dim, true);
                    }
                }
                #[expect(clippy::indexing_slicing)]
                masks_by_popcount[msk.count_ones()].push(msk);
            }
            masks_by_popcount.into_iter().rev().flatten().collect()
        };

        let mut slot = self.cse_slots();
        let mut groups = Vec::with_capacity(masks.len());

        for mask in masks {
            if let Some(combined) = self.compile_invariants_combined(&mask, &mut slot) {
                let level = spatial_dims
                    .iter()
                    .take_while(|&&d| mask.contains(d))
                    .count();
                groups.push(InvariantGroup { level, combined });
            }
        }

        self.fold_cse_aliases();
        let main = self.compile();
        let cse_slots = self.cse_slots();

        CompiledExprMulti {
            groups,
            main,
            cse_slots,
        }
    }

    #[inline]
    fn inlined(&self) -> InlinedChild {
        match self {
            Self::Num(v) => InlinedChild::Num(*v),
            Self::Var(i) => InlinedChild::Var(*i),
            Self::CseRef(slot, _) => InlinedChild::Cse(*slot),
            _ => InlinedChild::Compound(self.compile()),
        }
    }

    fn compile_unop(&self, f: fn(f64) -> f64) -> CompiledExpr {
        include!(concat!(env!("OUT_DIR"), "/unop.rs"))
    }

    fn compile_binop(&self, b: &Self, op: fn(f64, f64) -> f64) -> CompiledExpr {
        include!(concat!(env!("OUT_DIR"), "/binop.rs"))
    }

    fn compile_terop(&self, b: &Self, c: &Self, op: fn(f64, f64, f64) -> f64) -> CompiledExpr {
        include!(concat!(env!("OUT_DIR"), "/terop.rs"))
    }
}
