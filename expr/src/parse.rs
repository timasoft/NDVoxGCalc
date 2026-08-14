use std::str::FromStr;

use crate::errors::{Error, LexerErrorKind, ParseErrorKind};
use crate::node::Node;
use crate::{ExtF0, ExtF1, ExtF2, F0, F1, F2, LexerToken as Token, NoExtF, VarMap};

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    token_start: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            token_start: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        self.pos = self.pos.wrapping_add(1);
        c
    }

    fn next_token(&mut self) -> Result<Token, Error> {
        // skip whitespace
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }

        self.token_start = self.pos;

        if self.pos >= self.chars.len() {
            return Ok(Token::Eof);
        }

        match self.peek().expect("char must be available") {
            '+' => {
                self.advance();
                Ok(Token::Plus)
            }
            '-' => {
                self.advance();
                Ok(Token::Minus)
            }
            '*' => {
                self.advance();
                let token = if self.peek() == Some('*') {
                    self.advance();
                    Token::Caret
                } else {
                    Token::Star
                };
                Ok(token)
            }
            '/' => {
                self.advance();
                Ok(Token::Slash)
            }
            '%' => {
                self.advance();
                Ok(Token::Percent)
            }
            '|' => {
                self.advance();
                Ok(Token::Pipe)
            }
            '^' => {
                self.advance();
                Ok(Token::Caret)
            }
            '(' => {
                self.advance();
                Ok(Token::LParen)
            }
            ')' => {
                self.advance();
                Ok(Token::RParen)
            }
            ',' => {
                self.advance();
                Ok(Token::Comma)
            }
            c if c.is_ascii_digit() || c == '.' => self.read_number(),
            c if c.is_ascii_alphabetic() => Ok(self.read_ident()),
            c => Err(Error::Lexer {
                col: self.token_start.wrapping_add(1),
                kind: LexerErrorKind::UnexpectedChar(c),
            }),
        }
    }

    fn read_number(&mut self) -> Result<Token, Error> {
        let start = self.pos;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        if self.peek() == Some('.') {
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if let Some(c) = self.peek()
            && (c == 'e' || c == 'E')
        {
            self.advance();
            if self.peek() == Some('+') || self.peek() == Some('-') {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let s: String = self
            .chars
            .get(start..self.pos)
            .unwrap_or_default()
            .iter()
            .collect();
        s.parse::<f64>().map(Token::Num).map_err(|_| Error::Lexer {
            col: self.token_start.wrapping_add(1),
            kind: LexerErrorKind::InvalidNumber(s),
        })
    }

    fn read_ident(&mut self) -> Token {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() {
                self.advance();
            } else {
                break;
            }
        }
        let s: String = self
            .chars
            .get(start..self.pos)
            .unwrap_or_default()
            .iter()
            .collect();
        Token::Ident(s)
    }
}

struct Parser<'a, V: VarMap> {
    lexer: Lexer,
    current: Token,
    current_pos: usize,
    vars: &'a V,
}

impl<'a, V: VarMap> Parser<'a, V> {
    fn new(input: &str, vars: &'a V) -> Result<Self, Error> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        let current_pos = lexer.token_start;
        Ok(Self {
            lexer,
            current,
            current_pos,
            vars,
        })
    }

    fn advance(&mut self) -> Result<(), Error> {
        self.current = self.lexer.next_token()?;
        self.current_pos = self.lexer.token_start;
        Ok(())
    }

    const fn err_at(&self, kind: ParseErrorKind) -> Error {
        Error::Parser {
            col: self.current_pos.wrapping_add(1),
            kind,
        }
    }

    fn parse<EF0, EF1, EF2>(mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        if matches!(self.current, Token::Eof) {
            return Err(Error::Parser {
                col: 0,
                kind: ParseErrorKind::EmptyExpression,
            });
        }
        let node = self.parse_expr::<EF0, EF1, EF2>()?;
        if !matches!(self.current, Token::Eof) {
            return Err(self.err_at(ParseErrorKind::TrailingToken(self.current.clone())));
        }
        Ok(node)
    }

    fn parse_expr<EF0, EF1, EF2>(&mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        let mut left = self.parse_term::<EF0, EF1, EF2>()?;
        while matches!(self.current, Token::Plus | Token::Minus) {
            let op = std::mem::replace(&mut self.current, Token::Eof);
            self.advance()?;
            let right = self.parse_term::<EF0, EF1, EF2>()?;
            left = match op {
                Token::Plus => Node::Add(Box::new(left), Box::new(right)),
                Token::Minus => Node::Sub(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    fn parse_term<EF0, EF1, EF2>(&mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        let mut left = self.parse_unary::<EF0, EF1, EF2>()?;
        while matches!(self.current, Token::Star | Token::Slash | Token::Percent) {
            let op = std::mem::replace(&mut self.current, Token::Eof);
            self.advance()?;
            let right = self.parse_unary::<EF0, EF1, EF2>()?;
            left = match op {
                Token::Star => Node::Mul(Box::new(left), Box::new(right)),
                Token::Slash => Node::Div(Box::new(left), Box::new(right)),
                Token::Percent => Node::Mod(Box::new(left), Box::new(right)),
                _ => unreachable!(),
            };
        }
        Ok(left)
    }

    fn parse_unary<EF0, EF1, EF2>(&mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        if matches!(self.current, Token::Minus) {
            self.advance()?;
            let node = self.parse_unary::<EF0, EF1, EF2>()?;
            Ok(Node::Neg(Box::new(node)))
        } else if matches!(self.current, Token::Plus) {
            self.advance()?;
            self.parse_unary::<EF0, EF1, EF2>()
        } else {
            self.parse_power::<EF0, EF1, EF2>()
        }
    }

    /// right-associative: a^b^c = a^(b^c)
    fn parse_power<EF0, EF1, EF2>(&mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        let left = self.parse_primary::<EF0, EF1, EF2>()?;
        if matches!(self.current, Token::Caret) {
            self.advance()?;
            let right = self.parse_power::<EF0, EF1, EF2>()?;
            Ok(Node::Pow(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_primary<EF0, EF1, EF2>(&mut self) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        match &self.current {
            Token::Num(v) => {
                let val = *v;
                self.advance()?;
                Ok(Node::Num(val))
            }
            Token::Ident(name) => {
                let name_clone = name.clone();
                self.advance()?;
                if matches!(self.current, Token::LParen) {
                    self.parse_function_call::<EF0, EF1, EF2>(&name_clone)
                } else {
                    Ok(resolve_ident::<V, EF0, EF1, EF2>(
                        &name_clone,
                        self.current_pos.wrapping_add(1),
                        self.vars,
                    )?)
                }
            }
            Token::LParen => {
                self.advance()?;
                let node = self.parse_expr::<EF0, EF1, EF2>()?;
                if !matches!(self.current, Token::RParen) {
                    return Err(self.err_at(ParseErrorKind::ExpectedRParen(self.current.clone())));
                }
                self.advance()?;
                Ok(node)
            }
            Token::Pipe => {
                self.advance()?;
                let node = self.parse_expr::<EF0, EF1, EF2>()?;
                if !matches!(self.current, Token::Pipe) {
                    return Err(self.err_at(ParseErrorKind::ExpectedPipe(self.current.clone())));
                }
                self.advance()?;
                Ok(Node::F1(F1::Abs, Box::new(node)))
            }
            _ => Err(self.err_at(ParseErrorKind::UnexpectedToken(self.current.clone()))),
        }
    }

    fn parse_function_call<EF0, EF1, EF2>(&mut self, name: &str) -> Result<Node<EF1, EF2>, Error>
    where
        EF0: ExtF0 + FromStr,
        EF1: ExtF1 + FromStr,
        EF2: ExtF2 + FromStr,
    {
        debug_assert!(matches!(self.current, Token::LParen));
        self.advance()?;

        if matches!(self.current, Token::RParen) {
            self.advance()?;
            return if let Ok(f) = F0::from_str(name) {
                Ok(Node::Num(f.to_num()))
            } else {
                if let Ok(f) = EF0::from_str(name) {
                    return Ok(Node::Num(f.to_num()));
                }
                let kind = if F1::from_str(name).is_ok() || EF1::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 1,
                        found: 0,
                    }
                } else if F2::from_str(name).is_ok() || EF2::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 2,
                        found: 0,
                    }
                } else {
                    ParseErrorKind::UnknownIdentifier(name.to_owned())
                };
                Err(self.err_at(kind))
            };
        }

        let arg1 = self.parse_expr::<EF0, EF1, EF2>()?;

        if matches!(self.current, Token::Comma) {
            self.advance()?;
            let arg2 = self.parse_expr::<EF0, EF1, EF2>()?;
            if !matches!(self.current, Token::RParen) {
                return Err(self.err_at(ParseErrorKind::ExpectedRParen(self.current.clone())));
            }
            self.advance()?;

            if let Ok(f) = F2::from_str(name) {
                Ok(Node::F2(f, Box::new(arg1), Box::new(arg2)))
            } else {
                if let Ok(f) = EF2::from_str(name) {
                    return Ok(Node::ExtF2(f, Box::new(arg1), Box::new(arg2)));
                }
                let kind = if F1::from_str(name).is_ok() || EF1::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 1,
                        found: 2,
                    }
                } else if F0::from_str(name).is_ok() || EF0::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 0,
                        found: 2,
                    }
                } else {
                    ParseErrorKind::UnknownIdentifier(name.to_owned())
                };
                Err(self.err_at(kind))
            }
        } else {
            if !matches!(self.current, Token::RParen) {
                return Err(
                    self.err_at(ParseErrorKind::ExpectedRParenOrComma(self.current.clone()))
                );
            }
            self.advance()?;

            if let Ok(f) = F1::from_str(name) {
                Ok(Node::F1(f, Box::new(arg1)))
            } else {
                if let Ok(f) = EF1::from_str(name) {
                    return Ok(Node::ExtF1(f, Box::new(arg1)));
                }
                let kind = if F0::from_str(name).is_ok() || EF0::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 0,
                        found: 1,
                    }
                } else if F2::from_str(name).is_ok() || EF2::from_str(name).is_ok() {
                    ParseErrorKind::FunctionArgCount {
                        name: name.to_owned(),
                        expected: 2,
                        found: 1,
                    }
                } else {
                    ParseErrorKind::UnknownIdentifier(name.to_owned())
                };
                Err(self.err_at(kind))
            }
        }
    }
}

fn resolve_ident<V: VarMap, EF0, EF1, EF2>(
    name: &str,
    col: usize,
    vars: &V,
) -> Result<Node<EF1, EF2>, Error>
where
    EF0: ExtF0 + FromStr,
    EF1: ExtF1,
    EF2: ExtF2,
{
    if let Ok(f) = F0::from_str(name) {
        return Ok(Node::Num(f.to_num()));
    }

    if let Ok(f) = EF0::from_str(name) {
        return Ok(Node::Num(f.to_num()));
    }

    if let Some(idx) = vars.resolve_alias(name) {
        return Ok(Node::Var(idx));
    }

    let primary = vars.primary_prefix();
    if let Some(rest) = name.strip_prefix(primary)
        && !rest.is_empty()
        && let Ok(idx) = rest.parse::<usize>()
    {
        if idx < vars.ndim() {
            return Ok(Node::Var(idx));
        }
        return Err(Error::Parser {
            col,
            kind: ParseErrorKind::VarOutOfRange {
                name: name.to_owned(),
                max: vars.ndim().saturating_sub(1),
            },
        });
    }

    Err(Error::Parser {
        col,
        kind: ParseErrorKind::UnknownIdentifier(name.to_owned()),
    })
}

/// Parse an expression string into a `Node` AST.
///
/// # Errors
/// Returns [`Error::Lexer`] for invalid characters/numbers and [`Error::Parser`]
/// for malformed expressions (unknown identifiers, trailing tokens, wrong
/// function arity, missing delimiters, out-of-range variables).
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
/// let node = parse("x * x + y", &V).unwrap();
/// let mut cache = vec![0.0; node.cse_slots()];
/// let result = node.compile()(&[3.0, 4.0], &mut cache);
/// assert_eq!(result, 13.0);
/// ```
pub fn parse<V: VarMap>(expr_str: &str, vars: &V) -> Result<Node, Error> {
    parse_with_ext::<V, NoExtF, NoExtF, NoExtF>(expr_str, vars)
}

/// Validate an expression string without producing a compiled result.
///
/// # Errors
/// Returns [`Error::Lexer`] for invalid characters/numbers and [`Error::Parser`]
/// for malformed expressions (empty input, unknown identifiers, trailing
/// tokens, wrong function arity, missing delimiters, out-of-range variables).
///
/// # Examples
/// ```
/// # use hypervox_expr::{validate, VarMap};
/// # struct V;
/// # impl VarMap for V {
/// #     fn ndim(&self) -> usize { 3 }
/// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
/// #     fn primary_prefix(&self) -> &str { "x" }
/// # }
/// assert!(validate("x + y", &V).is_ok());
/// assert!(validate("x +", &V).is_err());
/// ```
pub fn validate<V: VarMap>(expr_str: &str, vars: &V) -> Result<(), Error> {
    validate_with_ext::<V, NoExtF, NoExtF, NoExtF>(expr_str, vars)
}

/// Parse an expression string into a `Node` AST, with support for custom extension functions.
///
/// # Errors
/// Returns [`Error::Lexer`] for invalid characters/numbers and [`Error::Parser`]
/// for malformed expressions (unknown identifiers, trailing tokens, wrong
/// function arity, missing delimiters, out-of-range variables).
///
/// # Examples
/// ```
/// # use hypervox_expr::{parse_with_ext, VarMap, NoExtF};
/// # use hypervox_expr::define_ext_f0;
/// # use hypervox_expr::define_ext_f1;
/// # struct V;
/// # impl VarMap for V {
/// #     fn ndim(&self) -> usize { 3 }
/// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
/// #     fn primary_prefix(&self) -> &str { "x" }
/// # }
/// define_ext_f0!(MyF0, Tau => "tau" = std::f64::consts::TAU);
/// define_ext_f1!(MyF1, Cube => "cube" = |x| x * x * x);
/// let node = parse_with_ext::<V, MyF0, MyF1, NoExtF>("tau + cube(y)", &V).unwrap();
/// let mut cache = vec![0.0; node.cse_slots()];
/// let result = node.compile()(&[0.0, 2.0, 0.0], &mut cache);
/// assert!((result - (std::f64::consts::TAU + 8.0)).abs() < 1e-9);
/// ```
pub fn parse_with_ext<V, EF0, EF1, EF2>(expr_str: &str, vars: &V) -> Result<Node<EF1, EF2>, Error>
where
    V: VarMap,
    EF0: ExtF0 + FromStr,
    EF1: ExtF1 + FromStr,
    EF2: ExtF2 + FromStr,
{
    let parser = Parser::new(expr_str, vars)?;
    parser.parse::<EF0, EF1, EF2>()
}

/// Validate an expression string without producing a compiled result, with support for custom extension functions.
///
/// # Errors
/// Returns [`Error::Lexer`] for invalid characters/numbers and [`Error::Parser`]
/// for malformed expressions (empty input, unknown identifiers, trailing
/// tokens, wrong function arity, missing delimiters, out-of-range variables).
///
/// # Examples
/// ```
/// # use hypervox_expr::{validate_with_ext, VarMap, NoExtF};
/// # use hypervox_expr::define_ext_f0;
/// # struct V;
/// # impl VarMap for V {
/// #     fn ndim(&self) -> usize { 3 }
/// #     fn resolve_alias(&self, name: &str) -> Option<usize> { match name { "x" => Some(0), "y" => Some(1), "z" => Some(2), _ => None } }
/// #     fn primary_prefix(&self) -> &str { "x" }
/// # }
/// define_ext_f0!(MyF0, Tau => "tau" = std::f64::consts::TAU);
/// assert!(validate_with_ext::<V, MyF0, NoExtF, NoExtF>("tau + x", &V).is_ok());
/// assert!(validate_with_ext::<V, MyF0, NoExtF, NoExtF>("tau +", &V).is_err());
/// ```
pub fn validate_with_ext<V, EF0, EF1, EF2>(expr_str: &str, vars: &V) -> Result<(), Error>
where
    V: VarMap,
    EF0: ExtF0 + FromStr,
    EF1: ExtF1 + FromStr,
    EF2: ExtF2 + FromStr,
{
    let trimmed = expr_str.trim();
    if trimmed.is_empty() {
        return Err(Error::Parser {
            col: 0,
            kind: ParseErrorKind::EmptyExpression,
        });
    }
    parse_with_ext::<V, EF0, EF1, EF2>(trimmed, vars).map(|_| ())
}
