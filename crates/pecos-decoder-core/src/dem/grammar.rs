//! Tokenization and validation of individual Stim DEM instruction lines.

use crate::errors::DecoderError;
use std::fmt::{self, Write};

/// A recognized DEM instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An error mechanism.
    Error,
    /// A detector declaration.
    Detector,
    /// An observable declaration.
    LogicalObservable,
    /// The start of a repeat block.
    Repeat,
    /// A detector and coordinate offset.
    ShiftDetectors,
    /// The end of a repeat block.
    EndRepeat,
    /// PECOS observable metadata, enabled explicitly.
    PecosObservable,
    /// PECOS tracked-Pauli metadata, enabled explicitly.
    PecosTrackedPauli,
}

/// A target, retaining component separators and full-width indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A detector index.
    Detector(u64),
    /// An observable index.
    Observable(u64),
    /// A component separator.
    Separator,
    /// A repeat count or detector offset.
    Integer(u64),
    /// A PECOS tracked-Pauli index, enabled explicitly.
    TrackedPauli(u64),
}

/// Explicit permissions for the PECOS DEM superset.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Accept `TP<n>` targets and the two PECOS JSON metadata statements.
    pub pecos_extensions: bool,
}

/// One validated instruction. Numeric arguments are independent of formatting.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// The normalized instruction name.
    pub kind: Kind,
    /// The optional instruction tag.
    pub tag: Option<String>,
    /// Probability or coordinate arguments.
    pub args: Vec<f64>,
    /// Ordered targets, including component separators.
    pub targets: Vec<Target>,
    /// JSON payload of an explicitly enabled PECOS metadata statement.
    pub payload: Option<String>,
    /// Whether a repeat block closes on this line without a body.
    pub empty_repeat: bool,
}

impl Instruction {
    /// Reject instructions requiring loop expansion or detector offsets.
    ///
    /// # Errors
    /// Returns an error for `repeat`, `shift_detectors`, and closing braces.
    pub fn require_flat(&self, consumer: &str) -> Result<(), DecoderError> {
        if matches!(
            self.kind,
            Kind::Repeat | Kind::ShiftDetectors | Kind::EndRepeat
        ) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "{consumer} requires a flattened DEM: `repeat` / `shift_detectors` are not supported. Flatten the DEM first."
            )));
        }
        Ok(())
    }

    /// Iterate over ordered components without interpreting their effects.
    pub fn components(&self) -> impl Iterator<Item = &[Target]> {
        self.targets.split(|target| *target == Target::Separator)
    }
}

/// Convert a target index for consumers using 32-bit indices.
///
/// # Errors
/// Returns an error naming the index and the supported maximum on overflow.
pub fn index_u32(index: u64, kind: &str) -> Result<u32, DecoderError> {
    u32::try_from(index).map_err(|_| index_overflow(index, kind, u64::from(u32::MAX)))
}

/// Report a consumer's index limit.
#[must_use]
pub fn index_overflow(index: u64, kind: &str, maximum: u64) -> DecoderError {
    DecoderError::InvalidConfiguration(format!(
        "{kind} index {index} exceeds the supported maximum {maximum}"
    ))
}

fn invalid(message: impl Into<String>) -> DecoderError {
    DecoderError::InvalidDemSyntax(message.into())
}

fn spacing(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r')
}

fn integer(value: &str) -> Result<u64, DecoderError> {
    if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(invalid(format!(
            "invalid DEM target token: {value}; targets must be separated by spacing"
        )));
    }
    value.parse().map_err(|_| {
        invalid(format!(
            "DEM index {value} exceeds the supported maximum {}",
            u64::MAX
        ))
    })
}

/// Parse a line using only standard Stim DEM instructions and targets.
///
/// # Errors
/// Returns an error for malformed or unknown instructions.
pub fn parse_line(line: &str) -> Result<Option<Instruction>, DecoderError> {
    parse_line_with_options(line, Options::default())
}

/// Parse one line, with explicit permission for PECOS extensions.
///
/// JSON payload syntax is validated by the metadata consumer. This tokenizer
/// retains the payload, including hashes inside JSON strings.
///
/// # Errors
/// Returns an error for malformed or unknown instructions or targets.
pub fn parse_line_with_options(
    line: &str,
    options: Options,
) -> Result<Option<Instruction>, DecoderError> {
    let line = line.trim_start_matches(char::is_whitespace);
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let name_end = line
        .find(|c: char| !c.is_ascii_alphabetic() && c != '_')
        .unwrap_or(line.len());
    let name = line[..name_end].to_ascii_lowercase();
    let kind = match name.as_str() {
        "error" => Kind::Error,
        "detector" => Kind::Detector,
        "logical_observable" => Kind::LogicalObservable,
        "repeat" => Kind::Repeat,
        "shift_detectors" => Kind::ShiftDetectors,
        "pecos_observable" if options.pecos_extensions => Kind::PecosObservable,
        "pecos_tracked_pauli" if options.pecos_extensions => Kind::PecosTrackedPauli,
        "" if line.split('#').next().unwrap_or_default().trim() == "}" => {
            return Ok(Some(Instruction {
                kind: Kind::EndRepeat,
                tag: None,
                args: vec![],
                targets: vec![],
                payload: None,
                empty_repeat: false,
            }));
        }
        _ => {
            let token = line.split(spacing).next().unwrap_or(line);
            return Err(invalid(format!("unrecognized DEM instruction: {token}")));
        }
    };
    let mut rest = &line[name_end..];
    if matches!(kind, Kind::PecosObservable | Kind::PecosTrackedPauli) {
        if !rest.starts_with(spacing) {
            return Err(invalid(
                "PECOS metadata requires spacing before its JSON payload",
            ));
        }
        return Ok(Some(Instruction {
            kind,
            tag: None,
            args: vec![],
            targets: vec![],
            payload: Some(metadata_payload(rest).trim().to_owned()),
            empty_repeat: false,
        }));
    }
    let mut tag = None;
    if let Some(after) = rest.strip_prefix('[') {
        let end = after
            .find(']')
            .ok_or_else(|| invalid("missing ] in DEM tag"))?;
        tag = parse_tag(&after[..end])?;
        rest = &after[end + 1..];
    }
    let mut args = Vec::new();
    if let Some(after) = rest.strip_prefix('(') {
        let end = after
            .find(')')
            .ok_or_else(|| invalid("missing ) in DEM arguments"))?;
        for arg in after[..end].split(',') {
            let value = if arg.trim().is_empty() {
                0.0
            } else {
                arg.trim()
                    .parse::<f64>()
                    .map_err(|_| invalid(format!("invalid DEM numeric argument: {arg}")))?
            };
            if !value.is_finite() {
                return Err(invalid(format!("DEM arguments must be finite: {arg}")));
            }
            args.push(value);
        }
        rest = &after[end + 1..];
    }
    if !rest.is_empty() && !rest.starts_with(spacing) && !rest.starts_with('#') {
        return Err(invalid("DEM targets must be separated by spacing"));
    }
    let mut target_text = rest.split('#').next().unwrap_or_default();
    let mut empty_repeat = false;
    if kind == Kind::Repeat {
        let (count, body) = target_text
            .split_once('{')
            .ok_or_else(|| invalid("repeat requires a count followed by {"))?;
        match body.trim_matches(spacing) {
            "" => {}
            "}" => empty_repeat = true,
            _ => return Err(invalid("repeat requires a count followed by {")),
        }
        target_text = count;
    }
    let tokens = target_text.split(spacing).filter(|token| !token.is_empty());
    let mut targets = Vec::new();
    for token in tokens {
        let target = if matches!(kind, Kind::Repeat | Kind::ShiftDetectors) {
            Target::Integer(integer(token)?)
        } else if token == "^" {
            Target::Separator
        } else if let Some(value) = token.strip_prefix('D').or_else(|| token.strip_prefix('d')) {
            Target::Detector(integer(value)?)
        } else if let Some(value) = token.strip_prefix('L').or_else(|| token.strip_prefix('l')) {
            Target::Observable(integer(value)?)
        } else if options.pecos_extensions && token.starts_with("TP") {
            Target::TrackedPauli(integer(&token[2..])?)
        } else {
            return Err(invalid(format!(
                "invalid DEM target token: {token}; targets must be separated by spacing"
            )));
        };
        targets.push(target);
    }
    match kind {
        Kind::Error => {
            if args.len() != 1 || !(0.0..=1.0).contains(&args[0]) {
                return Err(invalid("error requires exactly one probability in [0, 1]"));
            }
            if targets.first() == Some(&Target::Separator)
                || targets.last() == Some(&Target::Separator)
                || targets
                    .windows(2)
                    .any(|pair| pair == [Target::Separator, Target::Separator])
            {
                return Err(invalid(
                    "DEM separators must have targets on both sides and must not be adjacent",
                ));
            }
        }
        Kind::Detector if !matches!(targets.as_slice(), [Target::Detector(_)]) => {
            return Err(invalid("detector requires exactly one D target"));
        }
        Kind::LogicalObservable
            if !args.is_empty() || !matches!(targets.as_slice(), [Target::Observable(_)]) =>
        {
            return Err(invalid(
                "logical_observable requires exactly one L target and no arguments",
            ));
        }
        Kind::Repeat if !args.is_empty() || !matches!(targets.as_slice(), [Target::Integer(_)]) => {
            return Err(invalid("repeat requires one integer count"));
        }
        Kind::ShiftDetectors if !matches!(targets.as_slice(), [Target::Integer(_)]) => {
            return Err(invalid(
                "shift_detectors requires exactly one integer offset",
            ));
        }
        _ => {}
    }
    Ok(Some(Instruction {
        kind,
        tag,
        args,
        targets,
        payload: None,
        empty_repeat,
    }))
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.kind {
            Kind::Error => "error",
            Kind::Detector => "detector",
            Kind::LogicalObservable => "logical_observable",
            Kind::Repeat => "repeat",
            Kind::ShiftDetectors => "shift_detectors",
            Kind::EndRepeat => "}",
            Kind::PecosObservable => "pecos_observable",
            Kind::PecosTrackedPauli => "pecos_tracked_pauli",
        };
        f.write_str(name)?;
        if let Some(tag) = &self.tag {
            f.write_char('[')?;
            for c in tag.chars() {
                match c {
                    '\n' => f.write_str("\\n")?,
                    '\r' => f.write_str("\\r")?,
                    '\\' => f.write_str("\\B")?,
                    ']' => f.write_str("\\C")?,
                    c => f.write_char(c)?,
                }
            }
            f.write_char(']')?;
        }
        if !self.args.is_empty() {
            f.write_char('(')?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{arg}")?;
            }
            f.write_char(')')?;
        }
        for target in &self.targets {
            match target {
                Target::Detector(id) => write!(f, " D{id}")?,
                Target::Observable(id) => write!(f, " L{id}")?,
                Target::TrackedPauli(id) => write!(f, " TP{id}")?,
                Target::Separator => f.write_str(" ^")?,
                Target::Integer(id) => write!(f, " {id}")?,
            }
        }
        if self.kind == Kind::Repeat {
            f.write_str(if self.empty_repeat { " {}" } else { " {" })?;
        }
        if let Some(payload) = &self.payload {
            write!(f, " {payload}")?;
        }
        Ok(())
    }
}

/// Collect detector and observable indices without combining duplicate targets.
///
/// # Errors
/// Returns an error if an index exceeds the consumer's 32-bit representation.
pub fn target_indices(targets: &[Target]) -> Result<(Vec<u32>, Vec<u32>), DecoderError> {
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    for target in targets {
        match *target {
            Target::Detector(id) => detectors.push(index_u32(id, "detector")?),
            Target::Observable(id) => observables.push(index_u32(id, "observable")?),
            _ => {}
        }
    }
    Ok((detectors, observables))
}

fn parse_tag(text: &str) -> Result<Option<String>, DecoderError> {
    let mut tag = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        tag.push(if c == '\\' {
            match chars.next() {
                Some('n') => '\n',
                Some('r') => '\r',
                Some('B') => '\\',
                Some('C') => ']',
                _ => return Err(invalid("unrecognized escape in DEM tag")),
            }
        } else {
            c
        });
    }
    Ok((!tag.is_empty()).then_some(tag))
}

fn metadata_payload(text: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (index, c) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            '#' if !quoted => return &text[..index],
            _ => {}
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_escapes_decode_and_render() {
        let text = r"error[a\Cb\Bc\nd](0.1) D0";
        let instruction = parse_line(text).unwrap().unwrap();
        assert_eq!(instruction.tag.as_deref(), Some("a]b\\c\nd"));
        assert_eq!(instruction.to_string(), text);
    }
}
