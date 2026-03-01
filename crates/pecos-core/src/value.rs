// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! A general-purpose typed value for structured data.
//!
//! `Value` provides a canonical enum for carrying heterogeneous data
//! (strings, numbers, booleans, and nested structures) across the PECOS crate
//! ecosystem. Optional serde/JSON support is available behind feature flags.

use std::collections::HashMap;
use std::fmt;

/// A general-purpose typed value for structured data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Value>),
    Dict(HashMap<String, Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "\"{s}\""),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Dict(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::List(v)
    }
}

impl From<HashMap<String, Value>> for Value {
    fn from(m: HashMap<String, Value>) -> Self {
        Value::Dict(m)
    }
}

#[cfg(feature = "json")]
impl From<serde_json::Value> for Value {
    fn from(json: serde_json::Value) -> Self {
        match json {
            serde_json::Value::Null => Value::String(String::new()),
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else {
                    Value::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(arr) => {
                Value::List(arr.into_iter().map(Value::from).collect())
            }
            serde_json::Value::Object(obj) => {
                Value::Dict(obj.into_iter().map(|(k, v)| (k, Value::from(v))).collect())
            }
        }
    }
}

#[cfg(feature = "json")]
impl From<Value> for serde_json::Value {
    fn from(val: Value) -> Self {
        match val {
            Value::String(s) => serde_json::Value::String(s),
            Value::Int(i) => serde_json::Value::Number(i.into()),
            Value::Float(f) => serde_json::Number::from_f64(f)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            Value::Bool(b) => serde_json::Value::Bool(b),
            Value::List(items) => {
                serde_json::Value::Array(items.into_iter().map(serde_json::Value::from).collect())
            }
            Value::Dict(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(Value::String("hello".into()).to_string(), "\"hello\"");
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::Float(2.78).to_string(), "2.78");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(
            Value::List(vec![Value::Int(1), Value::Int(2)]).to_string(),
            "[1, 2]"
        );
    }

    #[test]
    fn test_from_conversions() {
        assert_eq!(Value::from("hello"), Value::String("hello".into()));
        assert_eq!(Value::from(42i64), Value::Int(42));
        assert_eq!(Value::from(2.78f64), Value::Float(2.78));
        assert_eq!(Value::from(true), Value::Bool(true));
    }

    #[test]
    fn test_nested_structures() {
        let mut inner = HashMap::new();
        inner.insert("x".to_string(), Value::Int(1));
        let val = Value::Dict(inner);

        let list = Value::List(vec![val.clone(), Value::String("test".into())]);
        if let Value::List(items) = &list {
            assert_eq!(items.len(), 2);
            if let Value::Dict(d) = &items[0] {
                assert_eq!(d.get("x"), Some(&Value::Int(1)));
            } else {
                panic!("Expected Dict");
            }
        } else {
            panic!("Expected List");
        }
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_json_round_trip() {
        let val = Value::Dict(HashMap::from([
            ("name".to_string(), Value::String("test".into())),
            ("count".to_string(), Value::Int(42)),
            ("rate".to_string(), Value::Float(2.78)),
            ("active".to_string(), Value::Bool(true)),
            (
                "tags".to_string(),
                Value::List(vec![Value::String("a".into()), Value::String("b".into())]),
            ),
        ]));

        let json: serde_json::Value = val.clone().into();
        let back: Value = json.into();

        // Int and Float round-trip correctly
        assert_eq!(back.clone(), val);

        // Check JSON structure
        let json2: serde_json::Value = back.into();
        assert_eq!(json2["name"], "test");
        assert_eq!(json2["count"], 42);
        assert_eq!(json2["active"], true);
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_json_null_becomes_empty_string() {
        let json = serde_json::Value::Null;
        let val: Value = json.into();
        assert_eq!(val, Value::String(String::new()));
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_json_nested_objects() {
        let json: serde_json::Value = serde_json::json!({
            "outer": {
                "inner": [1, 2, 3]
            }
        });
        let val: Value = json.into();
        if let Value::Dict(map) = &val
            && let Some(Value::Dict(inner_map)) = map.get("outer")
            && let Some(Value::List(items)) = inner_map.get("inner")
        {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Value::Int(1));
            return;
        }
        panic!("Unexpected structure: {val:?}");
    }
}
