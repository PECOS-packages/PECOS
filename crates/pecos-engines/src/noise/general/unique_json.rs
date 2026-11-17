//! Reject duplicate JSON object keys before any schema map or buffered hook can lose them.
//!
//! Delegate value construction to `serde_json`, including its arbitrary-precision number
//! representation. Only map/sequence traversal is intercepted; nulls remain unchanged.

use serde::de::{DeserializeSeed, IntoDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::BTreeSet;
use std::fmt;

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<serde_json::Value, D::Error> {
    serde_json::Value::deserialize(CheckedDeserializer {
        inner: deserializer,
        path: "$".into(),
    })
}

struct CheckedDeserializer<D> {
    inner: D,
    path: String,
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for CheckedDeserializer<D> {
    type Error = D::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
        self.inner.deserialize_any(CheckedVisitor {
            inner: visitor,
            path: self.path,
        })
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum identifier ignored_any
    }
}

struct CheckedVisitor<V> {
    inner: V,
    path: String,
}

macro_rules! forward_visit {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method<E: serde::de::Error>(self, value: $ty) -> Result<Self::Value, E> {
            self.inner.$method(value)
        })*
    };
}

impl<'de, V: Visitor<'de>> Visitor<'de> for CheckedVisitor<V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    forward_visit! {
        visit_bool(bool), visit_i64(i64), visit_i128(i128), visit_u64(u64), visit_u128(u128),
        visit_f64(f64), visit_str(&str), visit_borrowed_str(&'de str), visit_string(String)
    }

    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        self.inner.visit_unit()
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_map(CheckedMap {
            inner: map,
            path: self.path,
            keys: BTreeSet::new(),
            value_path: String::new(),
        })
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_seq(CheckedSeq {
            inner: seq,
            path: self.path,
            index: 0,
        })
    }
}

struct CheckedSeed<S> {
    inner: S,
    path: String,
}

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for CheckedSeed<S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.deserialize(CheckedDeserializer {
            inner: deserializer,
            path: self.path,
        })
    }
}

struct CheckedMap<A> {
    inner: A,
    path: String,
    keys: BTreeSet<String>,
    value_path: String,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for CheckedMap<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        let Some(key) = self.inner.next_key::<String>()? else {
            return Ok(None);
        };
        if !self.keys.insert(key.clone()) {
            return Err(serde::de::Error::custom(format!(
                "duplicate key {key:?} at {}; configure each key once",
                self.path
            )));
        }
        self.value_path = format!("{}.{key}", self.path);
        seed.deserialize(key.into_deserializer()).map(Some)
    }

    fn next_value_seed<S: DeserializeSeed<'de>>(&mut self, seed: S) -> Result<S::Value, A::Error> {
        self.inner.next_value_seed(CheckedSeed {
            inner: seed,
            path: self.value_path.clone(),
        })
    }
}

struct CheckedSeq<A> {
    inner: A,
    path: String,
    index: usize,
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for CheckedSeq<A> {
    type Error = A::Error;

    fn next_element_seed<S: DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, A::Error> {
        let path = format!("{}[{}]", self.path, self.index);
        self.index += 1;
        self.inner
            .next_element_seed(CheckedSeed { inner: seed, path })
    }
}
