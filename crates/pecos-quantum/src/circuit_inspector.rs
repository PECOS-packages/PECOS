// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Self-contained HTML presentation of an owned inspection snapshot.

use crate::{Attribute, CircuitInspection, InspectionOperation, InspectionSource};
use std::fmt::Write;

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn source_label(operation: &InspectionOperation) -> String {
    match operation.source {
        InspectionSource::TickBatch { tick, batch } => format!("Tick {tick} · batch {batch}"),
        InspectionSource::DagNode { node } => format!("DAG node {node}"),
    }
}

fn attribute_text(value: &Attribute) -> String {
    match value {
        Attribute::Float(value) => value.to_string(),
        Attribute::Int(value) => value.to_string(),
        Attribute::String(value) => value.clone(),
        Attribute::Bool(value) => value.to_string(),
        Attribute::IntList(value) => format!("{value:?}"),
        Attribute::StringList(value) => format!("{value:?}"),
        Attribute::Json(value) => format!("{value:#}"),
    }
}

/// Only a complete, versioned origin participates in linking. Format integers
/// here so browser number precision cannot collapse different trace identities.
fn trace_origin_key(operation: &InspectionOperation) -> Option<String> {
    let Attribute::Json(origin) = operation.metadata.get("_pecos_trace_origin")? else {
        return None;
    };
    if origin.get("format")?.as_str()? != "pecos_qis_lowered_origin_v1" {
        return None;
    }
    let engine = origin.get("engine_trace_id")?.as_u64()?;
    let shot = origin.get("shot_index")?.as_u64()?;
    let chunk = origin.get("chunk_index")?.as_u64()?;
    let operation = origin.get("lowered_op_index")?.as_u64()?;
    Some(format!("qis:{engine}:{shot}:{chunk}:{operation}"))
}

fn detail(out: &mut String, label: &str, value: &str) {
    write!(
        out,
        "<dt>{}</dt><dd><pre>{}</pre></dd>",
        escape_html(label),
        escape_html(value)
    )
    .unwrap();
}

impl CircuitInspection {
    /// Render a self-contained interactive HTML document, without network assets.
    ///
    /// Gate selection links all SVG components belonging to one stored batch or
    /// node to its details. Metadata is escaped text, never executable markup.
    /// Integer identities are formatted in Rust rather than parsed as JavaScript
    /// numbers. Channel payloads are indicated but not displayed.
    ///
    /// Without JavaScript, the diagram and all operation details remain readable.
    /// Notebook clients can embed this document in a sandboxed `srcdoc` iframe.
    #[must_use]
    pub fn to_html(&self) -> String {
        let mut out = String::from(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
             <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
             <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; \
             script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">\
             <title>PECOS circuit inspector</title><style>",
        );
        out.push_str(include_str!("circuit_inspector.css"));
        out.push_str("</style></head><body><main class=\"pecos-inspector\">");
        out.push_str("<header><h1>Circuit inspector</h1><p>Select a gate or a stored operation to inspect its source.</p></header>");
        write!(out, "<p class=\"summary\">{} stored operations · structural circuit view, not elapsed time</p>", self.operations().len()).unwrap();
        out.push_str("<nav class=\"controls\" aria-label=\"Operation selection\" hidden><label>Search <input type=\"search\" aria-label=\"Search operations\" placeholder=\"Gate, source, ID, or metadata\"></label><button type=\"button\" data-action=\"reset-filter\">Reset search</button><label>Operation <select aria-label=\"Operation\"><option value=\"\">Choose an operation</option>");
        for (index, operation) in self.operations().iter().enumerate() {
            let label = format!("{} · {}", source_label(operation), operation.gate.gate_type);
            write!(
                out,
                "<option value=\"{index}\">{}</option>",
                escape_html(&label)
            )
            .unwrap();
        }
        out.push_str("</select></label><button type=\"button\" data-action=\"previous\">Previous</button><button type=\"button\" data-action=\"next\">Next</button><button type=\"button\" data-action=\"clear\">Clear selection</button><label><input type=\"checkbox\" data-action=\"link-trace\" checked>Link trace stages</label><p class=\"filter-status\" role=\"status\"></p><p class=\"legend\">Orange: selected batch. Dashed teal: other stages with the same recorded trace origin. Search dims nonmatches without changing layout.</p></nav>");
        out.push_str("<nav class=\"viewport-controls\" aria-label=\"Diagram view\" hidden><button type=\"button\" data-action=\"zoom-out\" aria-label=\"Zoom out\">−</button><output aria-label=\"Zoom level\">100%</output><button type=\"button\" data-action=\"zoom-in\" aria-label=\"Zoom in\">+</button><button type=\"button\" data-action=\"actual-size\">100%</button><button type=\"button\" data-action=\"fit-width\">Fit width</button><button type=\"button\" data-action=\"show-selection\">Show selection</button></nav>");
        out.push_str("<div class=\"layout\"><section class=\"diagram\" aria-label=\"Circuit diagram\" tabindex=\"0\">");
        // This SVG is produced by PECOS's renderer; all source labels are escaped there.
        out.push_str(self.svg());
        out.push_str("</section><aside aria-label=\"Operation details\"><p class=\"selection-status\" role=\"status\">Select an operation to see its details.</p>");
        out.push_str("<div class=\"trace-controls\" hidden><p class=\"stage-position\"></p><button type=\"button\" data-action=\"previous-stage\">Previous trace stage</button><button type=\"button\" data-action=\"next-stage\">Next trace stage</button><p class=\"stage-hint\"></p></div>");
        if self.operations().is_empty() {
            out.push_str("<p>No stored operations in this view.</p>");
        }
        out.push_str("<div class=\"records\">");
        for operation in self.operations() {
            let origin = trace_origin_key(operation).map_or_else(String::new, |key| {
                format!(" data-pecos-trace-origin=\"{}\"", escape_html(&key))
            });
            let search_ids = operation
                .gate
                .qubits
                .iter()
                .map(|q| format!("q{}", q.0))
                .chain(
                    operation
                        .gate
                        .meas_ids
                        .iter()
                        .map(|id| format!("m{}", id.index())),
                )
                .collect::<Vec<_>>()
                .join(" ");
            write!(
                out,
                "<article data-pecos-record=\"{}\"{origin} data-pecos-search-ids=\"{search_ids}\"><h2>{}</h2><dl>",
                escape_html(&operation.id),
                escape_html(&operation.gate.gate_type.to_string())
            )
            .unwrap();
            detail(&mut out, "Source", &source_label(operation));
            detail(&mut out, "Operation ID", &operation.id);
            let qubits: Vec<_> = operation.gate.qubits.iter().map(|q| q.0).collect();
            detail(&mut out, "Qubits (operand order)", &format!("{qubits:?}"));
            let measurement_ids: Vec<_> = operation
                .gate
                .meas_ids
                .iter()
                .map(|id| id.index())
                .collect();
            detail(
                &mut out,
                "Stored measurement IDs",
                &format!("{measurement_ids:?}"),
            );
            if !operation.gate.angles.is_empty() {
                let radians: Vec<_> = operation
                    .gate
                    .angles
                    .iter()
                    .map(pecos_core::Angle64::to_radians)
                    .collect();
                detail(
                    &mut out,
                    "Angles (radians, approximate)",
                    &format!("{radians:?}"),
                );
                let fractions: Vec<_> = operation
                    .gate
                    .angles
                    .iter()
                    .map(pecos_core::Angle64::fraction)
                    .collect();
                detail(
                    &mut out,
                    "Exact angle numerators (divide by 2^64 turns)",
                    &format!("{fractions:?}"),
                );
            }
            if !operation.gate.params.is_empty() {
                detail(
                    &mut out,
                    "Parameters",
                    &format!("{:?}", operation.gate.params),
                );
            }
            if operation.gate.channel.is_some() {
                detail(
                    &mut out,
                    "Channel payload",
                    "Present; payload contents are not displayed.",
                );
            }
            out.push_str("</dl><h3>Metadata</h3>");
            if operation.metadata.is_empty() {
                out.push_str("<p>No stored metadata.</p>");
            } else {
                out.push_str("<dl>");
                for (key, value) in &operation.metadata {
                    detail(&mut out, key, &attribute_text(value));
                }
                out.push_str("</dl>");
            }
            out.push_str("</article>");
        }
        out.push_str("</div></aside></div><footer>A stored batch may contain several gate instances. IDs are scoped to this source snapshot. Measurement IDs identify results; they are not outcome values.</footer><noscript><p>JavaScript is disabled. All operation details are listed above.</p></noscript><script>");
        out.push_str(include_str!("circuit_inspector.js"));
        out.push_str("</script></main></body></html>");
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{Attribute, DiagramStyle, TickCircuit};

    #[test]
    fn html_keeps_metadata_as_text_and_integer_ids_exact() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]).meta(
            "</dl><script>alert('key')</script>",
            Attribute::String("</pre><img src=x onerror=alert(1)> & \"quoted\"".into()),
        );
        circuit.get_tick_mut(0).unwrap().set_gate_attr(
            0,
            "_pecos_trace_origin",
            Attribute::Json(serde_json::json!({"engine_trace_id": u64::MAX})),
        );
        let html = circuit.inspect(&DiagramStyle::default()).to_html();
        assert!(!html.contains("<script>alert"));
        assert!(!html.contains("<img src=x"));
        assert!(
            html.contains(
                "&lt;/pre&gt;&lt;img src=x onerror=alert(1)&gt; &amp; &quot;quoted&quot;"
            )
        );
        assert!(html.contains("18446744073709551615"));
        assert_eq!(html.matches("<script>").count(), 1);
        assert!(html.contains("data-pecos-record=\"tick:0:batch:0\""));
    }

    #[test]
    fn empty_html_is_explicit_and_deterministic() {
        let snapshot = TickCircuit::new().inspect(&DiagramStyle::default());
        assert_eq!(snapshot.to_html(), snapshot.to_html());
        assert!(
            snapshot
                .to_html()
                .contains("No stored operations in this view.")
        );
    }

    #[test]
    fn trace_links_require_complete_versioned_integer_identity() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        let mut operation = circuit.inspect(&DiagramStyle::default()).operations()[0].clone();
        let valid = serde_json::json!({
            "format": "pecos_qis_lowered_origin_v1", "engine_trace_id": u64::MAX,
            "shot_index": 0, "chunk_index": 1, "lowered_op_index": 2,
        });
        operation
            .metadata
            .insert("_pecos_trace_origin".into(), Attribute::Json(valid.clone()));
        let expected = super::trace_origin_key(&operation).unwrap();
        assert_eq!(expected, "qis:18446744073709551615:0:1:2");
        for field in [
            "engine_trace_id",
            "shot_index",
            "chunk_index",
            "lowered_op_index",
        ] {
            let mut other = valid.clone();
            other[field] = serde_json::json!(3);
            operation
                .metadata
                .insert("_pecos_trace_origin".into(), Attribute::Json(other));
            assert_ne!(super::trace_origin_key(&operation).unwrap(), expected);
            for invalid in [
                serde_json::Value::Null,
                serde_json::json!("1"),
                serde_json::json!(-1),
                serde_json::json!(1.5),
                serde_json::json!(true),
            ] {
                let mut malformed = valid.clone();
                malformed[field] = invalid;
                operation
                    .metadata
                    .insert("_pecos_trace_origin".into(), Attribute::Json(malformed));
                assert!(super::trace_origin_key(&operation).is_none());
            }
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            operation
                .metadata
                .insert("_pecos_trace_origin".into(), Attribute::Json(missing));
            assert!(super::trace_origin_key(&operation).is_none());
        }
        for invalid in [
            serde_json::json!({}),
            serde_json::json!({"format": "unknown"}),
            serde_json::json!([]),
        ] {
            operation
                .metadata
                .insert("_pecos_trace_origin".into(), Attribute::Json(invalid));
            assert!(super::trace_origin_key(&operation).is_none());
        }
        operation.metadata.clear();
        assert!(super::trace_origin_key(&operation).is_none());
    }
}
