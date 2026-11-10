// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Hosted-operation relationships in traced circuits.

use crate::{Attribute, TickCircuit};
use std::fmt::Write;

/// Extracted gate values, in circuit traversal order.
#[derive(Clone, Debug)]
pub struct HostedGateRecord {
    pub tick_index: usize,
    pub gate_index: usize,
    pub gate_name: String,
    pub qubits: Vec<usize>,
    pub host_id: String,
    pub local_role: String,
}

/// Validation policy and diagnostic context.
#[derive(Clone, Debug)]
pub struct HostedOperationOptions<'a> {
    pub max_tick_separation: Option<u64>,
    pub require_shared_qubit: bool,
    pub require_host_after_local: bool,
    pub require_unique_host_id: bool,
    pub host_id_key: &'a str,
    pub context: &'a str,
}

impl Default for HostedOperationOptions<'_> {
    fn default() -> Self {
        Self {
            max_tick_separation: None,
            require_shared_qubit: true,
            require_host_after_local: true,
            require_unique_host_id: false,
            host_id_key: "host_id",
            context: "hosted operation validation",
        }
    }
}

/// Indices into the input records and signed host distance.
#[derive(Debug, PartialEq, Eq)]
pub struct HostedOperationBinding {
    pub local: usize,
    pub host: usize,
    pub tick_separation: i128,
}

/// Invalid hosted metadata.
#[derive(Debug)]
pub struct HostedOperationError(String);

impl std::fmt::Display for HostedOperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HostedOperationError {}

fn location(record: &HostedGateRecord) -> String {
    format!(
        "{}@t{}/g{}",
        record.gate_name, record.tick_index, record.gate_index
    )
}

/// Validate extracted records, preserving traversal order for host selection.
///
/// # Errors
/// Returns the first policy violation, with the supplied context.
pub fn validate_hosted_operations(
    records: &[HostedGateRecord],
    options: &HostedOperationOptions<'_>,
) -> Result<Vec<HostedOperationBinding>, HostedOperationError> {
    let context = options.context;
    let key = format!("{:?}", options.host_id_key);
    let error = |message| Err(HostedOperationError(message));
    if options.require_unique_host_id {
        for record in records
            .iter()
            .filter(|r| r.local_role.is_empty() && !r.host_id.is_empty())
        {
            let hosts: Vec<_> = records
                .iter()
                .filter(|r| r.local_role.is_empty() && r.host_id == record.host_id)
                .collect();
            if hosts.len() > 1 {
                let mut locations = hosts
                    .iter()
                    .take(4)
                    .map(|r| location(r))
                    .collect::<Vec<_>>()
                    .join(", ");
                if hosts.len() > 4 {
                    write!(locations, ", ... (+{} more)", hosts.len() - 4).unwrap();
                }
                return error(format!(
                    "{context}: hosted metadata key {key} is ambiguous because host_id {:?} appears on {} host gates ({locations}). Strict hosted-operation validation requires host ids to identify one source host gate; include invocation-scoped metadata before validating ordering or tick separation.",
                    record.host_id,
                    hosts.len()
                ));
            }
        }
    }
    let mut bindings = Vec::new();
    for (local_index, local) in records
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.local_role.is_empty())
    {
        let prefix = format!(
            "{context}: hosted local gate {} on qubits {:?}",
            location(local),
            local.qubits
        );
        let role = format!("{:?}", local.local_role);
        let id = format!("{:?}", local.host_id);
        if local.host_id.is_empty() {
            return error(format!(
                "{prefix} has local_role {role} but no non-empty {key} metadata."
            ));
        }
        let candidates: Vec<_> = records
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                r.host_id == local.host_id
                    && r.local_role.is_empty()
                    && (!options.require_shared_qubit
                        || r.qubits.iter().any(|q| local.qubits.contains(q)))
            })
            .collect();
        let Some(&(last_index, last)) = candidates.last() else {
            let shared = if options.require_shared_qubit {
                " sharing a qubit"
            } else {
                ""
            };
            return error(format!(
                "{prefix} with host_id {id} and local_role {role} has no host gate{shared} carrying the same host_id."
            ));
        };
        let later = candidates
            .iter()
            .find(|(_, r)| (r.tick_index, r.gate_index) > (local.tick_index, local.gate_index));
        if options.require_host_after_local && later.is_none() {
            return error(format!(
                "{prefix} with host_id {id} and local_role {role} has matching host metadata only before it; nearest host is {}. This indicates source-host ordering drift in the traced runtime schedule.",
                location(last)
            ));
        }
        let (host_index, host) = later.copied().unwrap_or((last_index, last));
        let separation = host.tick_index as i128 - local.tick_index as i128;
        if let Some(max) = options.max_tick_separation
            && separation.unsigned_abs() > u128::from(max)
        {
            return error(format!(
                "{prefix} with host_id {id} binds to {} with signed tick separation {separation}, exceeding max_tick_separation={max}.",
                location(host)
            ));
        }
        bindings.push(HostedOperationBinding {
            local: local_index,
            host: host_index,
            tick_separation: separation,
        });
    }
    Ok(bindings)
}

/// Host IDs and roles are text; non-string values only need equal text for equal values.
fn metadata_text(value: Option<&Attribute>) -> String {
    match value {
        None | Some(Attribute::Json(serde_json::Value::Null)) => String::new(),
        Some(Attribute::String(s) | Attribute::Json(serde_json::Value::String(s))) => s.clone(),
        Some(Attribute::Int(n)) => n.to_string(),
        Some(Attribute::Float(n)) => n.to_string(),
        Some(Attribute::Bool(b)) => b.to_string(),
        Some(Attribute::IntList(v)) => format!("{v:?}"),
        Some(Attribute::StringList(v)) => format!("{v:?}"),
        Some(Attribute::Json(v)) => v.to_string(),
    }
}

/// Extract hosted values from a native circuit using caller-selected metadata keys.
#[must_use]
pub fn hosted_gate_records(
    circuit: &TickCircuit,
    host_id_key: &str,
    local_role_key: &str,
) -> Vec<HostedGateRecord> {
    let mut records = Vec::new();
    for (tick_index, tick) in circuit.ticks().iter().enumerate() {
        for (gate_index, gate) in tick.gate_batches().iter().enumerate() {
            let host_id = metadata_text(tick.get_gate_attr(gate_index, host_id_key));
            let local_role = metadata_text(tick.get_gate_attr(gate_index, local_role_key));
            if !host_id.is_empty() || !local_role.is_empty() {
                records.push(HostedGateRecord {
                    tick_index,
                    gate_index,
                    gate_name: gate.gate_type.to_string(),
                    qubits: gate.qubits.iter().map(|q| q.0).collect(),
                    host_id,
                    local_role,
                });
            }
        }
    }
    records
}

/// Validate native circuit metadata.
///
/// Binding indices refer to [`hosted_gate_records`] with these same metadata keys.
///
/// # Errors
/// Returns the first hosted policy violation.
pub fn validate_hosted_circuit(
    circuit: &TickCircuit,
    local_role_key: &str,
    options: &HostedOperationOptions<'_>,
) -> Result<Vec<HostedOperationBinding>, HostedOperationError> {
    validate_hosted_operations(
        &hosted_gate_records(circuit, options.host_id_key, local_role_key),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        tick: usize,
        gate: usize,
        name: &str,
        qubits: &[usize],
        id: &str,
        role: &str,
    ) -> HostedGateRecord {
        HostedGateRecord {
            tick_index: tick,
            gate_index: gate,
            gate_name: name.into(),
            qubits: qubits.to_vec(),
            host_id: id.into(),
            local_role: role.into(),
        }
    }

    fn local(tick: usize) -> HostedGateRecord {
        record(tick, 0, "H", &[0], "host:a", "basis_prefix")
    }
    fn host(tick: usize) -> HostedGateRecord {
        record(tick, 0, "SZZ", &[0, 1], "host:a", "")
    }

    #[test]
    fn rust_diagnostic_formatting() {
        let records = [record(3, 1, "H", &[2, 5], "", "a\"b\n")];
        assert_eq!(
            validate_hosted_operations(&records, &HostedOperationOptions::default())
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t3/g1 on qubits [2, 5] has local_role \"a\\\"b\\n\" but no non-empty \"host_id\" metadata."
        );
    }

    #[test]
    fn full_width_tick_and_qubit_values() {
        let mut records = [local(0), host(usize::MAX)];
        for record in &mut records {
            record.qubits = vec![usize::MAX];
        }
        let mut options = HostedOperationOptions {
            max_tick_separation: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(
            validate_hosted_operations(&records, &options).unwrap()[0].tick_separation,
            usize::MAX as i128
        );
        records[0].tick_index = usize::MAX;
        records[1].tick_index = 0;
        options.require_host_after_local = false;
        assert_eq!(
            validate_hosted_operations(&records, &options).unwrap()[0].tick_separation,
            -(usize::MAX as i128)
        );
        options.max_tick_separation = Some(0);
        assert!(validate_hosted_operations(&records, &options).is_err());
    }

    #[test]
    fn binds_local_to_later_host() {
        let bindings =
            validate_hosted_operations(&[local(0), host(2)], &HostedOperationOptions::default())
                .unwrap();
        assert_eq!(
            bindings,
            vec![HostedOperationBinding {
                local: 0,
                host: 1,
                tick_separation: 2
            }]
        );
    }

    #[test]
    fn selects_later_shared_host_record() {
        let records = [
            host(0),
            local(1),
            record(2, 0, "RZ", &[9], "host:a", ""),
            host(3),
            host(4),
        ];
        for require_host_after_local in [true, false] {
            let options = HostedOperationOptions {
                require_host_after_local,
                ..Default::default()
            };
            assert_eq!(
                validate_hosted_operations(&records, &options).unwrap()[0].host,
                3
            );
        }
    }

    #[test]
    fn can_require_unique_host_ids() {
        let options = HostedOperationOptions {
            require_unique_host_id: true,
            ..Default::default()
        };
        let expected = "hosted operation validation: hosted metadata key \"host_id\" is ambiguous because host_id \"host:a\" appears on 2 host gates (SZZ@t1/g0, SZZ@t3/g0). Strict hosted-operation validation requires host ids to identify one source host gate; include invocation-scoped metadata before validating ordering or tick separation.";
        assert_eq!(
            validate_hosted_operations(&[local(0), host(1), local(2), host(3)], &options)
                .unwrap_err()
                .to_string(),
            expected
        );
        assert_eq!(
            validate_hosted_operations(&[host(1), host(3)], &options)
                .unwrap_err()
                .to_string(),
            expected
        );
    }

    #[test]
    fn unique_host_ids_allow_many_locals() {
        let options = HostedOperationOptions {
            require_unique_host_id: true,
            ..Default::default()
        };
        let records = [
            local(0),
            record(0, 1, "S", &[0], "host:a", "basis_prefix"),
            host(1),
        ];
        let bindings = validate_hosted_operations(&records, &options).unwrap();
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|b| b.host == 2));
    }

    #[test]
    fn can_bind_without_shared_qubit_requirement() {
        let options = HostedOperationOptions {
            require_shared_qubit: false,
            ..Default::default()
        };
        let records = [local(0), record(1, 0, "RZ", &[9], "host:a", "")];
        assert_eq!(
            validate_hosted_operations(&records, &options).unwrap()[0].host,
            1
        );
    }

    #[test]
    fn rejects_ordering_drift_by_default() {
        assert_eq!(
            validate_hosted_operations(&[host(0), local(1)], &HostedOperationOptions::default())
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t1/g0 on qubits [0] with host_id \"host:a\" and local_role \"basis_prefix\" has matching host metadata only before it; nearest host is SZZ@t0/g0. This indicates source-host ordering drift in the traced runtime schedule."
        );
    }

    #[test]
    fn can_bind_prior_host_for_metadata_shape_audit() {
        let options = HostedOperationOptions {
            require_host_after_local: false,
            ..Default::default()
        };
        let binding = validate_hosted_operations(&[host(0), host(1), local(2)], &options)
            .unwrap()
            .remove(0);
        assert_eq!(binding.host, 1);
        assert_eq!(binding.tick_separation, -1);
    }

    #[test]
    fn rejects_missing_local_host_id() {
        let records = [record(0, 0, "H", &[0], "", "basis_prefix")];
        assert_eq!(
            validate_hosted_operations(&records, &HostedOperationOptions::default())
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t0/g0 on qubits [0] has local_role \"basis_prefix\" but no non-empty \"host_id\" metadata."
        );
    }

    #[test]
    fn rejects_unbound_local() {
        let records = [
            record(0, 0, "H", &[0], "missing", "basis_prefix"),
            record(1, 0, "SZZ", &[0, 1], "other", ""),
        ];
        assert_eq!(
            validate_hosted_operations(&records, &HostedOperationOptions::default())
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t0/g0 on qubits [0] with host_id \"missing\" and local_role \"basis_prefix\" has no host gate sharing a qubit carrying the same host_id."
        );
        let options = HostedOperationOptions {
            require_shared_qubit: false,
            ..Default::default()
        };
        assert!(
            validate_hosted_operations(&records, &options)
                .unwrap_err()
                .to_string()
                .contains("has no host gate carrying")
        );
    }

    #[test]
    fn rejects_large_tick_separation() {
        let options = HostedOperationOptions {
            max_tick_separation: Some(2),
            ..Default::default()
        };
        assert_eq!(
            validate_hosted_operations(&[local(0), host(3)], &options)
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t0/g0 on qubits [0] with host_id \"host:a\" binds to SZZ@t3/g0 with signed tick separation 3, exceeding max_tick_separation=2."
        );
        assert!(validate_hosted_operations(&[local(0), host(2)], &options).is_ok());
    }

    #[test]
    fn can_check_separation_without_order_guard() {
        let options = HostedOperationOptions {
            max_tick_separation: Some(0),
            require_host_after_local: false,
            ..Default::default()
        };
        assert_eq!(
            validate_hosted_operations(&[host(0), local(1)], &options)
                .unwrap_err()
                .to_string(),
            "hosted operation validation: hosted local gate H@t1/g0 on qubits [0] with host_id \"host:a\" binds to SZZ@t0/g0 with signed tick separation -1, exceeding max_tick_separation=0."
        );
    }

    #[test]
    fn same_tick_order() {
        let records = [local(0), record(0, 1, "SZZ", &[0, 1], "host:a", "")];
        assert_eq!(
            validate_hosted_operations(&records, &HostedOperationOptions::default()).unwrap()[0]
                .tick_separation,
            0
        );
    }

    #[test]
    fn native_extraction_uses_custom_keys() {
        let mut circuit = TickCircuit::new();
        circuit
            .tick()
            .h(&[0])
            .meta("id", Attribute::String("host:a".into()))
            .meta("role", Attribute::String("basis_prefix".into()));
        circuit
            .tick()
            .cx(&[(0, 1)])
            .meta("id", Attribute::String("host:a".into()));
        let options = HostedOperationOptions {
            host_id_key: "id",
            ..Default::default()
        };
        assert_eq!(
            validate_hosted_circuit(&circuit, "role", &options)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn repeated_hosts_report_first_id_and_truncate_locations() {
        let mut records = vec![
            host(0),
            record(1, 0, "X", &[9], "other", ""),
            record(2, 0, "X", &[9], "other", ""),
        ];
        records.extend((3..8).map(host));
        let options = HostedOperationOptions {
            require_unique_host_id: true,
            ..Default::default()
        };
        let message = validate_hosted_operations(&records, &options)
            .unwrap_err()
            .to_string();
        assert!(message.contains("host_id \"host:a\" appears on 6 host gates"));
        assert!(message.contains("SZZ@t0/g0, SZZ@t3/g0, SZZ@t4/g0, SZZ@t5/g0, ... (+2 more)"));
    }
}
