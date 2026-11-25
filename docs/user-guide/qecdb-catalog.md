# Loading codes from qecdb

PECOS can load abstract stabilizer-code records from [qecdb.org](https://qecdb.org/). A catalog record contains
algebraic code data and reported metadata. It does **not** provide a syndrome-extraction protocol, decoder, physical
geometry, or device implementation.

```hidden-python
from pecos.qec.catalogs import qecdb

code_id = "670588ce859a107b3f7601fd"
_FIVE_QUBIT_PAGE = b"""<table>
<tr><td>name</td><td>[[5, 1, 3]]</td></tr>
<tr><td>n</td><td>5</td></tr><tr><td>k</td><td>1</td></tr><tr><td>d</td><td>3</td></tr>
<tr><td>css</td><td>False</td></tr><tr><td>gf4</td><td>True</td></tr>
<tr><td>H</td><td><tt>XZZXI<br>IXZZX<br>XIXZZ<br>ZXIXZ</tt></td></tr>
<tr><td>T</td><td><tt>ZIZII<br>ZXIXI<br>ZXIII<br>ZXIIX</tt></td></tr>
<tr><td>L</td><td><tt>XYXII<br>ZXZII</tt></td></tr>
<tr><td>_id</td><td>670588ce859a107b3f7601fd</td></tr>
</table>"""
record = qecdb.parse_html(_FIVE_QUBIT_PAGE)
```

## Fetching a record

<!--skip: requires live network access; the same conversion is tested offline below-->
```python
from pecos.qec.catalogs import qecdb

record = qecdb.fetch("670588ce859a107b3f7601fd")
code = record.to_stabilizer_code()

print(record.reported_parameters)  # (5, 1, 3)
print(code.code_parameters())  # [[5, 1]]
print(code.logical_operators())  # a basis computed by PECOS
```

The same conversion is reproducible without network access from a saved response:

<!--test-name: qecdb_primary_offline_example-->
<!--expect-output-block-->
```python
record = qecdb.parse_html(_FIVE_QUBIT_PAGE)
code = record.to_stabilizer_code()

print(record.reported_parameters)
print(code.code_parameters())
print(len(code.logical_operators()))
```

```output
(5, 1, 3)
[[5, 1]]
2
```

The qecdb distance belongs to `record.reported_distance`. PECOS's lightweight `StabilizerCode` does not silently treat
that reported value as an independently computed result. Similarly, `record.reported_logical_operators` retains the
representatives supplied by qecdb, while `code.logical_operators()` may compute a different valid basis.

The record also exposes qecdb's stabilizers (`H`), destabilizers (`T`), logical operators (`L`), CSS and GF(4) flags,
family/name fields, and other recognized metadata. `record.raw_fields` preserves unrecognized fields for forward
compatibility.

## Record fields

`QecdbRecord` is immutable. Tuples preserve qecdb's operator-row order, while `raw_fields` is a read-only mapping.

| PECOS property | qecdb field | Meaning |
|---|---|---|
| `id` | `_id` | Stable catalog identifier |
| `n`, `k` | `n`, `k` | Reported physical and logical qubit counts |
| `reported_distance` | `d` | Reported distance; not recomputed during import |
| `reported_dx`, `reported_dz` | `dx`, `dz` | Reported directional distances |
| `stabilizers` | `H` | Stabilizer generators |
| `logical_operators` | `L` | Reported logical Pauli representatives |
| `destabilizers` | `T` | Destabilizer/pure-error partners—not transversal-T gates |
| `is_css`, `is_gf4`, `is_self_dual` | `css`, `gf4`, `selfdual` | Reported structural flags |
| `name`, `css_name`, `family`, `code_type` | `name`, `cssname`, `family`, `tp` | Catalog classification metadata |
| `automorphisms` | `automorphisms` | Reported automorphism-group order |
| `source_url`, `retrieved_at` | — | PECOS retrieval provenance |
| `source_sha256` | — | Digest of the exact HTML response parsed by PECOS |
| `raw_fields` | all table fields | Complete string-valued source mapping, including unknown fields |

## Validation

Fetching parses the public record but does not perform an expensive distance computation. Validate algebraic claims
explicitly, or rely on the validation performed by `to_stabilizer_code()`:

```python
report = record.validate()
for warning in report.warnings:
    print(warning.name, warning.detail)

report.raise_if_invalid()
code = record.to_stabilizer_code()
```

Validation checks Pauli formatting, stabilizer commutation and rank, supplied logical and destabilizer relationships,
complete-tableau rank when possible, inexpensive CSS claims, and basic distance consistency. It does not prove the
reported distance.

Logical pairings are verified symplectically without assuming that qecdb stores logical operators in a particular row
order.

`fetch()` only retrieves and parses. It does not perform complete algebraic validation until `validate()` or the
default-validating `to_stabilizer_code()` is called. Avoid `to_stabilizer_code(validate=False)` for untrusted or newly
retrieved records.

## Errors

All integration-specific exceptions inherit from `QecdbError`:

| Exception | Meaning |
|---|---|
| `QecdbInvalidIdError` | The requested or returned ID is not 24 hexadecimal characters |
| `QecdbFetchError` | Network, redirect, response-size, or cache I/O failure |
| `QecdbCacheMissError` | Offline mode was requested without a cached response |
| `QecdbParseError` | The public page no longer matches the supported schema |
| `QecdbValidationError` | Parsed algebraic data or metadata claims are inconsistent |

Validation errors retain their complete report:

<!--test-name: qecdb_error_handling-->
<!--expect-output: qecdb ID must be exactly 24 hexadecimal characters-->
```python
try:
    qecdb.fetch("not-a-qecdb-id")
except qecdb.QecdbInvalidIdError as error:
    print(error)
```

## Cache and offline use

`fetch()` caches the original HTML in a parser-versioned directory. Once fetched, a record can be required to load
without network access:

<!--skip: requires a previously populated user cache-->
```python
record = qecdb.fetch(code_id, offline=True)
```

Use `refresh=True` to replace an existing entry. Use `cache_dir=...` to select a project-specific cache, or set
`PECOS_CACHE_DIR`. The platform cache directory is used by default.

Cache entries do not expire automatically. The default behavior is:

- reuse an existing entry;
- fetch and atomically populate the cache when no entry exists;
- bypass both reads and writes with `cache=False`;
- require a cache hit and prohibit network access with `offline=True`;
- fetch and replace an entry with `refresh=True`.

Entries are separated by parser-schema version so a future parser can invalidate incompatible cached HTML. Default
locations follow platform conventions: `~/.cache/pecos/qecdb` (or `XDG_CACHE_HOME`) on Linux,
`~/Library/Caches/pecos/qecdb` on macOS, and the local application-data directory on Windows.

The catalog can grow without making an ID-addressed cache entry stale: a newly added code receives a different ID.
An existing record may still be edited upstream, so use `refresh=True` when current contents matter. The
`source_sha256` digest makes it possible to detect whether the response changed while retaining the original cached
HTML for reproducibility. If PECOS later adds catalog search, search-result caches should have their own shorter
freshness policy because the set of matching IDs changes as qecdb grows.

Tests and reproducible workflows can parse a checked-in response directly:

<!--skip: illustrative path; the offline parsing behavior is tested above-->
```python
from pathlib import Path

record = qecdb.parse_html(Path("record.html").read_bytes())
```

PECOS uses only the credential-free public qecdb website. It does not connect directly to qecdb's backing database and
does not provide database registration, update, or deletion operations.

Because the public site does not currently advertise a versioned JSON API, an upstream HTML change may raise
`QecdbParseError`. This is intentional: PECOS fails loudly rather than silently importing misidentified fields.
