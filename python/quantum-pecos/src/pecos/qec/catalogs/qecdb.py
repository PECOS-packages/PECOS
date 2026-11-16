# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only access to abstract stabilizer-code records from qecdb.org.

The public qecdb website does not currently expose a documented, versioned JSON
API. This module therefore keeps HTML transport and parsing behind a small
boundary and caches the original response for deterministic offline reuse.

Catalog records deliberately remain separate from PECOS mathematical code
objects. Reported properties such as distance are provenance-bearing metadata;
they are not silently treated as properties independently computed by PECOS.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from pecos_rslib import PauliStabilizerGroup, PauliString, StabilizerCode

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_QECDB_BASE_URL = "https://qecdb.org/codes"
_PARSER_SCHEMA_VERSION = 1
# Default timeout used by fetch(), in seconds.
DEFAULT_TIMEOUT_SECONDS = 10.0

_MAX_RESPONSE_BYTES = 2_000_000

_ID_PATTERN = re.compile(r"[0-9a-fA-F]{24}\Z")
_PAULI_PATTERN = re.compile(r"[IXYZ]+\Z")


class QecdbError(Exception):
    """Base class for qecdb integration errors."""


class QecdbInvalidIdError(QecdbError, ValueError):
    """Raised when a qecdb identifier is not a 24-character hexadecimal ID."""


class QecdbFetchError(QecdbError):
    """Raised when a qecdb record cannot be retrieved."""


class QecdbCacheMissError(QecdbFetchError):
    """Raised when offline loading is requested but no cached record exists."""


class QecdbParseError(QecdbError, ValueError):
    """Raised when a qecdb response does not match the supported record schema."""


class QecdbValidationError(QecdbError, ValueError):
    """Raised when a parsed qecdb record is algebraically inconsistent.

    Attributes:
        report: Complete structured validation report, including successful
            checks, warnings, and failures.
    """

    def __init__(self, report: QecdbValidationReport) -> None:
        """Create an error retaining the complete validation report."""
        self.report = report
        details = "\n".join(f"  - {check.name}: {check.detail}" for check in report.failures)
        super().__init__(f"qecdb record failed {len(report.failures)} validation check(s):\n{details}")


@dataclass(frozen=True)
class QecdbValidationCheck:
    """One catalog-record validation result.

    Attributes:
        name: Stable dotted identifier for the check, such as
            ``"stabilizers.rank"``.
        passed: Whether the reported property passed this check.
        detail: Human-readable evidence or failure explanation.
        severity: ``"error"`` for a required invariant or ``"warning"`` for
            a suspicious but importable catalog claim.
    """

    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass(frozen=True)
class QecdbValidationReport:
    """Structured results from validating a :class:`QecdbRecord`.

    A report is truthy exactly when it contains no failed error-severity
    checks. Warnings do not make a report invalid.

    Attributes:
        checks: All checks performed, including successful checks.
    """

    checks: tuple[QecdbValidationCheck, ...]

    @property
    def failures(self) -> tuple[QecdbValidationCheck, ...]:
        """Return failed error-severity checks."""
        return tuple(check for check in self.checks if not check.passed and check.severity == "error")

    @property
    def warnings(self) -> tuple[QecdbValidationCheck, ...]:
        """Return failed warning-severity checks."""
        return tuple(check for check in self.checks if not check.passed and check.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Whether every error-severity check passed."""
        return not self.failures

    def raise_if_invalid(self) -> QecdbValidationReport:
        """Raise :class:`QecdbValidationError` if any required check failed."""
        if not self.is_valid:
            raise QecdbValidationError(self)
        return self

    def __bool__(self) -> bool:
        """Return whether the report contains no error-severity failures."""
        return self.is_valid


@dataclass(frozen=True)
class QecdbRecord:
    """An immutable, provenance-bearing record parsed from qecdb.org.

    This object preserves what the catalog reported. It is intentionally
    separate from PECOS's mathematical :class:`pecos_rslib.StabilizerCode`,
    which may independently compute a different valid logical basis.

    Attributes:
        id: Normalized 24-character hexadecimal qecdb record ID.
        n: Reported number of physical qubits.
        k: Reported number of encoded logical qubits.
        stabilizers: Stabilizer generators from qecdb field ``H``.
        reported_distance: Reported code distance ``d``, if present. Import
            does not prove or recompute this value.
        reported_dx: Reported X-distance ``dx``, if present.
        reported_dz: Reported Z-distance ``dz``, if present.
        name: Catalog display name, if present.
        css_name: Catalog CSS parameter/name string, if present.
        family: Catalog family label, if present.
        created: Creation value exactly as exposed by the public page.
        is_css: Reported CSS flag, or ``None`` when absent.
        is_gf4: Reported GF(4) flag, or ``None`` when absent.
        is_self_dual: Reported self-dual flag, or ``None`` when absent.
        code_type: Catalog type field ``tp``, if present.
        automorphisms: Reported automorphism-group order, if present.
        logical_operators: Logical Pauli representatives from field ``L``.
            Their row order is preserved but not assigned X/Z meaning.
        destabilizers: Destabilizer or pure-error partners from field ``T``.
            These are not transversal-T gates.
        source_url: Public page URL associated with the record.
        retrieved_at: ISO-8601 UTC retrieval/cache timestamp, if known.
        source_sha256: SHA-256 digest of the exact HTML bytes parsed into this
            record. This can be used to pin or compare catalog snapshots.
        raw_fields: Read-only mapping containing every parsed table field,
            including fields not recognized by this parser version.
    """

    id: str
    n: int
    k: int
    stabilizers: tuple[str, ...]
    reported_distance: int | None = None
    reported_dx: int | None = None
    reported_dz: int | None = None
    name: str | None = None
    css_name: str | None = None
    family: str | None = None
    created: str | None = None
    is_css: bool | None = None
    is_gf4: bool | None = None
    is_self_dual: bool | None = None
    code_type: str | None = None
    automorphisms: int | None = None
    logical_operators: tuple[str, ...] = ()
    destabilizers: tuple[str, ...] = ()
    source_url: str = ""
    retrieved_at: str | None = field(default=None, compare=False)
    source_sha256: str = field(default="", compare=False)
    raw_fields: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze a defensive copy of all raw catalog fields."""
        object.__setattr__(self, "raw_fields", MappingProxyType(dict(self.raw_fields)))

    @property
    def reported_parameters(self) -> tuple[int, int, int | None]:
        """Return the ``(n, k, d)`` values reported by qecdb."""
        return self.n, self.k, self.reported_distance

    @property
    def reported_logical_operators(self) -> tuple[str, ...]:
        """Return qecdb's logical representatives without recomputing them."""
        return self.logical_operators

    def validate(self) -> QecdbValidationReport:
        """Validate the reported stabilizer tableau and inexpensive metadata claims.

        Distance is checked only for basic consistency. Computing code distance
        is intentionally not part of catalog import.

        Returns:
            A report containing every performed check. Call
            :meth:`QecdbValidationReport.raise_if_invalid` to require success.
        """
        checks: list[QecdbValidationCheck] = []

        def add(name: str, passed: bool, detail: str, *, severity: str = "error") -> None:
            checks.append(QecdbValidationCheck(name, passed, detail, severity))

        add("parameters.n", self.n > 0, f"n={self.n}; expected a positive integer")
        add("parameters.k", 0 <= self.k <= self.n, f"k={self.k}; expected 0 <= k <= n={self.n}")

        h_well_formed = _operators_well_formed(self.stabilizers, self.n)
        add(
            "format.stabilizers",
            h_well_formed,
            f"{len(self.stabilizers)} stabilizer row(s), each expected to be a length-{self.n} Pauli string",
        )
        if not h_well_formed or self.n <= 0 or not (0 <= self.k <= self.n):
            return QecdbValidationReport(tuple(checks))

        native_group: PauliStabilizerGroup | None = None
        try:
            native_group = _native_group(self.stabilizers)
        except ValueError as error:
            add("stabilizers.commute", passed=False, detail=str(error))
        else:
            add("stabilizers.commute", passed=True, detail="all stabilizer generators commute")
            rank = native_group.rank()
            add(
                "stabilizers.rank",
                rank == self.n - self.k,
                f"rank(H)={rank}; expected n-k={self.n - self.k}",
            )
            add(
                "stabilizers.independent",
                rank == len(self.stabilizers),
                f"rank(H)={rank}; H contains {len(self.stabilizers)} row(s)",
            )

        if self.reported_distance is not None:
            add(
                "distance.range",
                1 <= self.reported_distance <= self.n,
                f"reported d={self.reported_distance}; expected 1 <= d <= n={self.n}",
            )
        if self.reported_dx is not None:
            add(
                "distance.dx_range",
                1 <= self.reported_dx <= self.n,
                f"reported dx={self.reported_dx}; expected 1 <= dx <= n={self.n}",
            )
        if self.reported_dz is not None:
            add(
                "distance.dz_range",
                1 <= self.reported_dz <= self.n,
                f"reported dz={self.reported_dz}; expected 1 <= dz <= n={self.n}",
            )

        rows_are_css = all(_is_pure_pauli_row(row) for row in self.stabilizers)
        if self.is_css is True:
            add("css.pure_rows", rows_are_css, "css=True requires every stabilizer to be pure X or pure Z")
        elif self.is_css is False and rows_are_css:
            add(
                "css.reported_flag",
                passed=False,
                detail="css=False, but every stabilizer row is pure X or pure Z",
                severity="warning",
            )

        logicals_well_formed = _operators_well_formed(self.logical_operators, self.n)
        add(
            "format.logicals",
            logicals_well_formed,
            f"{len(self.logical_operators)} logical row(s), each expected to be a length-{self.n} Pauli string",
        )
        if logicals_well_formed and self.logical_operators:
            _validate_logicals(self, native_group, checks)

        destabilizers_well_formed = _operators_well_formed(self.destabilizers, self.n)
        add(
            "format.destabilizers",
            destabilizers_well_formed,
            f"{len(self.destabilizers)} destabilizer row(s), each expected to be a length-{self.n} Pauli string",
        )
        if destabilizers_well_formed and self.destabilizers:
            _validate_destabilizers(self, checks)

        return QecdbValidationReport(tuple(checks))

    def to_stabilizer_code(self, *, validate: bool = True) -> StabilizerCode:
        """Convert the record's stabilizers into PECOS's native code object.

        The returned object computes its own logical basis. qecdb's reported
        logical representatives and distance remain on this record.

        Args:
            validate: Validate the complete reported record before conversion.

        Returns:
            A Rust-backed PECOS ``StabilizerCode``.

        Raises:
            QecdbValidationError: If ``validate`` is true and a required
                catalog invariant fails.
            ValueError: If validation is disabled and the stabilizers cannot
                construct a native commuting stabilizer group.
        """
        if validate:
            self.validate().raise_if_invalid()
        return StabilizerCode(_native_group(self.stabilizers), self.n)


class _QecdbTableParser(HTMLParser):
    """Extract two-column table rows while preserving line breaks in cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, str]] = []
        self._row: list[str] = []
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "td":
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell_parts is not None:
            self._row.append("".join(self._cell_parts).strip())
            self._cell_parts = None
        elif tag == "tr":
            if len(self._row) == 2:
                self.rows.append((self._row[0], self._row[1]))
            self._row = []


def parse_html(
    html: str | bytes,
    *,
    source_url: str = "",
    expected_id: str | None = None,
    retrieved_at: str | None = None,
) -> QecdbRecord:
    """Parse one public qecdb code-detail page into an immutable record.

    Args:
        html: UTF-8 page contents.
        source_url: Provenance URL associated with the page.
        expected_id: Optional ID that must match the page's ``_id`` field.
        retrieved_at: Optional ISO-8601 retrieval timestamp.

    Raises:
        QecdbInvalidIdError: If the page or ``expected_id`` contains an invalid
            qecdb identifier.
        QecdbParseError: If required fields are absent, duplicated, or malformed.

    Returns:
        The parsed catalog record. Full algebraic validation remains explicit;
        call :meth:`QecdbRecord.validate` or
        :meth:`QecdbRecord.to_stabilizer_code` next.
    """
    if isinstance(html, bytes):
        source_bytes = html
        try:
            html = html.decode("utf-8")
        except UnicodeDecodeError as error:
            msg = "qecdb response is not valid UTF-8"
            raise QecdbParseError(msg) from error
    else:
        source_bytes = html.encode("utf-8")

    parser = _QecdbTableParser()
    parser.feed(html)
    fields: dict[str, str] = {}
    for key, value in parser.rows:
        if key in fields:
            msg = f"qecdb response contains duplicate field {key!r}"
            raise QecdbParseError(msg)
        fields[key] = value

    for required in ("_id", "n", "k", "H"):
        if required not in fields:
            msg = f"qecdb response is missing required field {required!r}"
            raise QecdbParseError(msg)

    record_id = _normalize_id(fields["_id"])
    if expected_id is not None and record_id != _normalize_id(expected_id):
        msg = f"qecdb response ID {record_id!r} does not match requested ID {_normalize_id(expected_id)!r}"
        raise QecdbParseError(msg)

    n = _parse_required_int(fields, "n")
    k = _parse_required_int(fields, "k")
    stabilizers = _parse_operators(fields["H"])
    logicals = _parse_operators(fields.get("L", ""))
    destabilizers = _parse_operators(fields.get("T", ""))

    for label, operators in (("H", stabilizers), ("L", logicals), ("T", destabilizers)):
        if not _operators_well_formed(operators, n):
            msg = f"qecdb field {label!r} contains an invalid or incorrectly sized Pauli string for n={n}"
            raise QecdbParseError(msg)

    return QecdbRecord(
        id=record_id,
        n=n,
        k=k,
        stabilizers=stabilizers,
        reported_distance=_parse_optional_int(fields, "d"),
        reported_dx=_parse_optional_int(fields, "dx"),
        reported_dz=_parse_optional_int(fields, "dz"),
        name=_optional_text(fields, "name"),
        css_name=_optional_text(fields, "cssname"),
        family=_optional_text(fields, "family"),
        created=_optional_text(fields, "created"),
        is_css=_parse_optional_bool(fields, "css"),
        is_gf4=_parse_optional_bool(fields, "gf4"),
        is_self_dual=_parse_optional_bool(fields, "selfdual"),
        code_type=_optional_text(fields, "tp"),
        automorphisms=_parse_optional_int(fields, "automorphisms"),
        logical_operators=logicals,
        destabilizers=destabilizers,
        source_url=source_url,
        retrieved_at=retrieved_at,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        raw_fields=fields,
    )


def fetch(
    code_id: str,
    *,
    cache: bool = True,
    refresh: bool = False,
    offline: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> QecdbRecord:
    """Fetch one qecdb record by ID, using a versioned disk cache by default.

    Args:
        code_id: The 24-character hexadecimal qecdb record ID.
        cache: Read and populate the disk cache.
        refresh: Ignore an existing cache entry and fetch a fresh page.
        offline: Require an existing cache entry and perform no network access.
        cache_dir: Override the qecdb cache directory.
        timeout: Network timeout in seconds.

    Raises:
        QecdbInvalidIdError: If ``code_id`` is malformed.
        QecdbCacheMissError: If offline mode has no cached response.
        QecdbFetchError: If retrieval fails.
        QecdbParseError: If the response schema is unsupported or invalid.
        ValueError: If ``timeout`` is not positive or offline mode is requested
            with caching disabled.

    Returns:
        A parsed catalog record. Fetching does not perform full algebraic
        validation; :meth:`QecdbRecord.to_stabilizer_code` validates by default.

    Notes:
        Cache entries do not expire automatically. With default options an
        existing entry is reused until ``refresh=True`` is requested.
        ``cache=False`` bypasses both cache reads and writes.

        Because records are addressed by qecdb ID, newly added catalog records
        do not stale an existing entry. Use ``refresh=True`` when current
        contents of a potentially edited record are required, and compare
        :attr:`QecdbRecord.source_sha256` to detect a changed response.
    """
    normalized_id = _normalize_id(code_id)
    if timeout <= 0:
        msg = "timeout must be positive"
        raise ValueError(msg)
    if offline and not cache:
        msg = "offline=True requires cache=True"
        raise ValueError(msg)

    url = f"{_QECDB_BASE_URL}/{normalized_id}"
    path = _cache_path(normalized_id, cache_dir)
    if cache and path.is_file() and not refresh:
        retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        return parse_html(path.read_bytes(), source_url=url, expected_id=normalized_id, retrieved_at=retrieved_at)

    if offline:
        msg = f"qecdb record {normalized_id!r} is not available in cache {path.parent}"
        raise QecdbCacheMissError(msg)

    html = _download_html(url, timeout)
    retrieved_at = datetime.now(UTC).isoformat()
    record = parse_html(html, source_url=url, expected_id=normalized_id, retrieved_at=retrieved_at)
    if cache:
        _write_cache(path, html)
    return record


def _download_html(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(  # noqa: S310 - URL is fixed HTTPS + validated ID
        url,
        headers={"User-Agent": "PECOS qecdb importer/1"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL is fixed HTTPS + validated ID
            request,
            timeout=timeout,
        ) as response:
            _validate_response_url(response.geturl())
            data = response.read(_MAX_RESPONSE_BYTES + 1)
    except QecdbFetchError:
        raise
    except (OSError, urllib.error.URLError) as error:
        msg = f"cannot retrieve {url}: {error}"
        raise QecdbFetchError(msg) from error
    if len(data) > _MAX_RESPONSE_BYTES:
        msg = f"qecdb response exceeds the {_MAX_RESPONSE_BYTES}-byte safety limit"
        raise QecdbFetchError(msg)
    return data


def _validate_response_url(response_url: str) -> None:
    final_url = urllib.parse.urlparse(response_url)
    if final_url.scheme != "https" or final_url.hostname != "qecdb.org":
        msg = f"qecdb redirected to an unexpected location: {response_url}"
        raise QecdbFetchError(msg)


def _write_cache(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            handle.write(data)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    except OSError as error:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        msg = f"cannot write qecdb cache entry {path}: {error}"
        raise QecdbFetchError(msg) from error


def _cache_path(code_id: str, cache_dir: str | os.PathLike[str] | None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    return root / f"schema-v{_PARSER_SCHEMA_VERSION}" / f"{code_id}.html"


def _default_cache_dir() -> Path:
    override = os.environ.get("PECOS_CACHE_DIR")
    if override:
        return Path(override) / "qecdb"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "PECOS" / "Cache" / "qecdb"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pecos" / "qecdb"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pecos" / "qecdb"


def _normalize_id(code_id: str) -> str:
    if not isinstance(code_id, str) or _ID_PATTERN.fullmatch(code_id) is None:
        msg = "qecdb ID must be exactly 24 hexadecimal characters"
        raise QecdbInvalidIdError(msg)
    return code_id.lower()


def _parse_required_int(fields: Mapping[str, str], key: str) -> int:
    try:
        return int(fields[key].strip())
    except ValueError as error:
        msg = f"qecdb field {key!r} is not an integer: {fields[key]!r}"
        raise QecdbParseError(msg) from error


def _parse_optional_int(fields: Mapping[str, str], key: str) -> int | None:
    value = fields.get(key, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        msg = f"qecdb field {key!r} is not an integer: {value!r}"
        raise QecdbParseError(msg) from error


def _parse_optional_bool(fields: Mapping[str, str], key: str) -> bool | None:
    value = fields.get(key, "").strip().lower()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    msg = f"qecdb field {key!r} is not a boolean: {fields[key]!r}"
    raise QecdbParseError(msg)


def _optional_text(fields: Mapping[str, str], key: str) -> str | None:
    value = " ".join(fields.get(key, "").split())
    return value or None


def _parse_operators(value: str) -> tuple[str, ...]:
    return tuple(token.upper() for token in value.split())


def _operators_well_formed(operators: Sequence[str], n: int) -> bool:
    return all(len(operator) == n and _PAULI_PATTERN.fullmatch(operator) is not None for operator in operators)


def _native_group(stabilizers: Sequence[str]) -> PauliStabilizerGroup:
    if not stabilizers:
        return PauliStabilizerGroup([])
    return PauliStabilizerGroup.from_str("\n".join(stabilizers))


def _native_paulis(operators: Iterable[str]) -> tuple[PauliString, ...]:
    return tuple(PauliString.from_dense_str(operator) for operator in operators)


def _is_pure_pauli_row(operator: str) -> bool:
    chars = set(operator)
    return chars <= {"I", "X"} or chars <= {"I", "Z"}


def _symplectic_row(operator: str) -> tuple[int, int]:
    z_bits = 0
    x_bits = 0
    for index, pauli in enumerate(operator):
        if pauli in {"Z", "Y"}:
            z_bits |= 1 << index
        if pauli in {"X", "Y"}:
            x_bits |= 1 << index
    return z_bits, x_bits


def _symplectic_product(left: tuple[int, int], right: tuple[int, int]) -> int:
    left_z, left_x = left
    right_z, right_x = right
    return ((left_z & right_x).bit_count() + (left_x & right_z).bit_count()) & 1


def _binary_rank(rows: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    for source_row in rows:
        row = source_row
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    return len(basis)


def _packed_rows(operators: Iterable[str], n: int) -> tuple[int, ...]:
    packed: list[int] = []
    for operator in operators:
        z_bits, x_bits = _symplectic_row(operator)
        packed.append(z_bits | (x_bits << n))
    return tuple(packed)


def _validate_logicals(
    record: QecdbRecord,
    native_group: PauliStabilizerGroup | None,
    checks: list[QecdbValidationCheck],
) -> None:
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(QecdbValidationCheck(name, passed, detail))

    logicals = _native_paulis(record.logical_operators)
    stabilizers = _native_paulis(record.stabilizers)
    add(
        "logicals.count",
        len(logicals) == 2 * record.k,
        f"L contains {len(logicals)} row(s); expected 2k={2 * record.k}",
    )
    commute = all(logical.commutes_with(stabilizer) for logical in logicals for stabilizer in stabilizers)
    add("logicals.commute_with_stabilizers", commute, "every reported logical must commute with every stabilizer")

    combined_rank = _binary_rank(_packed_rows((*record.stabilizers, *record.logical_operators), record.n))
    expected_rank = (record.n - record.k) + len(record.logical_operators)
    add(
        "logicals.independent",
        combined_rank == expected_rank,
        f"rank([H;L])={combined_rank}; expected {expected_rank}",
    )
    if native_group is not None:
        nontrivial = all(not native_group.contains(logical) for logical in logicals)
        add("logicals.nontrivial", nontrivial, "reported logicals must not belong to the stabilizer group")

    symplectic = tuple(_symplectic_row(operator) for operator in record.logical_operators)
    gram_rows = tuple(
        sum(_symplectic_product(left, right) << column for column, right in enumerate(symplectic))
        for left in symplectic
    )
    gram_rank = _binary_rank(gram_rows)
    add(
        "logicals.symplectic_pairing",
        gram_rank == 2 * record.k,
        f"rank(Gram(L,L))={gram_rank}; expected 2k={2 * record.k}; ordering is not assumed",
    )


def _validate_destabilizers(record: QecdbRecord, checks: list[QecdbValidationCheck]) -> None:
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(QecdbValidationCheck(name, passed, detail))

    stabilizers = tuple(_symplectic_row(operator) for operator in record.stabilizers)
    destabilizers = tuple(_symplectic_row(operator) for operator in record.destabilizers)
    logicals = tuple(_symplectic_row(operator) for operator in record.logical_operators)
    expected_count = record.n - record.k
    add(
        "destabilizers.count",
        len(destabilizers) == expected_count,
        f"T contains {len(destabilizers)} row(s); expected n-k={expected_count}",
    )
    pairing = all(
        _symplectic_product(destabilizer, stabilizer) == (destabilizer_index == stabilizer_index)
        for destabilizer_index, destabilizer in enumerate(destabilizers)
        for stabilizer_index, stabilizer in enumerate(stabilizers)
    )
    add("destabilizers.pairing", pairing, "expected symplectic pairing <T_i,H_j> = delta_ij")
    mutually_commute = all(
        _symplectic_product(destabilizers[left], destabilizers[right]) == 0
        for left in range(len(destabilizers))
        for right in range(left + 1, len(destabilizers))
    )
    add("destabilizers.commute", mutually_commute, "reported destabilizers must mutually commute")
    commute_with_logicals = all(
        _symplectic_product(destabilizer, logical) == 0 for destabilizer in destabilizers for logical in logicals
    )
    add(
        "destabilizers.commute_with_logicals",
        commute_with_logicals,
        "reported destabilizers must commute with reported logicals",
    )
    if record.logical_operators:
        tableau_rank = _binary_rank(
            _packed_rows((*record.stabilizers, *record.destabilizers, *record.logical_operators), record.n),
        )
        add(
            "tableau.complete",
            tableau_rank == 2 * record.n,
            f"rank([H;T;L])={tableau_rank}; expected 2n={2 * record.n}",
        )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "QecdbCacheMissError",
    "QecdbError",
    "QecdbFetchError",
    "QecdbInvalidIdError",
    "QecdbParseError",
    "QecdbRecord",
    "QecdbValidationCheck",
    "QecdbValidationError",
    "QecdbValidationReport",
    "fetch",
    "parse_html",
]
