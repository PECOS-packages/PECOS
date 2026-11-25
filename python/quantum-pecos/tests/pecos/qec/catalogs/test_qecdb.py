import hashlib
import urllib.error
from pathlib import Path
from types import MappingProxyType

import pytest
from pecos.qec.catalogs import qecdb

FIXTURES = Path(__file__).parent / "fixtures"
FIVE_QUBIT_ID = "670588ce859a107b3f7601fd"
STEANE_ID = "6705224a4315521360862353"


def fixture_bytes(name):
    return (FIXTURES / name).read_bytes()


def test_parse_non_css_record_and_preserve_provenance():
    record = qecdb.parse_html(
        fixture_bytes("five_qubit.html"),
        source_url=f"https://qecdb.org/codes/{FIVE_QUBIT_ID}",
        expected_id=FIVE_QUBIT_ID.upper(),
        retrieved_at="2026-08-14T12:00:00+00:00",
    )

    assert record.id == FIVE_QUBIT_ID
    assert record.reported_parameters == (5, 1, 3)
    assert record.name == "[[5, 1, 3]]"
    assert record.is_css is False
    assert record.is_gf4 is True
    assert record.code_type == "gf4"
    assert record.stabilizers == ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    assert record.destabilizers == ("ZIZII", "ZXIXI", "ZXIII", "ZXIIX")
    assert record.reported_logical_operators == ("XYXII", "ZXZII")
    assert record.source_sha256 == hashlib.sha256(fixture_bytes("five_qubit.html")).hexdigest()
    assert record.raw_fields["future_field"] == "preserved verbatim"
    assert isinstance(record.raw_fields, MappingProxyType)
    with pytest.raises(TypeError):
        record.raw_fields["future_field"] = "changed"


@pytest.mark.parametrize("fixture", ["five_qubit.html", "steane.html"])
def test_live_shape_fixtures_validate(fixture):
    record = qecdb.parse_html(fixture_bytes(fixture))
    report = record.validate()
    assert report.is_valid, report.failures
    assert not report.warnings


def test_css_metadata_and_native_conversion():
    record = qecdb.parse_html(fixture_bytes("steane.html"))

    assert record.reported_parameters == (7, 1, 3)
    assert record.reported_dx == 3
    assert record.reported_dz == 3
    assert record.css_name == "[[7, 1, (3, 3)]]"
    assert record.is_css is True
    assert record.is_self_dual is True
    assert record.automorphisms == 168

    code = record.to_stabilizer_code()
    assert code.num_qubits() == 7
    assert code.num_logical_qubits() == 1
    assert code.code_parameters() == "[[7, 1]]"
    assert len(code.logical_operators()) == 2


def test_validation_does_not_assume_logical_order():
    original = qecdb.parse_html(fixture_bytes("five_qubit.html"))
    swapped = qecdb.QecdbRecord(
        id=original.id,
        n=original.n,
        k=original.k,
        stabilizers=original.stabilizers,
        logical_operators=tuple(reversed(original.logical_operators)),
        destabilizers=original.destabilizers,
    )
    assert swapped.validate().is_valid


def test_validation_reports_inconsistent_claims():
    record = qecdb.QecdbRecord(
        id=FIVE_QUBIT_ID,
        n=5,
        k=2,
        stabilizers=("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"),
        reported_distance=99,
        is_css=True,
        logical_operators=("XYXII", "ZXZII"),
        destabilizers=("ZIZII", "ZXIXI", "ZXIII", "ZXIIX"),
    )

    report = record.validate()
    failed_names = {check.name for check in report.failures}
    assert "stabilizers.rank" in failed_names
    assert "distance.range" in failed_names
    assert "css.pure_rows" in failed_names
    assert "logicals.count" in failed_names
    with pytest.raises(qecdb.QecdbValidationError) as error:
        report.raise_if_invalid()
    assert error.value.report is report


@pytest.mark.parametrize(
    ("html", "message"),
    [
        ("<table><tr><td>n</td><td>5</td></tr></table>", "missing required field"),
        (
            fixture_bytes("five_qubit.html").replace(b"XZZXI", b"XZZQZ", 1),
            "invalid or incorrectly sized Pauli string",
        ),
        (
            fixture_bytes("five_qubit.html").replace(b"<td>5</td>", b"<td>five</td>", 1),
            "not an integer",
        ),
    ],
)
def test_parse_rejects_malformed_pages(html, message):
    with pytest.raises(qecdb.QecdbParseError, match=message):
        qecdb.parse_html(html)


def test_parse_rejects_response_for_different_id():
    with pytest.raises(qecdb.QecdbParseError, match="does not match requested ID"):
        qecdb.parse_html(fixture_bytes("five_qubit.html"), expected_id=STEANE_ID)


@pytest.mark.parametrize("code_id", ["", "123", "g" * 24, "../codes/anything"])
def test_fetch_rejects_invalid_id(code_id, tmp_path):
    with pytest.raises(qecdb.QecdbInvalidIdError):
        qecdb.fetch(code_id, cache_dir=tmp_path, offline=True)


class FakeResponse:
    def __init__(self, data, url):
        self.data = data
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self):
        return self.url

    def read(self, size):
        return self.data[:size]


def test_fetch_populates_and_reuses_cache(monkeypatch, tmp_path):
    calls = []

    def open_record(request, *, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(fixture_bytes("five_qubit.html"), request.full_url)

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", open_record)
    fetched = qecdb.fetch(FIVE_QUBIT_ID, cache_dir=tmp_path, timeout=3.0)
    assert fetched.id == FIVE_QUBIT_ID
    assert fetched.retrieved_at is not None
    assert calls == [(f"https://qecdb.org/codes/{FIVE_QUBIT_ID}", 3.0)]

    def unexpected_network(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", unexpected_network)
    cached = qecdb.fetch(FIVE_QUBIT_ID, cache_dir=tmp_path, offline=True)
    assert cached == fetched
    assert cached.source_sha256 == fetched.source_sha256


def test_fetch_refreshes_existing_cache(monkeypatch, tmp_path):
    def populate(request, *, timeout):
        assert timeout == qecdb.DEFAULT_TIMEOUT_SECONDS
        return FakeResponse(fixture_bytes("five_qubit.html"), request.full_url)

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", populate)
    qecdb.fetch(FIVE_QUBIT_ID, cache_dir=tmp_path)

    refresh_calls = []

    def open_record(request, *, timeout):
        assert timeout == qecdb.DEFAULT_TIMEOUT_SECONDS
        refresh_calls.append(request.full_url)
        return FakeResponse(fixture_bytes("five_qubit.html"), request.full_url)

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", open_record)
    assert qecdb.fetch(FIVE_QUBIT_ID, cache_dir=tmp_path, refresh=True).id == FIVE_QUBIT_ID
    assert refresh_calls == [f"https://qecdb.org/codes/{FIVE_QUBIT_ID}"]


def test_fetch_offline_cache_miss(tmp_path):
    with pytest.raises(qecdb.QecdbCacheMissError, match="not available in cache"):
        qecdb.fetch(FIVE_QUBIT_ID, cache_dir=tmp_path, offline=True)


def test_fetch_wraps_network_errors(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        message = "unavailable"
        raise urllib.error.URLError(message)

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", fail)
    with pytest.raises(qecdb.QecdbFetchError, match="cannot retrieve"):
        qecdb.fetch(FIVE_QUBIT_ID, cache=False, cache_dir=tmp_path)


def test_fetch_rejects_unexpected_redirect(monkeypatch, tmp_path):
    def redirect(_request, *, timeout: float):
        assert timeout == qecdb.DEFAULT_TIMEOUT_SECONDS
        return FakeResponse(fixture_bytes("five_qubit.html"), "https://example.com/record")

    monkeypatch.setattr(qecdb.urllib.request, "urlopen", redirect)
    with pytest.raises(qecdb.QecdbFetchError, match="unexpected location"):
        qecdb.fetch(FIVE_QUBIT_ID, cache=False, cache_dir=tmp_path)
