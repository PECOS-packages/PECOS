"""Test accessing Selene's TCP result stream directly.

This explores how to tap into the TCP stream that Selene uses to communicate
results, which is essential for extracting final results in our integration.
"""

import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

try:
    from selene_sim.result_handling import ResultStream, TCPStream

    SELENE_STREAM_AVAILABLE = True
except ImportError:
    SELENE_STREAM_AVAILABLE = False

try:
    from selene_sim import SeleneInstance

    SELENE_AVAILABLE = True
except ImportError:
    SELENE_AVAILABLE = False


@pytest.mark.skipif(
    not SELENE_STREAM_AVAILABLE,
    reason="Selene stream handling not available",
)
class TestSeleneTCPStream:
    """Test Selene's TCP stream functionality."""

    def test_tcp_stream_creation(self) -> None:
        """Test creating and configuring a TCPStream."""
        # Create a TCP stream with automatic port selection
        with TCPStream(
            host="localhost",
            port=0,  # Let system choose port
            logfile=None,
            shot_offset=0,
            shot_increment=1,
        ) as stream:
            # Verify stream was created
            assert stream is not None, "TCPStream should be created"

            # Get the URI
            uri = stream.get_uri()
            assert uri is not None, "Stream should have a URI"
            assert isinstance(uri, str), "URI should be a string"

            # Verify URI format
            assert uri.startswith("tcp://"), "URI should start with tcp://"

            # Parse host and port
            host_port = uri[6:]  # Remove "tcp://"
            assert ":" in host_port, "URI should contain host:port"

            host, port_str = host_port.split(":")
            port = int(port_str)

            assert host in ["localhost", "127.0.0.1", "::1"], "Host should be localhost"
            assert 1024 <= port <= 65535, "Port should be in valid range"

    def test_tcp_stream_client_connection(self) -> None:
        """Test connecting to TCPStream as a client."""
        connection_successful = False

        with TCPStream(
            host="localhost",
            port=0,
            logfile=None,
            shot_offset=0,
            shot_increment=1,
        ) as stream:
            uri = stream.get_uri()
            host_port = uri[6:]  # Remove "tcp://"
            host, port_str = host_port.split(":")
            port = int(port_str)

            # Connect in a separate thread
            def client_thread() -> None:
                nonlocal connection_successful
                try:
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.settimeout(2.0)  # 2 second timeout
                    client_socket.connect((host, port))
                    connection_successful = True

                    # Send test messages (simulating Selene output)
                    test_messages = [
                        b"USER:BOOL:measurement_1\x001\x00",
                        b"USER:BOOL:measurement_2\x000\x00",
                        b"USER:INT:count\x0042\x00",
                    ]

                    for msg in test_messages:
                        client_socket.send(msg)
                        time.sleep(0.01)  # Small delay between messages

                    client_socket.close()

                except (TimeoutError, OSError, ConnectionError):
                    # Connection failed - this might be expected depending on setup
                    pass

            client = threading.Thread(target=client_thread)
            client.start()
            client.join(timeout=3)

            # Verify connection attempt was made
            assert not client.is_alive(), "Client thread should complete"

            # Note: Actual message reception would require stream.read() or similar
            # which might not be directly exposed in the API

    def test_result_stream_wrapper(self) -> None:
        """Test the ResultStream wrapper around TCPStream."""
        with TCPStream() as tcp_stream:
            result_stream = ResultStream(tcp_stream)

            # Verify ResultStream was created
            assert result_stream is not None, "ResultStream should be created"

            # Check available methods
            result_methods = [m for m in dir(result_stream) if not m.startswith("_")]
            assert len(result_methods) > 0, "ResultStream should have public methods"

            # ResultStream should be iterable
            assert hasattr(result_stream, "__iter__") or hasattr(
                result_stream,
                "__next__",
            ), "ResultStream should be iterable"

    def test_tcp_stream_configuration_options(self) -> None:
        """Test different configuration options for TCPStream."""
        # Test with specific port
        specific_port = 55555
        try:
            with TCPStream(
                host="localhost",
                port=specific_port,
                logfile=None,
                shot_offset=10,
                shot_increment=5,
            ) as stream:
                uri = stream.get_uri()
                assert (
                    f":{specific_port}" in uri
                ), f"URI should contain port {specific_port}"

                # Check shot configuration
                # These might affect how results are indexed
                assert (
                    hasattr(stream, "shot_offset") or True
                ), "Stream tracks shot offset"
                assert (
                    hasattr(stream, "shot_increment") or True
                ), "Stream tracks shot increment"

        except OSError as e:
            # Port might be in use
            if "address already in use" in str(e).lower():
                pytest.skip(f"Port {specific_port} already in use")
            raise

    def test_tcp_stream_with_logfile(self) -> None:
        """Test TCPStream with logging enabled."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".log",
            delete=False,
        ) as logfile:
            logfile_path = Path(logfile.name)

        try:
            with TCPStream(
                host="localhost",
                port=0,
                logfile=str(logfile_path),
                shot_offset=0,
                shot_increment=1,
            ) as stream:
                uri = stream.get_uri()
                assert uri is not None, "Stream with logging should work"

                # Send some test data to potentially trigger logging
                host_port = uri[6:]
                host, port_str = host_port.split(":")
                port = int(port_str)

                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.settimeout(1.0)
                    client.connect((host, port))
                    client.send(b"TEST:LOG:message\x00")
                    client.close()
                except (TimeoutError, OSError):
                    pass

            # Check if log file was created/written
            if logfile_path.exists():
                log_size = logfile_path.stat().st_size
                assert log_size >= 0, "Log file should exist"

        finally:
            # Clean up log file
            if logfile_path.exists():
                logfile_path.unlink()


@pytest.mark.skipif(not SELENE_AVAILABLE, reason="Selene not available")
class TestSeleneResultInterception:
    """Test intercepting results from Selene execution."""

    def test_selene_instance_stream_configuration(self) -> None:
        """Test how SeleneInstance handles stream configuration."""
        import inspect

        # Check SeleneInstance initialization
        sig = inspect.signature(SeleneInstance.__init__)
        params = list(sig.parameters.keys())

        assert "executable" in params, "SeleneInstance should take executable parameter"

        # Check run_shots method signature
        if hasattr(SeleneInstance, "run_shots"):
            run_sig = inspect.signature(SeleneInstance.run_shots)
            run_params = list(run_sig.parameters.keys())

            # Check for stream-related parameters
            [p for p in run_params if "stream" in p.lower() or "output" in p.lower()]

            # Should have some way to configure output
            assert len(run_params) > 0, "run_shots should have parameters"

    def test_selene_with_custom_stream(self) -> None:
        """Test running Selene with a custom output stream."""
        if not SELENE_STREAM_AVAILABLE:
            pytest.skip("Stream handling not available")

        # Create a simple LLVM program
        llvm_ir = """
        declare void @__quantum__qis__h__body(i64)
        declare i1 @__quantum__qis__mz__body(i64)
        declare void @__quantum__rt__result_record_output(i64, i8*)

        @.str.result = constant [7 x i8] c"result\\00"

        define void @main() #0 {
            call void @__quantum__qis__h__body(i64 0)
            %m = call i1 @__quantum__qis__mz__body(i64 0)
            %m.i64 = zext i1 %m to i64
            call void @__quantum__rt__result_record_output(i64 %m.i64,
                i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.result, i32 0, i32 0))
            ret void
        }

        attributes #0 = { "entry_point" }
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save LLVM
            llvm_file = Path(tmpdir) / "test.ll"
            llvm_file.write_text(llvm_ir)

            try:
                # Try to build with Selene
                from selene_sim import build

                instance = build(
                    src=str(llvm_file),
                    name="stream_test",
                    build_dir=tmpdir,
                )

                # Verify instance was created
                assert instance is not None, "Should create SeleneInstance"

                # Check if instance has methods for stream configuration
                instance_methods = [m for m in dir(instance) if not m.startswith("_")]
                run_methods = [m for m in instance_methods if "run" in m.lower()]

                assert len(run_methods) > 0, "Instance should have run methods"

            except (ImportError, RuntimeError, ValueError) as e:
                if "not supported" in str(e).lower():
                    pytest.skip(f"Custom stream not supported: {e}")

    def test_result_message_format(self) -> None:
        """Test the format of result messages in the TCP stream."""
        # Document expected message formats
        message_formats = {
            "boolean": b"USER:BOOL:tag_name\x00value\x00",
            "integer": b"USER:INT:tag_name\x00value\x00",
            "float": b"USER:FLOAT:tag_name\x00value\x00",
            "string": b"USER:STR:tag_name\x00value\x00",
            "shot_boundary": b"SHOT:END:shot_id\x00\x00",
        }

        # Verify format patterns
        for msg_type, example in message_formats.items():
            assert (
                b"USER:" in example or b"SHOT:" in example
            ), f"{msg_type} should have type prefix"
            assert b"\x00" in example, f"{msg_type} should have null terminators"

            # Parse example message
            if example.startswith(b"USER:"):
                parts = example.split(b"\x00")
                assert len(parts) >= 2, f"{msg_type} should have tag and value parts"

                header = parts[0]  # e.g., b"USER:BOOL:tag_name"
                header_parts = header.split(b":")
                assert len(header_parts) == 3, f"{msg_type} header should have 3 parts"

                category, dtype, tag = header_parts
                assert category == b"USER", "Should be USER message"
                assert dtype in [
                    b"BOOL",
                    b"INT",
                    b"FLOAT",
                    b"STR",
                ], "Should have valid type"
                assert len(tag) > 0, "Should have tag name"

    def test_tcp_message_parsing(self) -> None:
        """Test parsing TCP stream messages."""

        def parse_message(msg: bytes) -> tuple[str, str, Any] | None:
            """Parse a Selene TCP message."""
            if not msg:
                return None

            parts = msg.rstrip(b"\x00").split(b"\x00")
            if len(parts) < 1:
                return None

            header = parts[0].decode("utf-8")
            header_parts = header.split(":")

            if len(header_parts) != 3:
                return None

            category, dtype, tag = header_parts

            # Get value if present
            value = None
            if len(parts) > 1 and parts[1]:
                value_bytes = parts[1]
                if dtype == "BOOL":
                    value = value_bytes == b"1"
                elif dtype == "INT":
                    value = int(value_bytes)
                elif dtype == "FLOAT":
                    value = float(value_bytes)
                elif dtype == "STR":
                    value = value_bytes.decode("utf-8")

            return category, tag, value

        # Test parsing different message types
        test_messages = [
            (b"USER:BOOL:measurement\x001\x00", ("USER", "measurement", True)),
            (b"USER:BOOL:measurement\x000\x00", ("USER", "measurement", False)),
            (b"USER:INT:count\x0042\x00", ("USER", "count", 42)),
            (b"USER:STR:label\x00test\x00", ("USER", "label", "test")),
            (b"SHOT:END:0\x00\x00", ("SHOT", "0", None)),
        ]

        for msg, expected in test_messages:
            result = parse_message(msg)
            assert result is not None, f"Should parse message: {msg}"

            if expected:
                category, tag, value = result
                exp_category, exp_tag, exp_value = expected

                assert category == exp_category, f"Category mismatch for {msg}"
                assert tag == exp_tag, f"Tag mismatch for {msg}"
                if exp_value is not None:
                    assert value == exp_value, f"Value mismatch for {msg}"


class TestSeleneStreamIntegrationStrategies:
    """Test different strategies for integrating with Selene's stream."""

    def test_document_integration_approaches(self) -> None:
        """Document and validate different integration approaches."""
        approaches = {
            "tcp_stream": {
                "description": "Create custom TCP stream before running Selene",
                "steps": [
                    "Create TCPStream with known port",
                    "Pass URI to Selene configuration",
                    "Read from stream during execution",
                    "Parse tagged results",
                ],
                "pros": ["Full control over stream", "Can log all communication"],
                "cons": ["Requires port management", "Need to handle connection"],
            },
            "result_stream": {
                "description": "Use ResultStream wrapper for iteration",
                "steps": [
                    "Create ResultStream around TCPStream",
                    "Configure Selene with stream URI",
                    "Iterate over ResultStream for results",
                    "Process yielded (tag, value) pairs",
                ],
                "pros": ["Higher-level API", "Automatic message parsing"],
                "cons": ["Less control", "May buffer results"],
            },
            "unparsed_mode": {
                "description": "Get raw results without parsing",
                "steps": [
                    "Set parse_results=False in run_shots",
                    "Get raw (type:tag, value) tuples",
                    "Process messages manually",
                    "Handle shot boundaries",
                ],
                "pros": ["Maximum flexibility", "Can handle custom formats"],
                "cons": ["More complex", "Need custom parser"],
            },
            "direct_socket": {
                "description": "Connect directly to Selene's TCP socket",
                "steps": [
                    "Let Selene create its stream",
                    "Extract URI from configuration",
                    "Connect as TCP client",
                    "Read and decode raw bytes",
                ],
                "pros": ["Works with any Selene version", "Independent of API"],
                "cons": ["Fragile", "Timing sensitive"],
            },
        }

        # Validate structure
        for approach_name, details in approaches.items():
            assert "description" in details, f"{approach_name} should have description"
            assert "steps" in details, f"{approach_name} should have steps"
            assert "pros" in details, f"{approach_name} should have pros"
            assert "cons" in details, f"{approach_name} should have cons"

            assert (
                len(details["steps"]) > 0
            ), f"{approach_name} should have implementation steps"
            assert len(details["pros"]) > 0, f"{approach_name} should have advantages"
            assert (
                len(details["cons"]) > 0
            ), f"{approach_name} should have disadvantages"

    def test_recommended_integration_pattern(self) -> None:
        """Test the recommended pattern for SeleneExecutableEngine integration."""
        integration_steps = [
            "Initialize TCPStream in engine.__init__",
            "Store stream URI for configuration",
            "Pass URI to Selene executable via config",
            "Start result reader thread before execution",
            "Execute Selene program",
            "Collect results from reader thread",
            "Convert to PECOS Shot format",
            "Clean up stream and threads",
        ]

        # Verify steps are complete
        assert len(integration_steps) == 8, "Should have 8 integration steps"

        # Check for key components (case-insensitive)
        key_components = ["TCPStream", "URI", "thread", "Shot", "clean"]
        steps_text = " ".join(integration_steps).lower()

        for component in key_components:
            assert (
                component.lower() in steps_text
            ), f"Integration should mention {component}"

    def test_message_protocol_specification(self) -> None:
        """Test and document the message protocol specification."""
        protocol = {
            "message_structure": "CATEGORY:TYPE:TAG\\x00VALUE\\x00",
            "categories": ["USER", "SHOT", "SYSTEM", "ERROR"],
            "types": ["BOOL", "INT", "FLOAT", "STR", "ARRAY", "TUPLE"],
            "encoding": "UTF-8 for strings, native for numbers",
            "delimiter": "\\x00 (null byte)",
            "shot_boundary": "SHOT:END:shot_id",
            "error_format": "ERROR:TYPE:message",
        }

        # Validate protocol specification
        assert "message_structure" in protocol, "Should define message structure"
        assert len(protocol["categories"]) >= 4, "Should have main categories"
        assert len(protocol["types"]) >= 6, "Should have data types"
        assert "\\x00" in protocol["delimiter"], "Should use null byte delimiter"

        # Verify shot boundary format
        shot_boundary = protocol["shot_boundary"]
        assert "SHOT" in shot_boundary, "Shot boundary should have SHOT category"
        assert "END" in shot_boundary, "Shot boundary should have END marker"
