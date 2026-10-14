"""Test accessing Selene's TCP result stream directly.

This explores how to tap into the TCP stream that Selene uses to communicate
results, which is essential for extracting final results in our integration.
"""

import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from selene_sim.result_handling import TCPStream


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
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(2.0)  # 2 second timeout
                client_socket.connect((host, port))

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

            with ThreadPoolExecutor(max_workers=1) as executor:
                client = executor.submit(client_thread)
                client.result(timeout=3)

            # Note: Actual message reception would require stream.read() or similar
            # which might not be directly exposed in the API

    def test_tcp_stream_configuration_options(self) -> None:
        """Test different configuration options for TCPStream."""
        # Port 0 lets the OS assign a free port at bind time, so parallel workers cannot race for it.
        with TCPStream(
            host="127.0.0.1",
            port=0,
            logfile=None,
            shot_offset=10,
            shot_increment=5,
        ) as stream:
            assert stream.port > 0, "Stream should report the port the OS assigned"
            assert stream.get_uri() == f"tcp://127.0.0.1:{stream.port}"
            assert stream.current_shot == 10, "Stream should start at the shot offset"
            assert stream.shot_increment == 5

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

                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(1.0)
                client.connect((host, port))
                client.send(b"TEST:LOG:message\x00")
                client.close()

        finally:
            # Clean up log file
            if logfile_path.exists():
                logfile_path.unlink()
