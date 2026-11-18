# generated-by: gemini-ultra
"""Real TCP_NODELAY loopback/LAN micro-probe server/client and UDP peer discovery beacon."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import statistics
import struct
import threading
import time
from typing import Any

MAGIC = b"LMSH"
HDR_STRUCT = struct.Struct("!4sBII")

MSG_PING = 1
MSG_PONG = 2
MSG_SLICE_PUSH = 3
MSG_SLICE_ACK = 4


@dataclass(frozen=True)
class ProbeMetrics:
    """Measured socket latency and tensor-slice transfer speed to a peer."""

    target: str
    reachable: bool
    rtt_median_ms: float
    rtt_p95_ms: float
    bandwidth_mbps: float
    bandwidth_mbs: float
    error: str | None = None


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Peer closed socket mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ProbeServer:
    """Lightweight threaded TCP micro-probe daemon (binds to `127.0.0.1` by default)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.bound_port: int = 0

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._host, self._port))
        srv.listen(16)
        srv.settimeout(0.1)
        self.bound_port = int(srv.getsockname()[1])
        self._sock = srv
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(2.0)
            while not self._stop_event.is_set():
                try:
                    raw_hdr = _recv_exact(conn, HDR_STRUCT.size)
                except ConnectionError:
                    break
                magic, msg_type, seq, payload_len = HDR_STRUCT.unpack(raw_hdr)
                if magic != MAGIC:
                    break
                if payload_len > 0:
                    _recv_exact(conn, payload_len)
                if msg_type == MSG_PING:
                    conn.sendall(HDR_STRUCT.pack(MAGIC, MSG_PONG, seq, 0))
                elif msg_type == MSG_SLICE_PUSH:
                    conn.sendall(HDR_STRUCT.pack(MAGIC, MSG_SLICE_ACK, seq, 0))
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            worker = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
            worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> ProbeServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def probe_tcp_peer(
    host: str,
    port: int,
    iters: int = 20,
    slice_bytes: int = 262144,
    timeout_sec: float = 2.0,
) -> ProbeMetrics:
    """Measure real TCP ping-pong RTT (`ms`) and tensor-frame throughput (`Mbps` / `MB/s`)."""
    target = f"{host}:{port}"
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if slice_bytes < 1024:
        raise ValueError("slice_bytes must be >= 1024")

    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(timeout_sec)

            # Warmup 1 ping
            sock.sendall(HDR_STRUCT.pack(MAGIC, MSG_PING, 0, 0))
            _recv_exact(sock, HDR_STRUCT.size)

            rtt_samples_ms: list[float] = []
            for seq in range(1, iters + 1):
                t0 = time.perf_counter()
                sock.sendall(HDR_STRUCT.pack(MAGIC, MSG_PING, seq, 0))
                resp = _recv_exact(sock, HDR_STRUCT.size)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                r_magic, r_type, r_seq, _ = HDR_STRUCT.unpack(resp)
                if r_magic != MAGIC or r_type != MSG_PONG or r_seq != seq:
                    raise ConnectionError("Unexpected PONG header from peer")
                rtt_samples_ms.append(elapsed_ms)

            # Measure sustained tensor activation slice streaming throughput
            tensor_frame = HDR_STRUCT.pack(MAGIC, MSG_SLICE_PUSH, 999, slice_bytes) + (
                b"\x5a" * slice_bytes
            )
            stream_iters = max(4, min(iters, 16))
            t_start = time.perf_counter()
            for seq in range(stream_iters):
                sock.sendall(tensor_frame)
                ack = _recv_exact(sock, HDR_STRUCT.size)
                a_magic, a_type, _, _ = HDR_STRUCT.unpack(ack)
                if a_magic != MAGIC or a_type != MSG_SLICE_ACK:
                    raise ConnectionError("Unexpected SLICE_ACK header from peer")
            total_sec = max(1e-6, time.perf_counter() - t_start)

            total_payload_bytes = float(slice_bytes * stream_iters)
            bw_mbs = (total_payload_bytes / (1024.0 * 1024.0)) / total_sec
            bw_mbps = (total_payload_bytes * 8.0 / 1_000_000.0) / total_sec

            sorted_rtts = sorted(rtt_samples_ms)
            p95_idx = min(len(sorted_rtts) - 1, int(0.95 * (len(sorted_rtts) - 1)))
            return ProbeMetrics(
                target=target,
                reachable=True,
                rtt_median_ms=round(statistics.median(sorted_rtts), 4),
                rtt_p95_ms=round(sorted_rtts[p95_idx], 4),
                bandwidth_mbps=round(bw_mbps, 2),
                bandwidth_mbs=round(bw_mbs, 2),
                error=None,
            )
    except Exception as exc:
        return ProbeMetrics(
            target=target,
            reachable=False,
            rtt_median_ms=0.0,
            rtt_p95_ms=0.0,
            bandwidth_mbps=0.0,
            bandwidth_mbs=0.0,
            error=str(exc),
        )


class BeaconListener:
    """UDP JSON peer announcement listener (bound to `127.0.0.1` by default for safe testing)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self.bound_port: int = 0

    def __enter__(self) -> BeaconListener:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        self.bound_port = int(sock.getsockname()[1])
        self._sock = sock
        return self

    def collect(self, timeout_sec: float = 0.5, max_peers: int = 8) -> list[dict[str, Any]]:
        assert self._sock is not None
        deadline = time.monotonic() + timeout_sec
        seen: dict[str, dict[str, Any]] = {}
        while time.monotonic() < deadline and len(seen) < max_peers:
            rem = max(0.01, deadline - time.monotonic())
            self._sock.settimeout(rem)
            try:
                raw, _ = self._sock.recvfrom(4096)
            except TimeoutError:
                break
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict) and msg.get("_magic") == "LMSH_BEACON":
                peer_data = msg.get("peer", {})
                if isinstance(peer_data, dict) and "name" in peer_data:
                    seen[str(peer_data["name"])] = peer_data
        return list(seen.values())

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def send_udp_beacon(
    peer_info: dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 50059,
) -> None:
    """Send a single UDP peer discovery announcement."""
    envelope = {
        "_generated_by": "gemini-ultra",
        "_magic": "LMSH_BEACON",
        "peer": peer_info,
    }
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(raw, (host, port))
