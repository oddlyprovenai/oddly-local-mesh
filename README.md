<!-- generated-by: gemini-ultra -->
# local-mesh (`oddly-local-mesh`)

> **"Three old laptops, one big model."**

`oddly-local-mesh` (`local-mesh` CLI) is an anti-hype LAN planner, real TCP socket micro-prober, and roofline speed simulator for **upstream `llama.cpp` RPC** (`rpc-server` + `llama-server --rpc`).

Before you spend hours copying a 20 GB–43 GB `.gguf` file across spare laptops, `local-mesh` calculates exact per-layer GGUF + KV-cache memory footprints (reserving `2.5 GiB` OS headroom per node), probes real TCP RTT and activation-slice throughput, **automatically excludes slow Wi-Fi laptops when adding them hurts `tok/s`**, simulates mid-session node drops, and prints copy-pasteable `llama.cpp` commands.

- **License:** Apache-2.0
- **Dependencies:** Python 3.12+ standard library only (`0` external runtime dependencies, `0` telemetry).
- **Name Note:** Distributed on PyPI as `oddly-local-mesh` (unrelated to 3D polygon mesh generation tools).

---

## Quickstart

```console
lab@mac-studio ~ % git clone https://github.com/oddlyprovenai/local-mesh.git
lab@mac-studio ~ % cd local-mesh
lab@mac-studio ~ % python3 -m pip install -e .
```

### 1. Plan a 30B `Q4_K_M` Split Across Three Old Laptops (`16 GB + 16 GB + 8 GB`)

```console
lab@mac-studio ~ % local-mesh plan --preset presets/three_old_laptops_30b.json --model 30b --quant Q4_K_M
=== local-mesh Roofline Plan: Qwen2.5-32B / 30B-Class (Dense) (Q4_K_M, n_ctx=4096, kv=f16) ===
Model Footprint (weights + KV + scratch): 20.50 GiB | Cluster Usable RAM: 32.50 GiB
Honest Speed Estimate: Decode = 1.43 tok/s (bounds 1.22–1.60 tok/s) | TTFT (512 tok) = 35.68s (bounds 31.40–42.82s)

Node Allocations (OS headroom preserved):
  - thinkpad-t480      [coordinator] layers 00..21 (22) | RAM  8.52/13.50 GiB usable (free OS headroom:  7.48 GiB)
  - latitude-7490      [worker     ] layers 22..46 (25) | RAM  7.58/13.50 GiB usable (free OS headroom:  8.42 GiB)
  - macbook-air-m1     [worker     ] layers 47..63 (17) | RAM  5.30/ 5.50 GiB usable (free OS headroom:  2.70 GiB)

Ready-to-Run llama.cpp Commands:
  # On worker latitude-7490:
  rpc-server --host latitude-7490.local -p 50052
  # On worker macbook-air-m1:
  rpc-server --host macbook-air-m1.local -p 50052
  # On coordinator:
  llama-server -m ./models/qwen2.5-32b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 99 --split-mode layer --rpc latitude-7490.local:50052,macbook-air-m1.local:50052 --tensor-split 22,25,17
```

### 2. Anti-Hype Check: When Adding a Third Wi-Fi Laptop Makes Generation *Slower*

```console
lab@mac-studio ~ % local-mesh plan --preset presets/two_wired_plus_slow_wifi_30b.json --model 30b --quant Q4_K_M
Node Allocations (OS headroom preserved):
  - thinkpad-wired-a   [coordinator] layers 00..29 (30) | RAM 10.80/13.50 GiB usable (free OS headroom:  5.20 GiB)
  - thinkpad-wired-b   [worker     ] layers 30..63 (34) | RAM 10.15/13.50 GiB usable (free OS headroom:  5.85 GiB)
  - old-wifi-notebook  [worker     ] EXCLUDED (slow link / unnecessary hop)

Planner Warnings:
  [!] Excluded 'old-wifi-notebook' (wifi-2.4ghz, RTT=38.0ms, 12 GB/s): naive 3-node split drops decode speed from 1.48 tok/s down to 0.99 tok/s.
```

### 3. Simulate a Mid-Session Laptop Drop (`simulate-drop`)

```console
lab@mac-studio ~ % local-mesh simulate-drop --preset presets/three_old_laptops_30b.json --drop macbook-air-m1 --model 30b --quant Q4_K_M
[REBALANCED] Survived drop of ['macbook-air-m1']: rebalanced 30B Q4_K_M at 1.24 tok/s.
```

If surviving usable RAM drops below the active quantization footprint (for example, running `30B Q8_0` when workers disconnect), `simulate-drop` reports `[DOWNGRADE_REQUIRED]` with the exact GiB deficit and recommends the highest quantization (`Q5_K_M` or `Q4_K_M`) that fits without OOM.

### 4. Benchmark Real TCP Loopback / LAN Links (`bench` & `probe`)

```console
lab@mac-studio ~ % local-mesh bench --nodes 3 --iters 30 --slice-kb 256
=== Measured Local Loopback TCP Socket Probes (127.0.0.1) ===
  - 127.0.0.1:60539: RTT median=0.058 ms (p95=0.071 ms), throughput=2266.5 MB/s (19012.8 Mbps)
  - 127.0.0.1:60540: RTT median=0.056 ms (p95=0.075 ms), throughput=2255.9 MB/s (18923.9 Mbps)
```

---

## Security Warning (`llama.cpp` `rpc-server`)

Upstream `llama.cpp` `rpc-server` has **no authentication and no transport encryption**.
- **Never** expose `rpc-server` on untrusted Wi-Fi, campus/office networks, or router WAN port-forwards.
- Bind `--host` explicitly to an isolated home-LAN `.local` interface, or bind `--host 127.0.0.1` on each worker and forward over SSH (`ssh -L 50052:127.0.0.1:50052`) or WireGuard.

---

## Running Tests

```console
lab@mac-studio ~ % PYTHONPATH=src python3 -m pytest -v
```
