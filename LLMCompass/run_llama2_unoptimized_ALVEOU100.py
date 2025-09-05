#!/usr/bin/env python3
"""
run_llama2.py simulator for LLaMA-2-7B model + Alveo U280 hardware.
- Uses Alveo U280 JSON fields (HBM2 total_bandwidth_GBps, DSP slices @ kernel clock, PCIe link)
- Persists results to simulation_results.json and saves two plots in output_dir
"""

import argparse
import json
import os
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Hardware Model
# ----------------------------------------------------------
class HardwareSimulator:
    def __init__(self, hw_config: Dict, default_batch: int = 1, default_seq: int = 2048):
        self.hw = hw_config
        self.default_batch = int(default_batch)
        self.default_seq = int(default_seq)
        self._normalize_hw()
        self.validate_hardware()

    def _normalize_hw(self):
        """Normalize fields so downstream code can rely on them."""
        # Mirror device.memory to top-level memory if missing
        if 'memory' not in self.hw and 'device' in self.hw and 'memory' in self.hw['device']:
            self.hw['memory'] = self.hw['device']['memory']

        # Ensure device.io exists and basic keys
        io = dict(self.hw.get('device', {}).get('io', {}) or {})
        io.setdefault('memory_channel_active_count',
                      io.get('memory_channel_active_count', io.get('memory_channel_physical_count', 0)))
        io.setdefault('pin_count_per_channel', io.get('pin_count_per_channel', 0))
        io.setdefault('bandwidth_per_pin_bit', io.get('bandwidth_per_pin_bit', 0))
        self.hw.setdefault('device', {})
        self.hw['device']['io'] = io

    def validate_hardware(self):
        for k in ['device', 'interconnect']:
            if k not in self.hw:
                raise ValueError(f"Missing hardware section: {k}")
        if 'frequency_Hz' not in self.hw['device'] and 'typical_kernel_clock_Hz' not in self.hw['device']:
            raise ValueError("Missing device frequency (frequency_Hz or typical_kernel_clock_Hz) in hardware config")

    # -------------------------
    # Helpers
    # -------------------------
    def _get_freq(self) -> float:
        # Prefer explicit 'frequency_Hz' else fall back to 'typical_kernel_clock_Hz'
        return float(self.hw['device'].get('frequency_Hz',
                     self.hw['device'].get('typical_kernel_clock_Hz', 300e6)))

    def _get_memory_bandwidth_bytes(self) -> float:
        """
        Compute memory bandwidth in BYTES/sec.
        Priority:
         1) device.memory.HBM2.total_bandwidth_GBps -> *1e9
         2) device.memory.total_bandwidth_GBps -> *1e9
         3) top-level memory.bandwidth_gbps -> *1e9 / 8
         4) derive from device.io: channels * pins * per_pin_rate (bits/s) -> /8
         5) conservative fallback (1e6 B/s)
        """
        # 1) Prefer HBM2 bandwidth
        dev_mem = self.hw.get('device', {}).get('memory', {})
        if isinstance(dev_mem, dict):
            hbm = dev_mem.get('HBM2', {})
            if isinstance(hbm, dict) and hbm.get('total_bandwidth_GBps'):
                try:
                    return float(hbm['total_bandwidth_GBps']) * 1e9
                except Exception:
                    pass

            # If a flat total_bandwidth_GBps exists at device.memory level
            if dev_mem.get('total_bandwidth_GBps'):
                try:
                    return float(dev_mem['total_bandwidth_GBps']) * 1e9
                except Exception:
                    pass

        # 2) top-level
        top_mem = self.hw.get('memory', {})
        if top_mem.get('total_bandwidth_GBps'):
            try:
                return float(top_mem['total_bandwidth_GBps']) * 1e9
            except Exception:
                pass

        # 3) bandwidth_gbps (Gbit/s)
        if top_mem.get('bandwidth_gbps'):
            try:
                return float(top_mem['bandwidth_gbps']) * 1e9 / 8.0
            except Exception:
                pass

        # 4) derive from IO
        io = self.hw['device'].get('io', {})
        channels = int(io.get('memory_channel_active_count', 0) or 0)
        pins = int(io.get('pin_count_per_channel', 0) or 0)
        per_pin_rate = float(io.get('bandwidth_per_pin_bit', 0) or 0)  # bits/s per pin
        if channels and pins and per_pin_rate:
            bits_per_sec = channels * pins * per_pin_rate
            return bits_per_sec / 8.0

        # 5) fallback to tiny bandwidth to avoid infinities
        return 1e6

    def _get_peak_flops(self) -> float:
        """
        Estimate peak FLOPs/sec at FP16/INT8-like throughput.
        - Prefer a systolic array description if provided
        - Else use DSP slices: assume 2 MACs/DSP/cycle, 2 flops/MAC (multiply+accumulate) -> *2 again
        """
        freq = self._get_freq()
        chip = self.hw['device'].get('compute_chiplet', {})
        core = chip.get('core', {})

        sa = core.get('systolic_array')
        if sa:
            array_w = int(sa.get('array_width', 0))
            array_h = int(sa.get('array_height', 0))
            mac_per_cycle = float(sa.get('mac_per_cycle', 1.0))
            macs_per_cycle = array_w * array_h * mac_per_cycle
            return macs_per_cycle * freq * 2.0

        # Fallback: DSP slices
        resources = self.hw['device'].get('device_resources', {})
        dsp = int(resources.get('dsp_slices', 0) or 0)
        if dsp:
            # (2 MACs per DSP per cycle) * (2 FLOPs per MAC) = *4 FLOPs per DSP per cycle
            flops_per_cycle = dsp * 2.0 * 2.0
            return flops_per_cycle * freq

        return 1e6  # conservative fallback

    # -------------------------
    # Shape parsing
    # -------------------------
    def _resolve_symbolic_dims(self, dims: List[Any]) -> List[int]:
        """
        Convert symbols like B, S, Nh, head_dim into ints.
        """
        defaults = {
            'B': self.default_batch,
            'S': self.default_seq,
            'Nh': int(self.hw['device'].get('n_heads', self.hw['device'].get('n_head', 32))),
            'head_dim': int(self.hw['device'].get('head_dim', 128)),
        }
        resolved = []
        for d in dims or []:
            if isinstance(d, str):
                key = d.strip()
                if key.isdigit():
                    resolved.append(int(key))
                elif key in defaults:
                    resolved.append(int(defaults[key]))
                else:
                    # try generic parse
                    try:
                        resolved.append(int(float(key)))
                    except Exception:
                        resolved.append(1)
            else:
                resolved.append(int(d))
        return resolved if resolved else [1]

    # -------------------------
    # Operator simulation
    # -------------------------
    def simulate_operator(self, op_type: str, input_shape: List[Any], weight_shape: List[Any] = None,
                          output_shape: List[Any] = None) -> Dict:
        """
        Always returns a dict with keys:
          compute_time, memory_time, total_time, compute_cycles, utilization, op_flops
        """
        t = str(op_type or "").strip().lower()

        # Normalize common synonyms
        # Matmul variants
        if t in {"matmul", "matmul_op", "mm", "gemm", "linear", "dense", "fc", "qk_matmul", "attn_v",
                 "attn_out_proj", "ffn_in_proj", "ffn_out_proj", "qkv_proj", "lm_head"}:
            return self._simulate_matmul(input_shape, weight_shape, output_shape)

        # LayerNorm variants
        if t in {"layernorm", "layer_norm", "ln", "ln_attn", "ln_ffn", "final_ln"}:
            return self._simulate_reduction(input_shape)

        # Softmax variants
        if t in {"softmax", "softmax_attn"}:
            return self._simulate_softmax(input_shape)

        # Embedding variants
        if t in {"embedding", "token_embedding", "word_embedding"}:
            return self._simulate_embedding(input_shape, output_shape)

        # All-Reduce variants
        if t in {"all-reduce", "all_reduce", "allreduce", "allreduce_attn", "allreduce_ffn"}:
            return self._simulate_allreduce(output_shape or input_shape)

        # Unknown op: treat as lightweight memory op with safe defaults
        return {
            'compute_time': 0.0,
            'memory_time': 0.0,
            'total_time': 0.0,
            'compute_cycles': 0,
            'utilization': 0.0,
            'op_flops': 0.0
        }

    def _simulate_embedding(self, input_shape: List[Any], output_shape: List[Any]) -> Dict:
        # Prefer output shape to count elements
        if output_shape:
            numeric_out = self._resolve_symbolic_dims(output_shape)
            elements = int(np.prod(numeric_out))
        else:
            numeric_in = self._resolve_symbolic_dims(input_shape)
            hidden = int(self.hw['device'].get('dim', 4096))
            if len(numeric_in) >= 2:
                elements = int(numeric_in[0] * numeric_in[1] * hidden)
            else:
                elements = hidden

        bytes_accessed = elements * 2  # FP16 bytes
        bw_bytes = self._get_memory_bandwidth_bytes()
        mem_time = bytes_accessed / bw_bytes if bw_bytes > 0 else float('inf')

        return {
            'compute_time': 0.0,
            'memory_time': mem_time,
            'total_time': mem_time,
            'compute_cycles': 0,
            'utilization': 0.0,
            'op_flops': 0.0
        }

    def _simulate_matmul(self, input_shape: List[Any], weight_shape: List[Any], output_shape: List[Any]) -> Dict:
        in_shape = self._resolve_symbolic_dims(input_shape or [])
        wt_shape = self._resolve_symbolic_dims(weight_shape or [])
        out_shape = self._resolve_symbolic_dims(output_shape or [])

        # Infer (m, k, n)
        m = k = n = None
        if len(in_shape) == 3 and len(wt_shape) == 2:
            # [B,S,dim] x [dim, n] -> [B,S,n]
            B, S, dim = in_shape
            m = B * S
            k = dim
            n = wt_shape[1]
        elif len(in_shape) == 4 and len(wt_shape) == 4:
            # [B,Nh,S,H] x [B,Nh,H,S] style attention-like multiply
            B, Nh, S, H = in_shape
            m = B * Nh * S
            k = H
            n = wt_shape[-1] if len(wt_shape) >= 1 else S
        elif len(in_shape) == 3 and len(out_shape) == 3:
            # infer n from output
            B, S, dim = in_shape
            m = B * S
            k = dim
            n = out_shape[2]
        else:
            if len(in_shape) >= 2 and len(wt_shape) >= 2:
                m = int(np.prod(in_shape[:-1]))
                k = int(in_shape[-1])
                n = int(wt_shape[-1])
            else:
                m = int(np.prod(in_shape)) if in_shape else 1
                k = int(wt_shape[0]) if wt_shape else 1
                n = int(wt_shape[-1]) if wt_shape else 1

        m, k, n = int(m), int(k), int(n)

        # FLOPs and compute time
        op_flops = 2.0 * m * k * n
        peak_flops = self._get_peak_flops()
        compute_time = op_flops / peak_flops if peak_flops > 0 else float('inf')

        # Memory time: A + B + C reads/writes (rough model)
        bytes_transferred = (m * k + k * n + m * n) * 2  # FP16 bytes
        bw_bytes = self._get_memory_bandwidth_bytes()
        mem_time = bytes_transferred / bw_bytes if bw_bytes > 0 else float('inf')

        total_time = max(compute_time, mem_time)
        utilization = (op_flops / (peak_flops * total_time)) if (peak_flops * total_time) > 0 else 0.0

        return {
            'compute_time': compute_time,
            'memory_time': mem_time,
            'total_time': total_time,
            'compute_cycles': int(np.ceil(compute_time * self._get_freq())),
            'utilization': float(utilization),
            'op_flops': float(op_flops)
        }

    def _simulate_reduction(self, shape: List[Any]) -> Dict:
        numeric = self._resolve_symbolic_dims(shape or [1])
        elements = int(np.prod(numeric))
        # Toy reduction model: O(log(N)) steps, each ~10 cycles, vectorized by vector_width
        vector_width = int(self.hw['device'].get('compute_chiplet', {}).get('core', {})
                           .get('vector_unit', {}).get('vector_width', 32))
        steps = np.ceil(np.log2(max(1, elements / max(1, vector_width))))
        cycles = float(steps * 10)
        freq = self._get_freq()
        compute_time = cycles / freq
        bytes_transferred = elements * 2
        mem_time = bytes_transferred / self._get_memory_bandwidth_bytes()
        total_time = max(compute_time, mem_time)
        return {
            'compute_time': compute_time,
            'memory_time': mem_time,
            'total_time': total_time,
            'compute_cycles': int(np.ceil(cycles)),
            'utilization': 0.0,
            'op_flops': 0.0
        }

    def _simulate_softmax(self, shape: List[Any]) -> Dict:
        # Modeled similarly to reduction with a bit more arithmetic
        numeric = self._resolve_symbolic_dims(shape or [1])
        elements = int(np.prod(numeric))
        freq = self._get_freq()

        # Approx: per element: exp + sum + div -> treat as a few cycles/element amortized by vector unit
        vector_width = int(self.hw['device'].get('compute_chiplet', {}).get('core', {})
                           .get('vector_unit', {}).get('vector_width', 32))
        effective_chunks = max(1, elements // max(1, vector_width))
        cycles = float(effective_chunks * 8)  # toy model
        compute_time = cycles / freq

        bytes_transferred = elements * 2
        mem_time = bytes_transferred / self._get_memory_bandwidth_bytes()
        total_time = max(compute_time, mem_time)
        return {
            'compute_time': compute_time,
            'memory_time': mem_time,
            'total_time': total_time,
            'compute_cycles': int(np.ceil(cycles)),
            'utilization': 0.0,
            'op_flops': float(elements)  # rough counter for “ops”
        }

    def _simulate_allreduce(self, shape: List[Any]) -> Dict:
        numeric = self._resolve_symbolic_dims(shape or [1])
        elements = int(np.prod(numeric))
        bytes_transfer = elements * 2  # FP16
        inter = self.hw.get('interconnect', {})
        link = inter.get('link', {})
        # Use bandwidth_per_direction_byte if provided (already in bytes/sec in your JSON)
        bw_bytes_per_dir = float(link.get('bandwidth_per_direction_byte', 0) or 0)
        link_count = int(inter.get('link_count_per_device', 1) or 1)
        total_bw = bw_bytes_per_dir * max(1, link_count)
        comm_time = bytes_transfer / total_bw if total_bw > 0 else float('inf')
        return {
            'compute_time': 0.0,
            'memory_time': comm_time,
            'total_time': comm_time,
            'compute_cycles': 0,
            'utilization': 0.0,
            'op_flops': 0.0
        }


# ----------------------------------------------------------
# Model loader
# ----------------------------------------------------------
def load_model_config(config_path: str) -> pd.DataFrame:
    """
    Accepts:
      - flat op-list JSON
      - structured JSON with {"layers": [..., {"operators":[...]}, ...]}
      - CSV with headers: layer, op_type, name, input_shape, weight_shape, output_shape
    Returns DataFrame with: layer, op_type, name, input_shape, weight_shape, output_shape
    """
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            ops_list = data
        elif isinstance(data, dict) and 'layers' in data:
            ops_list = []
            for layer in data['layers']:
                layer_id = layer.get('id', layer.get('layer', 0))
                for op in layer.get('operators', []):
                    ops_list.append({
                        'layer': layer_id,
                        'op_type': op.get('type', op.get('op_type')),
                        'name': op.get('name', op.get('op_name')),
                        'input_shape': op.get('shape', op.get('input_shape')),
                        'weight_shape': op.get('weight_shape'),
                        'output_shape': op.get('output_shape'),
                        'notes': op.get('description', op.get('notes', ''))
                    })
        else:
            raise ValueError("Unrecognized model JSON format: expected list or top-level 'layers' key")

        rows = []
        for op in ops_list:
            layer = op.get('layer', op.get('layer_id', 0))
            input_shape = op.get('input_shape') or op.get('shape') or op.get('inputShape') or op.get('in_shape')
            weight_shape = op.get('weight_shape') or op.get('weightShape') or op.get('weight')
            output_shape = op.get('output_shape') or op.get('outputShape') or op.get('out_shape')

            rows.append({
                'layer': int(layer),
                'op_type': op.get('op_type') or op.get('type'),
                'name': op.get('name') or op.get('op_name'),
                'input_shape': input_shape,
                'weight_shape': weight_shape,
                'output_shape': output_shape,
                'notes': op.get('notes', '')
            })

        df = pd.DataFrame(rows)
        return df.reset_index(drop=True)

    # CSV fallback
    return pd.read_csv(config_path)


# ----------------------------------------------------------
# Simulation orchestration
# ----------------------------------------------------------
def simulate_model(model_df: pd.DataFrame, simulator: HardwareSimulator) -> Tuple[List[Dict], Dict]:
    results = []
    total_stats = {
        'total_latency': 0.0,
        'total_ops': 0.0,
        'memory_bound': 0,
        'compute_bound': 0
    }

    print("\n--- Per-Op Timing Breakdown ---")
    print(f"{'Layer':<6} {'Op':<25} {'Type':<12} {'CompTime(s)':>12} {'MemTime(s)':>12} {'TotalTime(s)':>12} Bound")

    for _, row in model_df.iterrows():
        op_type = str(row['op_type'])
        input_shape = row.get('input_shape') or row.get('shape') or []
        weight_shape = row.get('weight_shape') or []
        output_shape = row.get('output_shape') or []

        op_result = simulator.simulate_operator(op_type, input_shape, weight_shape, output_shape)

        # Ensure we always have a dict (belt-and-braces)
        if not isinstance(op_result, dict):
            op_result = {
                'compute_time': 0.0,
                'memory_time': 0.0,
                'total_time': 0.0,
                'compute_cycles': 0,
                'utilization': 0.0,
                'op_flops': 0.0
            }

        # FLOPs accounting
        op_flops = float(op_result.get('op_flops', 0.0))
        if op_flops == 0.0 and str(op_type).strip().lower() in {"matmul", "matmul_op", "mm", "gemm", "linear", "dense", "fc"}:
            # re-evaluate just in case shapes were odd
            try:
                mm = simulator._simulate_matmul(input_shape, weight_shape, output_shape)
                op_flops = float(mm.get('op_flops', 0.0))
            except Exception:
                op_flops = 0.0

        if op_flops == 0.0 and str(op_type).strip().lower() not in {'matmul', 'mm', 'gemm', 'linear', 'dense', 'fc'}:
            # simple proxy: output size (or input size)
            try:
                out_dims = simulator._resolve_symbolic_dims(output_shape) if output_shape else simulator._resolve_symbolic_dims(input_shape)
                op_flops = float(np.prod(out_dims))
            except Exception:
                op_flops = 0.0

        comp_t = float(op_result.get('compute_time', 0.0))
        mem_t = float(op_result.get('memory_time', 0.0))
        tot_t = float(op_result.get('total_time', max(comp_t, mem_t)))

        total_stats['total_latency'] += tot_t
        total_stats['total_ops'] += op_flops

        # Determine bound type by which side dominates total
        bound_type = "Compute" if comp_t >= mem_t else "Memory"
        if bound_type == "Compute":
            total_stats['compute_bound'] += 1
        else:
            total_stats['memory_bound'] += 1

        print(f"{str(row['layer']):<6} {str(row.get('name')):<25} {op_type:<12} "
              f"{comp_t:>12.6f} {mem_t:>12.6f} {tot_t:>12.6f} {bound_type}")

        results.append({
            'layer': int(row['layer']),
            'op': row.get('name'),
            'type': op_type,
            'latency_s': tot_t,
            'compute_time_s': comp_t,
            'memory_time_s': mem_t,
            'compute_cycles': int(op_result.get('compute_cycles', 0)),
            'utilization': float(op_result.get('utilization', 0.0)),
            'flops': float(op_flops)
        })

    total_stats['throughput_tflops'] = (total_stats['total_ops'] / total_stats['total_latency']) / 1e12 \
        if total_stats['total_latency'] > 0 else 0.0

    print("--- End of Timing Breakdown ---\n")
    return results, total_stats


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------
def plot_results(results: List[Dict], total_stats: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    ops_labels = [f"{r['layer']}_{r['op']}" for r in results]
    latencies = [r['latency_s'] for r in results]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(latencies)), latencies)
    plt.xticks(range(len(ops_labels)), ops_labels, rotation=90, fontsize=6)
    plt.ylabel("Latency (s)")
    plt.title(f"Operator Latencies — Total: {total_stats['total_latency']:.6f}s | "
              f"Throughput: {total_stats['throughput_tflops']:.3f} TFLOPS")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_breakdown.png"))
    plt.close()

    bound_types = ['Memory Bound', 'Compute Bound']
    bound_counts = [total_stats['memory_bound'], total_stats['compute_bound']]
    plt.figure(figsize=(5, 5))
    plt.pie(bound_counts, labels=bound_types, autopct='%1.1f%%')
    plt.title("Operation Bound Analysis")
    plt.savefig(os.path.join(output_dir, "bound_analysis.png"))
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLMCompass-style Simulation for LLaMA-2-7B")
    parser.add_argument("--model_config", required=True, help="Path to model config (JSON or CSV)")
    parser.add_argument("--hw_config", required=True, help="Path to hardware config JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (B) to use for symbolic dims")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (S) to use for symbolic dims")
    args = parser.parse_args()

    model_df = load_model_config(args.model_config)
    hw_cfg = json.load(open(args.hw_config, 'r'))

    if model_df.shape[0] == 0:
        raise RuntimeError("Loaded model config contains no operators")

    simulator = HardwareSimulator(hw_cfg, default_batch=args.batch, default_seq=args.seq_len)

    results, total_stats = simulate_model(model_df, simulator)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "simulation_results.json")
    with open(out_path, "w") as f:
        json.dump({
            'results': results,
            'summary': total_stats,
            'hardware': hw_cfg
        }, f, indent=2)

    plot_results(results, total_stats, args.output_dir)

    print(f"Simulation completed. Results saved to {args.output_dir}")
    print(f"Total Latency: {total_stats['total_latency']:.6f}s")
    print(f"Throughput: {total_stats['throughput_tflops']:.3f} TFLOPS")
    print(f"Memory Bound Ops: {total_stats['memory_bound']}; Compute Bound Ops: {total_stats['compute_bound']}")


if __name__ == "__main__":
    main()

