import json
import math
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    multiple_of: int
    ffn_dim_multiplier: float
    norm_eps: float


# LLaMA-2-7B configuration
model_args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=4,
    vocab_size=32000,
    max_batch_size=32,
    max_seq_len=2048,
    multiple_of=256,
    ffn_dim_multiplier=None,  # None means use default ~2.688
    norm_eps=1e-5
)

# Compute FFN intermediate size
if model_args.ffn_dim_multiplier is None:
    ffn_dim = int((model_args.dim * 8 / 3) + 0.5)  # default from Meta code
else:
    ffn_dim = int(model_args.dim * model_args.ffn_dim_multiplier)

# Round to multiple_of
ffn_dim = math.ceil(ffn_dim / model_args.multiple_of) * model_args.multiple_of

# Derived values
HEAD_DIM = model_args.dim // model_args.n_heads
B = "B"  # placeholder for batch
S = "S"  # placeholder for sequence length

graph = []
op_id = 0

def add_op(layer, op_type, op_name, input_shape, weight_shape, output_shape, notes=""):
    global op_id
    graph.append({
        "op_id": op_id,
        "layer": layer,
        "op_type": op_type,
        "op_name": op_name,
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "output_shape": output_shape,
        "notes": notes
    })
    op_id += 1


# Embedding
add_op(0, "Embedding", "token_embedding",
       [B, S],
       [model_args.vocab_size, model_args.dim],
       [B, S, model_args.dim],
       "Token embeddings")

# Transformer layers
for layer in range(model_args.n_layers):
    # LayerNorm before attention
    add_op(layer, "LayerNorm", "ln_attn",
           [B, S, model_args.dim],
           [model_args.dim],
           [B, S, model_args.dim])

    # QKV projection
    add_op(layer, "Matmul", "qkv_proj",
           [B, S, model_args.dim],
           [model_args.dim, model_args.dim * 3],
           [B, S, model_args.dim * 3],
           "QKV combined projection")

    # QK matmul
    add_op(layer, "Matmul", "qk_matmul",
           [B, model_args.n_heads, S, HEAD_DIM],
           [B, model_args.n_heads, HEAD_DIM, S],
           [B, model_args.n_heads, S, S],
           "Q · K^T")

    # Softmax
    add_op(layer, "Softmax", "softmax_attn",
           [B, model_args.n_heads, S, S],
           None,
           [B, model_args.n_heads, S, S])

    # Attention weighted value
    add_op(layer, "Matmul", "attn_v",
           [B, model_args.n_heads, S, S],
           [B, model_args.n_heads, S, HEAD_DIM],
           [B, model_args.n_heads, S, HEAD_DIM],
           "Attention weights × V")

    # Merge heads projection
    add_op(layer, "Matmul", "attn_out_proj",
           [B, S, model_args.dim],
           [model_args.dim, model_args.dim],
           [B, S, model_args.dim])

    # All-Reduce after attention
    add_op(layer, "All-Reduce", "allreduce_attn",
           [B, S, model_args.dim],
           None,
           [B, S, model_args.dim])

    # LayerNorm before FFN
    add_op(layer, "LayerNorm", "ln_ffn",
           [B, S, model_args.dim],
           [model_args.dim],
           [B, S, model_args.dim])

    # FFN in projection
    add_op(layer, "Matmul", "ffn_in_proj",
           [B, S, model_args.dim],
           [model_args.dim, ffn_dim],
           [B, S, ffn_dim])

    # FFN out projection
    add_op(layer, "Matmul", "ffn_out_proj",
           [B, S, ffn_dim],
           [ffn_dim, model_args.dim],
           [B, S, model_args.dim])

    # All-Reduce after FFN
    add_op(layer, "All-Reduce", "allreduce_ffn",
           [B, S, model_args.dim],
           None,
           [B, S, model_args.dim])

# Final LayerNorm + LM head
add_op(model_args.n_layers, "LayerNorm", "final_ln",
       [B, S, model_args.dim],
       [model_args.dim],
       [B, S, model_args.dim])

add_op(model_args.n_layers, "Matmul", "lm_head",
       [B, S, model_args.dim],
       [model_args.dim, model_args.vocab_size],
       [B, S, model_args.vocab_size])

# Save JSON
with open("llama2_7b_graph.json", "w") as f:
    json.dump(graph, f, indent=2)

print(f"Graph saved with {len(graph)} operators to llama2_7b_graph.json")
