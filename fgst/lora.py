from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

TARGET_PROJECTIONS = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


class TrajectoryLayer(nn.Module):
    """The paper's rank-r trajectory layer: W + (alpha/r) BA."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / int(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = hidden_states @ self.lora_A.T @ self.lora_B.T
        return self.base(hidden_states) + self.scaling * update


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory_layers(model: nn.Module, adapter_dir: Path) -> dict:
    manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    state_path = adapter_dir / "adapter_state.safetensors"
    actual_hash = sha256_file(state_path)
    if actual_hash != manifest["state_sha256"]:
        raise ValueError(f"adapter SHA-256 mismatch: {actual_hash}")

    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in TARGET_PROJECTIONS:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, TrajectoryLayer(module, manifest["rank"], manifest["alpha"]))
        replaced.append(name)

    state = load_file(str(state_path))
    model_keys = set(model.state_dict())
    incompatible = sorted(set(state) - model_keys)
    if incompatible:
        raise ValueError(f"adapter has incompatible keys, first entries: {incompatible[:5]}")
    result = model.load_state_dict(state, strict=False)
    unexpected = [key for key in result.unexpected_keys if "lora_" in key]
    loaded = sum(key.endswith(("lora_A", "lora_B")) for key in state)
    if unexpected or loaded != len(replaced) * 2:
        raise ValueError(f"trajectory-layer load incomplete: loaded={loaded}, modules={len(replaced)}")
    model.eval()
    return {**manifest, "targeted_modules": len(replaced), "verified_sha256": actual_hash}
