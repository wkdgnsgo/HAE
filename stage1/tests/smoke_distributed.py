"""Two-process CPU Gloo correctness smoke; no DINO weights or dataset required.

Run through torchrun, normally via tests/test_distributed.py.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from hae import distributed
from hae.checkpoints import atomic_save, capture_rng_state, restore_rng_state, seed_all
from hae.training import TrainingEngine


class ToyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_model = nn.Module()
        self.dino_model.patch_embed = nn.Conv2d(3, 3, 1)
        self.decoder = nn.Module()
        self.decoder.project_in = nn.Conv2d(3, 4, 1)
        self.decoder.project_out = nn.Conv2d(4, 3, 1)
        self.register_buffer("broadcast_marker", torch.tensor([123.0]))

    def forward(self, x):
        with torch.no_grad():
            features = self.dino_model.patch_embed(x)
        reconstruction = torch.sigmoid(self.decoder.project_out(torch.tanh(self.decoder.project_in(features))))
        return reconstruction, features.flatten(2).transpose(1, 2)


def config():
    return {"training": {"scope": "decoder", "learning_rate": 0.003,
                          "betas": [0.5, 0.9], "weight_decay": 0.0,
                          "precision": "fp32", "max_grad_norm": 1e6},
            "gan": {"enabled": False, "warmup_updates": 0},
            "losses": {"l1": 1.0, "lpips": 0.0, "alignment": 0.0},
            "ema": {"decay": 0.9}}


def run(output_dir):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4567)
    batches = [torch.rand(8, 3, 4, 4, generator=generator) for _ in range(3)]
    torch.manual_seed(1234)
    reference = ToyAutoencoder()
    expected = TrainingEngine(reference, config(), "cpu")
    for images in batches:
        expected.train_update([images], accumulation_steps=1, world_size=1)

    context = distributed.initialize("cpu")
    try:
        if context.world_size != 2:
            raise ValueError("This test requires exactly two ranks")
        torch.manual_seed(1234 if context.rank == 0 else 9876)
        model = ToyAutoencoder()
        model.broadcast_marker.fill_(123 if context.rank == 0 else -99)
        distributed.broadcast_module(model)
        assert model.broadcast_marker.item() == 123
        initial = distributed.gather_objects({k: v.detach().cpu() for k, v in model.state_dict().items()})
        for key in initial[0]:
            torch.testing.assert_close(initial[0][key], initial[1][key], rtol=0, atol=0)

        p = nn.Parameter(torch.zeros(3))
        q = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        p.grad = torch.full_like(p, context.rank + 1)
        q.grad = torch.full_like(q, 2 * context.rank + 2)
        distributed.average_gradients([p, q], bucket_bytes=8)
        torch.testing.assert_close(p.grad, torch.full_like(p, 1.5), rtol=0, atol=0)
        torch.testing.assert_close(q.grad, torch.full_like(q, 3.0), rtol=0, atol=0)

        engine = TrainingEngine(model, config(), "cpu")
        for images in batches:
            local = images[context.rank * 4:(context.rank + 1) * 4]
            engine.train_update([local[:2], local[2:]], accumulation_steps=2, world_size=2)
        max_error = 0.0
        for name, tensor in model.state_dict().items():
            target = reference.state_dict()[name]
            torch.testing.assert_close(tensor, target, rtol=1e-5, atol=2e-7,
                                       msg=lambda message: f"{name}: {message}")
            max_error = max(max_error, float((tensor - target).abs().max()))
        for name, value in engine.ema.shadow.items():
            torch.testing.assert_close(value, expected.ema.shadow[name], rtol=1e-5, atol=2e-7)
        assert engine.progress == {"stage_update": 3, "generator_updates": 3,
                                   "discriminator_updates": 0, "microbatches": 6,
                                   "samples_seen": 24}
        for parameter in engine.parameters:
            assert engine.optimizer_g.state[parameter]["step"].item() == 3

        final_states = distributed.gather_objects({k: v.detach().cpu() for k, v in model.state_dict().items()})
        for name in final_states[0]:
            torch.testing.assert_close(final_states[0][name], final_states[1][name], rtol=0, atol=0)
        seed_all(100 + context.rank)
        rng_states = distributed.gather_objects(capture_rng_state())
        expected_random = (random.random(), np.random.rand(), torch.rand(4))
        checkpoint = Path(output_dir) / "rng_checkpoint.pt"
        if context.rank == 0:
            atomic_save({"rng_states": rng_states}, checkpoint)
        distributed.barrier()
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seed_all(999)
        restore_rng_state(reloaded["rng_states"][context.rank])
        actual_random = (random.random(), np.random.rand(), torch.rand(4))
        assert actual_random[:2] == expected_random[:2]
        torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
        records = distributed.gather_objects({"rank": context.rank, "progress": engine.progress,
                                              "max_error_vs_global_batch": max_error})
        if context.rank == 0:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "result.json").write_text(json.dumps({"passed": True, "world_size": 2,
                "backend": "gloo", "rank_weights_identical": True,
                "broadcast_parameters_and_buffers": True, "rank_rng_roundtrip": True,
                "records": records}, indent=2) + "\n")
        distributed.barrier()
    finally:
        distributed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args().output_dir)
