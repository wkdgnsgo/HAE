"""EMA of explicitly trainable parameters, retaining the frozen model weights."""
from contextlib import contextmanager
from collections.abc import Mapping
import math

import torch


def _check_tensors(actual, expected, label):
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        found = set(actual) if isinstance(actual, Mapping) else set()
        raise ValueError(f"{label} keys differ: missing={sorted(set(expected)-found)}, "
                         f"unexpected={sorted(found-set(expected))}")
    for name, target in expected.items():
        value = actual[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"{label}.{name}: expected tensor shape {tuple(target.shape)}")


class ModelEMA:
    """Construct AFTER loading initial weights and selecting the trainable stage.

    Only trainable parameters are averaged. Persistent buffers are copied on each update;
    frozen parameters always come from the supplied model when exporting. The
    trainable parameter set must remain fixed for this EMA object's lifetime.

    With ``warmup``, the decay ramps up as min(decay, (1 + step) / (10 + step)), the
    schedule of TensorFlow's ExponentialMovingAverage(num_updates) and CompVis LDM's
    LitEma, so short stages do not stay biased toward their initial weights. The step
    comes from the caller's checkpointed progress, so the state schema is unchanged.
    """

    def __init__(self, model, decay=0.999, warmup=False):
        self.decay = self._validate_decay(decay)
        self.warmup = bool(warmup)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n, b in self._persistent_buffers(model).items()}
        self._averaging = False

    @staticmethod
    def _validate_decay(decay):
        decay = float(decay)
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1).")
        return decay

    @staticmethod
    def _persistent_buffers(model):
        state_keys = set(model.state_dict())
        return {n: b for n, b in model.named_buffers() if n in state_keys}

    def _validate_model(self, model):
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        buffers = self._persistent_buffers(model)
        _check_tensors(self.shadow, params, "EMA parameters")
        _check_tensors(self.buffers, buffers, "EMA buffers")
        return params, buffers

    def decay_at(self, step=None):
        """Decay to apply after ``step`` previously completed updates."""
        if not self.warmup:
            return self.decay
        if not isinstance(step, int) or step < 0:
            raise ValueError("EMA warmup requires a non-negative integer step.")
        return min(self.decay, (1 + step) / (10 + step))

    @torch.no_grad()
    def update(self, model, step=None):
        if self._averaging:
            raise RuntimeError("Cannot update EMA while average_parameters is active.")
        decay = self.decay_at(step)
        params, buffers = self._validate_model(model)
        for name, param in params.items():
            shadow = self.shadow[name].to(device=param.device, dtype=param.dtype)
            shadow.lerp_(param.detach(), 1 - decay)
            self.shadow[name] = shadow
        for name, buffer in buffers.items():
            self.buffers[name] = buffer.detach().clone()

    def state_dict(self):
        return {"format_version": 1, "decay": self.decay,
                "shadow": {n: v.detach().cpu().clone() for n, v in self.shadow.items()},
                "buffers": {n: v.detach().cpu().clone() for n, v in self.buffers.items()}}

    def load_state_dict(self, state):
        required = {"format_version", "decay", "shadow", "buffers"}
        if not isinstance(state, Mapping) or set(state) != required or state["format_version"] != 1:
            raise ValueError("Unsupported EMA state schema; expected format_version=1.")
        decay = self._validate_decay(state["decay"])
        _check_tensors(state["shadow"], self.shadow, "EMA parameters")
        _check_tensors(state["buffers"], self.buffers, "EMA buffers")
        self.shadow = {n: state["shadow"][n].detach().to(device=v.device, dtype=v.dtype).clone()
                       for n, v in self.shadow.items()}
        self.buffers = {n: state["buffers"][n].detach().to(device=v.device, dtype=v.dtype).clone()
                        for n, v in self.buffers.items()}
        self.decay = decay

    def full_state_dict(self, model):
        self._validate_model(model)
        state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        state.update({n: v.detach().cpu().clone() for n, v in self.shadow.items()})
        state.update({n: v.detach().cpu().clone() for n, v in self.buffers.items() if n in state})
        return state

    @contextmanager
    def average_parameters(self, model):
        if self._averaging:
            raise RuntimeError("Nested EMA averaging is not supported.")
        params, buffers = self._validate_model(model)
        targets = {**params, **buffers}
        values = {**self.shadow, **self.buffers}
        backup = {n: v.detach().clone() for n, v in targets.items()}
        self._averaging = True
        try:
            with torch.no_grad():
                for name, target in targets.items():
                    target.copy_(values[name].to(device=target.device, dtype=target.dtype))
            yield model
        finally:
            with torch.no_grad():
                for name, target in targets.items():
                    target.copy_(backup[name])
            self._averaging = False
