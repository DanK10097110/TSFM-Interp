"""A portable inference wrapper for a frozen adapter and fitted steering profile."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import random
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from ..config import DataConfig, ModelConfig
from ..models import build_adapter
from ..utils import resolve_device, resolve_dtype
from .dynamic import DynamicSteerer
from .model import SteeringModel
from .requests import CombinedRequest, SteeringRequest, request_from_dict


@dataclass
class SteeredForecaster:
    """A request-bound predictor. Artifacts reference the original base checkpoint."""
    steerer: DynamicSteerer
    request: SteeringRequest | CombinedRequest
    model_config: ModelConfig
    data_config: DataConfig
    dtype: str
    seed: int
    checkpoint_archive: Path | None = None

    def compare(self, contexts):
        x = np.asarray(contexts, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.data_config.context_len:
            raise ValueError(f"expected contexts of length {self.data_config.context_len}")
        return self.steerer.steer(x, self.request, prediction_length=self.data_config.horizon,
                                  auto_discover=False)

    def predict(self, contexts):
        """Return steered forecasts with shape [series, horizon, quantile]."""
        return self.compare(contexts).forecast

    def export(self, path, *, include_checkpoint=False):
        """Write one portable ZIP with a model recipe and numeric steering parameters."""
        path = Path(path)
        checkpoint_dir = None
        if include_checkpoint and self.checkpoint_archive is None:
            checkpoint = self.model_config.checkpoint
            if not checkpoint:
                raise ValueError("Including weights requires a named pretrained checkpoint.")
            checkpoint_dir = Path(checkpoint).resolve()
            if not checkpoint_dir.is_dir():
                from huggingface_hub import snapshot_download
                checkpoint_dir = Path(snapshot_download(checkpoint, local_files_only=True))
        metadata = dict(version=1, model=asdict(self.model_config),
                        context_len=self.data_config.context_len, horizon=self.data_config.horizon,
                        data=asdict(self.data_config), dtype=self.dtype, seed=self.seed,
                        quantiles=self.steerer.model.quantiles.tolist(), request=asdict(self.request))
        # Corpora are only needed for fitting. Exclude private local dataset paths.
        metadata['data'].pop('path', None)
        with TemporaryDirectory(dir=path.parent, prefix=".steering-") as tmp:
            root = Path(tmp)
            # Include the policy selected for this export, even after other studio experiments.
            selected = DynamicSteerer(self.steerer.model, self.steerer.config)
            key = self.request.key(self.data_config.horizon)
            if key in self.steerer.profiles:
                selected.profiles[key] = self.steerer.profiles[key]
            elif not self.steerer._direct_context(self.request, self.data_config.horizon):
                raise ValueError("fit this request before export")
            selected.save(root / "profile")
            archive = root / "bundle.zip"
            with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                bundle.writestr("model.json", json.dumps(metadata, indent=2))
                if self.model_config.random_init and self.steerer.model.module is not None:
                    # Other stages may have consumed RNG before the adapter was initialized.
                    # A seed alone cannot reproduce the run's actual random weights.
                    state_file = root / "model-state.pt"
                    torch.save(self.steerer.model.module.state_dict(), state_file)
                    bundle.write(state_file, state_file.name)
                if include_checkpoint and self.checkpoint_archive is not None:
                    with ZipFile(self.checkpoint_archive) as source:
                        for name in source.namelist():
                            if name.startswith("checkpoint/"):
                                with source.open(name) as src, bundle.open(name, "w", force_zip64=True) as dst:
                                    shutil.copyfileobj(src, dst)
                if checkpoint_dir is not None:
                    for entry in sorted(checkpoint_dir.rglob("*")):
                        rel = entry.relative_to(checkpoint_dir)
                        if entry.is_file() and entry != path.resolve() and not any(p.startswith(".") for p in rel.parts):
                            bundle.write(entry, "checkpoint/" + rel.as_posix())
                for name in ("profiles.json", "profiles.npz"):
                    bundle.write(root / "profile" / name, name)
            archive.replace(path)
        return path

    @classmethod
    def load(cls, path, device="cpu", checkpoint=None):
        """Load on a machine with the corresponding adapter and exact base weights.

        checkpoint can point to a relocated copy of the original checkpoint.
        """
        with ZipFile(path) as bundle, TemporaryDirectory() as tmp:
            meta = json.loads(bundle.read("model.json"))
            if meta["version"] != 1:
                raise ValueError("unsupported steered model export version")
            original = ModelConfig(**meta["model"])
            spec = ModelConfig(**meta["model"])
            embedded = [n for n in bundle.namelist() if n.startswith("checkpoint/")]
            if checkpoint is not None:
                spec.checkpoint = str(checkpoint)
            elif embedded:
                for member in embedded:
                    rel = PurePosixPath(member)
                    if ".." in rel.parts or "\\" in member:
                        raise ValueError("invalid checkpoint member")
                    destination = Path(tmp, *rel.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                spec.checkpoint = str(Path(tmp) / "checkpoint")
            data = DataConfig(**meta["data"])
            resolved = resolve_device(device)
            # Adapter construction may initialize tensors. Preserve the host application's RNG streams.
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                devices = [resolved] if resolved.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    random.seed(meta["seed"])
                    np.random.seed(meta["seed"])
                    torch.random.default_generator.manual_seed(meta["seed"])
                    if devices:
                        with torch.cuda.device(resolved):
                            torch.cuda.manual_seed(meta["seed"])
                    adapter = build_adapter(spec, data, resolved, resolve_dtype(meta["dtype"], resolved))
                    model = SteeringModel.from_adapter(adapter, quantiles=meta["quantiles"])
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
            if "model-state.pt" in bundle.namelist():
                with bundle.open("model-state.pt") as state:
                    model.module.load_state_dict(torch.load(state, map_location="cpu", weights_only=True))
            model.model_id = f"{original.adapter}:{original.checkpoint}" + getattr(adapter, "forecast_revision", "")
            for name in ("profiles.json", "profiles.npz"):
                (Path(tmp) / name).write_bytes(bundle.read(name))
            steerer = DynamicSteerer.load(tmp, model)
        return cls(steerer, request_from_dict(meta["request"]), original, data, meta["dtype"], meta["seed"],
                   Path(path).resolve() if embedded and checkpoint is None else None)
