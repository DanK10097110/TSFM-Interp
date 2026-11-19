"""Local desktop controls for fitting, comparing and exporting steering requests."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from ..config import PipelineConfig, DataConfig, load_config
from ..pipeline import Context
from ..utils import set_seed
from . import DynamicSteerer, SearchConfig, SteeredForecaster, SteeringModel, SteeringRequest, TASKS
from .report import _sample
from .requests import CombinedRequest, request_from_dict

# Options relevant to each request, with starting values for new configurations.
_OPTIONS = {
    "seasonality": {"period": 8, "phase": 0}, "deseasonalize": {"period": 8},
    "mean_revert": {"value": None}, "smooth": {"smoothing": 2}, "roughness": {"smoothing": 2},
    "quantile": {"quantile": 0.9}, "anchor": {"value": 0}, "bounds": {"lower": -1, "upper": 1},
    "delay": {"delay": 1}, "context_focus": {"context_window": "recent", "outside_weight": 0.2},
    "context_ignore": {"context_window": "recent"}, "context_emphasize": {"context_window": "recent"},
}


class StudioSession:
    def __init__(self, cfg, predictor=None):
        self.cfg = cfg
        self.ctx = Context(cfg)
        self.search = SearchConfig(seed=cfg.run.seed, **cfg.steering.search)
        self.contexts, self.train, self.valid, self.examples = _sample(
            self.ctx.data, cfg.run.seed, max(3, cfg.steering.max_series),
            self.search.calibration_samples, self.search.validation_samples)
        self.active_model = predictor.model_config.name if predictor else None
        self.steerer = predictor.steerer if predictor else None
        self.checkpoint_archive = predictor.checkpoint_archive if predictor else None

    def compare(self, model_name, request, context):
        replace(self.cfg.steering, requests=[asdict(request)]).validate(self.cfg.data)
        x = np.asarray(context, dtype=np.float32)
        if x.shape != (self.cfg.data.context_len,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Enter exactly {self.cfg.data.context_len} finite context values.")
        if model_name != self.active_model:
            if self.active_model:
                self.ctx.hub.release(self.active_model)
            self.steerer = self.active_model = self.checkpoint_archive = None
            set_seed(self.cfg.run.seed)
            model = SteeringModel.from_adapter(self.ctx.hub.get(model_name),
                                               quantiles=self.cfg.steering.quantiles)
            self.steerer = DynamicSteerer(model, self.search)
            self.active_model = model_name
        horizon = self.cfg.data.horizon
        if request.key(horizon) not in self.steerer.profiles and not self.steerer._direct_context(request, horizon):
            self.steerer.fit(self.contexts[self.train], request, prediction_length=horizon,
                             validation_contexts=self.contexts[self.valid])
        spec = next(m for m in self.cfg.models if m.name == model_name)
        predictor = SteeredForecaster(self.steerer, request, spec, self.cfg.data,
                                      self.cfg.run.dtype, self.cfg.run.seed, self.checkpoint_archive)
        result = predictor.compare(x)
        replay = self.steerer.replay(x[None], result)
        return predictor, result, np.allclose(result.forecast, replay, rtol=1e-5, atol=1e-6)


def launch(cfg, predictor=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    session = StudioSession(cfg, predictor)
    root = tk.Tk()
    root.title("TSFM-Lens · Steering studio")
    body = ttk.Frame(root, padding=14)
    body.pack(fill="both", expand=True)
    body.columnconfigure(1, weight=1)
    status = tk.StringVar(value="Choose a reference context or enter your own values.")
    model = tk.StringVar(value=cfg.models[0].name)
    kind = tk.StringVar(value=(cfg.steering.requests or [{"kind": "level"}])[0]["kind"])
    strength = tk.StringVar()
    option_values = {}
    pending_terms = []
    mix_label = tk.StringVar()
    ref = tk.StringVar()
    labels = [str(session.ctx.data.meta.iloc[i]["series_id"]) for i in session.examples[:3]]
    executor = ThreadPoolExecutor(max_workers=1)
    displayed = None

    def dropdown(row, title, variable, values, event=None):
        ttk.Label(body, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12))
        widget = ttk.Combobox(body, state="readonly", textvariable=variable, values=values)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        if event:
            widget.bind("<<ComboboxSelected>>", event)
        return widget

    def change_direction(event=None):
        preset = next((r for r in cfg.steering.requests if r["kind"] == kind.get()),
                      {"kind": kind.get(), **_OPTIONS.get(kind.get(), {})})
        for widget in settings.winfo_children():
            widget.destroy()
        option_values.clear()
        if kind.get() == "combined":
            pending_terms.clear()
            mix_label.set("")
            terms = preset.get("terms", [{"kind": "level", "strength": .2}, {"kind": "trend", "strength": -.2}])
            strength.set("")
            amount.configure(state="disabled")
            ttk.Label(settings, text="Forecast directions and amounts (JSON list)").pack(anchor="w")
            value = tk.StringVar(value=json.dumps(terms))
            option_values["terms"] = value
            ttk.Entry(settings, textvariable=value, width=75).pack(fill="x")
            return
        item = asdict(SteeringRequest(**preset))
        if kind.get() == "quantile" and item["quantile"] not in cfg.steering.quantiles:
            item["quantile"] = max(cfg.steering.quantiles)
        strength.set(item["strength"])
        amount.configure(state="disabled" if kind.get() in {"bounds", "delay", "context_focus"} else "normal")
        defaults = {"forecast_window": "all", **_OPTIONS.get(kind.get(), {})}
        for column, name in enumerate(defaults):
            value = item[name]
            if isinstance(value, (tuple, list)):
                value = ", ".join(str(step) for step in value)
            var = tk.StringVar(value="" if value is None else value)
            option_values[name] = var
            ttk.Label(settings, text=name.replace("_", " ").capitalize()).grid(row=0, column=column, sticky="w")
            if name.endswith("_window"):
                choices = ("all", "short", "long") if name == "forecast_window" else ("recent", "old", "all")
                widget = ttk.Combobox(settings, textvariable=var, values=choices, width=13)
            elif name == "quantile":
                widget = ttk.Combobox(settings, textvariable=var, values=cfg.steering.quantiles, width=8,
                                      state="readonly")
            else:
                widget = ttk.Entry(settings, textvariable=var, width=11)
            widget.grid(row=1, column=column, sticky="ew", padx=(0, 5))


    dropdown(0, "Model", model, [m.name for m in cfg.models])
    dropdown(1, "Direction", kind, (*TASKS, "combined"), change_direction)
    ttk.Label(body, text="Strength").grid(row=2, column=0, sticky="w")
    amount = ttk.Spinbox(body, textvariable=strength, from_=-1e6, to=1e6, increment=.05)
    amount.grid(row=2, column=1, sticky="ew")
    ttk.Label(body, text="Request settings").grid(row=3, column=0, sticky="w")
    settings = ttk.Frame(body)
    settings.grid(row=3, column=1, sticky="ew", pady=4)
    change_direction()

    def use_reference(event=None):
        row = session.examples[labels.index(ref.get())]
        input_box.delete("1.0", "end")
        input_box.insert("1.0", ", ".join(str(v) for v in session.contexts[row]))

    dropdown(4, "Reference input", ref, labels, use_reference)
    ttk.Label(body, text=f"Context ({cfg.data.context_len} values)").grid(row=5, column=0, sticky="nw")
    input_box = tk.Text(body, height=4, width=76, wrap="word")
    input_box.grid(row=5, column=1, sticky="ew")
    ref.set(labels[0])
    use_reference()
    actions = ttk.Frame(body)
    actions.grid(row=6, column=0, columnspan=2, sticky="w", pady=8)
    ttk.Label(body, textvariable=status, wraplength=720).grid(row=7, column=0, columnspan=2, sticky="w")
    figure = Figure(figsize=(8, 3.6), tight_layout=True)
    axis = figure.subplots()
    axis.set(xlabel="Forecast step", ylabel="Forecast")
    canvas = FigureCanvasTkAgg(figure, body)
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, sticky="nsew")
    body.rowconfigure(8, weight=1)

    def busy(task, complete):
        run_button.state(["disabled"])
        export_button.state(["disabled"])
        future = executor.submit(task)

        def poll():
            if not future.done():
                root.after(80, poll)
                return
            try:
                complete(future.result())
            except Exception as exc:
                status.set(str(exc))
            run_button.state(["!disabled"])
            if displayed:
                export_button.state(["!disabled"])
        root.after(80, poll)

    def draw(payload):
        nonlocal displayed
        displayed, result, matched = payload
        q = int(np.argmax(np.linalg.norm(result.requested_change[0], axis=0)))
        axis.clear()
        for values, label, style in ((result.baseline, "Unsteered", "-"),
                                     (result.baseline + result.requested_change, "Requested", "--"),
                                     (result.forecast, "Steered", ":")):
            axis.plot(np.arange(1, values.shape[1]+1), values[0,:,q], style, label=label)
        axis.set(xlabel="Forecast step", ylabel=f"Forecast (q={result.quantiles[q]:g})",
                 title=f"{displayed.model_config.name} · {displayed.request.kind}")
        axis.legend()
        canvas.draw()
        info = result.diagnostics[0]
        target = "Zero requested change" if info["zero_target"] else f"Relative error {info['relative_error']:.1%}"
        if info["radius"] is not None and np.isclose(info["coefficient_norm"], info["radius"]):
            target += " · edit radius reached"
        status.set(f"{target} · replay {'matched' if matched else 'failed'} · {info['site']['source']}")

    def current_request():
        parameters = {}
        for name, var in option_values.items():
            value = var.get().strip()
            if name == "terms":
                parameters[name] = json.loads(value)
            elif name.endswith("_window"):
                if "," in value:
                    value = tuple(int(step.strip()) for step in value.strip("[]()").split(","))
                parameters[name] = value
            elif value:
                parameters[name] = float(value)
        if kind.get() != "combined":
            parameters["strength"] = float(strength.get())
        return request_from_dict(dict(kind=kind.get(), **parameters))

    def add_component():
        try:
            request = current_request()
            if isinstance(request, CombinedRequest) or request.kind.startswith("context_"):
                raise ValueError("Choose a forecast direction to add to the mix.")
            pending_terms.append(request)
            mix_label.set(" + ".join(r.kind for r in pending_terms) + " + current direction")
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid component", str(exc))

    def clear_mix():
        pending_terms.clear()
        mix_label.set("")

    def compare():
        nonlocal displayed
        try:
            request = current_request()
            if pending_terms:
                request = CombinedRequest((*pending_terms, request))
            text = input_box.get("1.0", "end").strip().strip("[]")
            values = [float(v) for v in text.replace(",", " ").split()]
            name = model.get()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        displayed = None
        status.set("Fitting and comparing forecasts…")
        busy(lambda: session.compare(name, request, values), draw)

    weights = tk.BooleanVar(value=False)

    def export():
        include = weights.get()
        path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Steered model", "*.zip")])
        if path:
            status.set("Exporting displayed model…")
            busy(lambda: displayed.export(path, include_checkpoint=include), lambda saved: status.set(f"Exported {saved}"))

    mix = ttk.Frame(body)
    mix.grid(row=9, column=0, columnspan=2, sticky="w")
    ttk.Button(mix, text="Add current direction to mix", command=add_component).pack(side="left")
    ttk.Button(mix, text="Clear mix", command=clear_mix).pack(side="left", padx=6)
    ttk.Label(mix, textvariable=mix_label).pack(side="left")
    run_button = ttk.Button(actions, text="Compare forecasts", command=compare)
    run_button.pack(side="left", padx=(0, 8))
    export_button = ttk.Button(actions, text="Export displayed model…", command=export, state="disabled")
    export_button.pack(side="left")
    ttk.Checkbutton(actions, variable=weights, text="Include cached base checkpoint").pack(side="left", padx=8)
    try:
        root.mainloop()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="toolkit run YAML")
    source.add_argument("--bundle", help="a report or studio export ZIP")
    args = parser.parse_args()
    if args.config:
        launch(load_config(args.config))
    else:
        predictor = SteeredForecaster.load(args.bundle)
        cfg = PipelineConfig()
        cfg.run.device, cfg.run.dtype, cfg.run.seed = "cpu", predictor.dtype, predictor.seed
        cfg.models = [predictor.model_config]
        cfg.data = DataConfig(context_len=predictor.data_config.context_len, horizon=predictor.data_config.horizon)
        cfg.steering.requests = [asdict(predictor.request)]
        cfg.steering.quantiles = predictor.steerer.model.quantiles.tolist()
        cfg.steering.search = asdict(predictor.steerer.config)
        cfg.steering.search.pop("seed")
        launch(cfg, predictor)


if __name__ == "__main__":
    main()
