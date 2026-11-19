"""Bounded discovery and reusable response profiles for previously unseen models.

No model names, layer names, SAE features or differentiable prediction methods
are built into the search. A profile learns a normalized response map, chooses
a site using separate validation contexts, and optionally refreshes the local
response at inference. All measurements use real forecasts under input hooks.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from .controller import trust_region_direction
from .model import InterventionSite, SteeringModel, SteeringUnavailable
from .requests import CombinedRequest, request_from_dict, SteeringRequest, context_counterfactual, requested_change, window_mask
from .solver import response_metrics
from .readout import ReadoutSpace


@dataclass
class SearchConfig:
    """Compute bounds and optional expert overrides; location choice is automatic."""
    max_candidates: int = 12
    screen_candidates: int = 64
    rank: int = 256
    basis_strategy: str = 'auto'
    calibration_samples: int = 2
    validation_samples: int = 2
    radius: float = .2
    ridge: float = 1e-5
    probe_step: float = .005
    probe_batch: int = 16
    refinement_steps: int = 2
    max_calls: int = 4000
    max_seconds: float = 300.
    include_input: bool = True
    context_mode: str = 'auto'
    reuse_profiles: bool = True
    module_pattern: str | None = None
    token_modes: tuple = ('last','shared')
    min_width: int = 4
    max_width: int = 8192
    max_batch_multiplier: int = 64
    seed: int = 0
    tolerance: float = .1

    def __post_init__(self):
        for name in ('max_candidates','screen_candidates','rank','calibration_samples','validation_samples','probe_batch','max_calls','min_width','max_width','max_batch_multiplier'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if self.refinement_steps<0 or not isinstance(self.refinement_steps,int):
            raise ValueError('refinement_steps must be a nonnegative integer')
        for name in ('radius','probe_step','max_seconds','tolerance'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.ridge) or self.ridge<0:
            raise ValueError('ridge must be finite and nonnegative')
        if not self.token_modes or any(m not in ('last','shared','full') for m in self.token_modes):
            raise ValueError('token_modes must use last/shared/full')
        if self.module_pattern is not None:re.compile(self.module_pattern)
        if self.context_mode not in ('auto','discover','input_policy'):
            raise ValueError('context_mode must be auto/discover/input_policy')
        if self.basis_strategy not in ('auto','random'):
            raise ValueError('basis_strategy must be auto/random')
        if self.context_mode=='input_policy' and not self.include_input:
            raise ValueError('input_policy requires include_input=True')


@dataclass
class SteeringProfile:
    """A learned site/basis/mean response, bound to one request and geometry."""
    site: InterventionSite
    basis: np.ndarray
    response: np.ndarray
    request: SteeringRequest | CombinedRequest
    prediction_length: int
    context_length: int
    report: dict


@dataclass
class DynamicSteeringResult:
    """Actual forecasts and explicit interventions, in original series units."""
    baseline: np.ndarray
    forecast: np.ndarray
    requested_change: np.ndarray
    diagnostics: list
    interventions: list
    quantiles: np.ndarray


class _BudgetExhausted(SteeringUnavailable):
    pass


class DynamicSteerer:
    """Discover useful interventions for a registered forecast model.

    fit uses separate calibration and validation contexts. With no fitted
    profile, steer may run same-context online discovery, labeled explicitly
    and never described as validation. Model weights are never trained.
    Calls on this object are serialized while temporary hooks are installed.
    """
    def __init__(self,model:SteeringModel,config:SearchConfig | None=None):
        self.model=model;self.config=config or SearchConfig();self.profiles={}
        self._started=0.;self._start_calls=0
        self._budget_fraction=1.
        self._model_stamp=self._stamp()
        self._basis_sources={}
        self._readout_rows = {}

    def _stamp(self):
        """Cheap guard against normal in-place updates; saved profiles also hash weights."""
        if self.model.module is None:return (self.model.model_id,)
        return (self.model.model_id,tuple((name,id(t),t._version,tuple(t.shape))
            for name,t in list(self.model.module.named_parameters())+list(self.model.module.named_buffers())),
            tuple(m.training for m in self.model.module.modules()))

    def _check_model(self):
        if self._stamp()!=self._model_stamp:
            raise SteeringUnavailable('registered model state changed; create a new steerer and refit')

    def _begin(self):
        self._started=time.monotonic();self._start_calls=self.model.calls

    def _check_budget(self):
        if self.model.calls-self._start_calls>=self.config.max_calls*self._budget_fraction or time.monotonic()-self._started>self.config.max_seconds*self._budget_fraction:
            raise _BudgetExhausted('steering call/time budget exhausted; increase the explicit search budget')

    def _forecast(self,x,h):
        self._check_budget();return self.model.forecast(x,h)

    def _evaluate(self,x,h,site,basis,z):
        self._check_budget();return self.model.evaluate(x,h,site,basis,z)

    @staticmethod
    def _scale(x):
        return max(float(np.std(x)),1e-6)

    def _basis(self,site):
        """Use an exact linear-input effect span when cheap; otherwise random probes.

        A Linear module cannot see directions in its weight's null space. Its
        row space loses no possible intervention effect at this module input,
        and avoids wasting the norm budget on invisible random components.
        This is a generic module property, not a forecast-head name heuristic.
        """
        width=site.width;rank=min(self.config.rank,width)
        self._basis_sources[site.label]='random orthonormal probe subspace'
        if rank==width:
            self._basis_sources[site.label]='complete input coordinate space'
            return np.eye(width)
        if self.config.basis_strategy=='auto' and site.source=='module_input' and self.model.module is not None and site.mode!='full':
            module=self.model.module.get_submodule(site.module)
            repeats=len(site.occurrences) if site.occurrences is not None else 1
            if (type(module) is torch.nn.Linear and module.in_features==site.shape[-1]
                    and module.out_features*repeats<=rank):
                w=module.weight.detach().float().cpu().numpy()
                _,s,vh=np.linalg.svd(w.astype(float),full_matrices=False)
                rows=vh[s>max(w.shape)*np.finfo(float).eps*s[0]]
                if len(rows):
                    basis=np.zeros((len(rows)*repeats,width))
                    for i in range(repeats):basis[i*len(rows):(i+1)*len(rows),i*w.shape[1]:(i+1)*w.shape[1]]=rows
                    self._basis_sources[site.label]='exact row space of discovered Linear input (independent per invocation)'
                    return basis
        seed=int.from_bytes(hashlib.sha256((site.label+str(self.config.seed)).encode()).digest()[:8],'little')
        matrix=np.random.default_rng(seed).normal(size=(width,rank))
        return np.linalg.qr(matrix,mode='reduced')[0].T

    def _infer_readout(self, x, h, site, layer, base):
        """Identify returned rows of an unfamiliar wide Linear from four input perturbations.

        Each row has a characteristic response across random directions. A sliced
        or reordered affine readout preserves that pattern up to a scale on each forecast coordinate.
        The fourth direction checks the mapping independently of its selection.
        """
        rng = np.random.default_rng(self.config.seed)
        directions = rng.normal(size=(4, site.width))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        weight = layer.weight.detach()
        projected = weight @ torch.as_tensor(directions.T, device=weight.device, dtype=weight.dtype)
        predicted = projected.float().cpu().numpy().astype(float)
        step = min(self.config.probe_step, self.config.radius)
        observed = np.stack([
            (self._evaluate(x[None], h, site, direction[None], np.array([[step]]))[0] - base).ravel()
            for direction in directions
        ], axis=1)
        norms = np.linalg.norm(predicted[:, :3], axis=1)
        patterns = predicted[:, :3] / np.maximum(norms[:, None], 1e-30)
        rows = np.array([np.argmax(patterns @ response[:3]) for response in observed])
        expected = predicted[rows]
        power = np.sum(expected[:, :3] ** 2, axis=1)
        if np.any(power == 0):
            return None
        gains = np.sum(expected[:, :3] * observed[:, :3], axis=1) / power
        residual = observed - gains[:, None] * expected
        if (np.all(gains > 0) and np.all(np.linalg.norm(residual, axis=1) <= .02 * np.linalg.norm(observed, axis=1))
                and np.linalg.norm(residual[:, 3]) <= .02 * np.linalg.norm(observed[:, 3])):
            return rows
        return None

    def _candidates(self,x,h):
        """Screen real output effects before spending full Jacobians on candidates.

        Executed branches need not contribute to the registered output. A cheap
        signed probe rejects inactive branches without architecture assumptions.
        """
        self._check_budget()
        sites=self.model.trace(x[None],h,min_width=self.config.min_width,max_width=self.config.max_width,
                               modes=tuple(self.config.token_modes),max_batch_multiplier=self.config.max_batch_multiplier)
        if self.config.module_pattern is not None:
            sites=[s for s in sites if re.search(self.config.module_pattern,s.module)]
        total=len(sites)
        allowance=self.config.max_candidates-int(self.config.include_input)
        self._readout_rows = {}
        preferred = []
        if self.config.basis_strategy == 'auto' and self.model.readout_rows and allowance:
            declared = self.model.readout_rows(h, self.model.quantiles)
            for name, indices in declared.items():
                for site in sites:
                    if (site.module == name and site.occurrence == 0 and site.occurrences is None
                            and site.mode in ('last', 'single', 'shared') and site.batch_multiplier == 1
                            and (site.mode != 'shared' or site.shape[0] == 1)):
                        module = self.model.module.get_submodule(name)
                        indices = np.asarray(indices)
                        if (isinstance(module, torch.nn.Linear) and indices.shape == (h * len(self.model.quantiles),)
                                and np.issubdtype(indices.dtype, np.integer)
                                and np.all((indices >= 0) & (indices < module.out_features))):
                            preferred.append(site)
                            self._readout_rows[site.label] = indices.copy()
                            break
                if len(preferred) == allowance:
                    break
        self._inferred_sites = set()
        if (not preferred and allowance and self.config.basis_strategy == "auto"
                and self.model.module is not None):
            # Small Linear heads already get their complete row space in _basis.
            # Try only the last two wide heads, leaving the normal search available.
            tried = set()
            base = None
            for site in reversed(sites):
                if (site.mode not in ("last", "single") or site.occurrence or site.occurrences
                        or site.batch_multiplier != 1 or site.module in tried):
                    continue
                layer = self.model.module.get_submodule(site.module)
                if not isinstance(layer, torch.nn.Linear) or min(layer.weight.shape) <= self.config.rank:
                    continue
                if self.config.max_calls - (self.model.calls - self._start_calls) < 16:
                    break
                tried.add(site.module)
                if base is None:
                    base = self._forecast(x[None], h)[0]
                try:
                    rows = self._infer_readout(x, h, site, layer, base)
                except _BudgetExhausted:
                    raise
                except (SteeringUnavailable, ValueError, RuntimeError):
                    rows = None
                if rows is not None:
                    self._inferred_sites.add(site.label)
                    self._readout_rows[site.label] = rows
                    preferred.append(site)
                if len(tried) == 2 or len(preferred) == allowance:
                    break
        preferred_modules = {p.module for p in preferred}
        sites = [s for s in sites if s.module not in preferred_modules]
        allowance -= len(preferred)
        if allowance == 0:
            sites = []
        screening=[]
        if len(sites)>allowance and allowance>0:
            limit=min(len(sites),max(allowance,self.config.screen_candidates))
            late=limit//2;early=limit-late
            indices=list(np.linspace(0,len(sites)-late-1,early,dtype=int))+list(range(len(sites)-late,len(sites)))
            active=[]
            for index in sorted(set(indices)):
                site=sites[index]
                try:
                    rng=np.random.default_rng(self.config.seed+index)
                    basis=rng.normal(size=(1,site.width));basis/=np.linalg.norm(basis)
                    plus=self._evaluate(x[None],h,site,basis,np.array([[self.config.probe_step]]))[0]
                    minus=self._evaluate(x[None],h,site,basis,np.array([[-self.config.probe_step]]))[0]
                    effect=float(np.linalg.norm(plus-minus)/self._scale(x))
                    screening.append(dict(site=site.label,normalized_signed_effect=effect))
                    if effect>1e-7:active.append(site)
                except _BudgetExhausted:raise
                except (SteeringUnavailable,ValueError,RuntimeError) as error:
                    screening.append(dict(site=site.label,reason=str(error)))
            sites=active
        if len(sites)>allowance:
            late=min(allowance//2,len(sites));early=allowance-late
            indices=list(np.linspace(0,max(0,len(sites)-late-1),early,dtype=int)) if early else []
            indices+=list(range(len(sites)-late,len(sites)))
            sites=[sites[i] for i in sorted(set(indices))]
        sites = preferred + sites
        if self.config.include_input:
            sites.append(InterventionSite('__input__',[],0,[len(x)],'single','input','forecast_input'))
        if not sites:raise SteeringUnavailable('no supported internal sites; supply a module override or enable input interventions')
        return sites,total,screening

    def _jacobian(self,x,h,site,basis,z,base=None):
        """Central differences, with per-site intervention batching verified empirically."""
        rank=len(basis);step=self.config.probe_step;scale=self._scale(x)
        zero=self._evaluate(x[None],h,site,basis,z[None])[0]
        if base is not None and np.max(abs(zero-base))/scale>2e-5:
            raise SteeringUnavailable('zero intervention changed the forecast or predictor is stochastic')
        probe=np.stack([z+step*np.eye(rank)[0],z-step*np.eye(rank)[0]])
        serial=np.stack([self._evaluate(x[None],h,site,basis,row[None])[0] for row in probe])
        batch_size=1;batch_reason='disabled'
        if self.config.probe_batch>1:
            try:
                together=self._evaluate(np.repeat(x[None],2,axis=0),h,site,basis,probe)
                if np.max(abs(together-serial))/scale<=2e-5:
                    batch_size=self.config.probe_batch;batch_reason='two signed interventions match singleton forecasts'
                else:batch_reason='intervention batch dependence detected; serial fallback'
            except _BudgetExhausted:raise
            except (ValueError,RuntimeError) as error:
                batch_reason=f'batch geometry unavailable; serial fallback: {error}'
        coefficients=np.concatenate([z+step*np.eye(rank),z-step*np.eye(rank)])
        if batch_size>2:
            # Batch-size-dependent predictors may pass a two-row check but fail
            # at the actual probe size. Check opposite ends of that batch too.
            try:
                audit=coefficients[:batch_size]
                together=self._evaluate(np.repeat(x[None],len(audit),axis=0),h,site,basis,audit)
                last=self._evaluate(x[None],h,site,basis,audit[-1:])[0]
                if max(np.max(abs(together[0]-serial[0])),np.max(abs(together[-1]-last)))/scale>2e-5:
                    batch_size=1;batch_reason='actual probe-size dependence detected; serial fallback'
                else:batch_reason='signed pair and actual probe-size endpoints match singleton forecasts'
            except _BudgetExhausted:raise
            except (ValueError,RuntimeError) as error:
                batch_size=1;batch_reason=f'actual probe-size unsupported; serial fallback: {error}'
        outputs=[]
        for start in range(0,len(coefficients),batch_size):
            rows=coefficients[start:start+batch_size]
            outputs.append(self._evaluate(np.repeat(x[None],len(rows),axis=0),h,site,basis,rows))
        values=np.concatenate(outputs)
        j=((values[:rank]-values[rank:])/(2*step*scale)).reshape(rank,-1).T
        if not np.isfinite(j).all():raise SteeringUnavailable('nonfinite response measurement')
        return j,dict(probe_batch=batch_size,batching=batch_reason,probe_step=step)

    def _select_candidate(self, scores):
        """Prefer the smaller edit if every validation residual is at probe precision."""
        precise = [score for score in scores if max(score["errors"]) <= min(1e-4, self.config.tolerance)]
        if precise:
            return min(precise, key=lambda score: (score["mean_coefficient_norm"], score["error"]))
        return min(scores, key=lambda score: (score["error"], score["mean_coefficient_norm"]))

    def _target(self,x,h,base,request):
        counterfactual=None
        if request.kind.startswith('context_'):
            context=context_counterfactual(x,request)
            counterfactual=self._forecast(context[None],h)[0]
        return requested_change(base,x,self.model.quantiles,request,counterfactual_forecast=counterfactual)

    def _direct_context(self,request,h):
        if not request.kind.startswith('context_') or not self.config.include_input or self.config.context_mode=='discover':
            return False
        full=window_mask(request.forecast_window,h).all()
        if self.config.context_mode=='input_policy' and not full:
            raise ValueError('direct input policies affect the entire forecast; use context_mode=discover for partial horizons')
        return full

    def _context_result(self,x,request,h):
        """Exact operational context policy, explicitly distinct from learned control.

        Ignore=1 removes dependence on selected values by preprocessing. The
        policy's own relevance weights specify its magnitude; hidden-state
        radius does not apply to this separately identified backend.
        """
        bases=[];forecasts=[];diagnostics=[];interventions=[]
        for row in x:
            self._begin()
            base=self._forecast(row[None],h)[0]
            changed=context_counterfactual(row,request).astype(np.float32)
            forecast=self._forecast(changed[None],h)[0]
            repeated=self._forecast(changed[None],h)[0]
            if np.max(abs(repeated-forecast))/self._scale(row)>2e-5:
                raise SteeringUnavailable('predictor is stochastic; context policy requires a deterministic forecast')
            delta=forecast-base;norm=float(np.linalg.norm(delta))
            site=dict(source='context_policy',module='__input__',mode='relevance_gate')
            diagnostics.append(dict(gain=1. if norm else None,collateral=0.,relative_error=0.,achieved=True,
                zero_target=norm==0,protected_rms=0.,protected_relative_error=0.,coefficient_norm=None,radius=None,
                input_relative_norm=float(np.linalg.norm(changed-row)/max(np.linalg.norm(row-row.mean()),1e-6)),
                site=site,selection='explicit input relevance policy; no learned internal intervention',
                validation_error=None,history=[],model_calls=self.model.calls-self._start_calls,context_policy=True,
                crossing_fraction=float(np.mean(np.diff(forecast,axis=-1)<0)) if len(self.model.quantiles)>1 else None,
                baseline_crossing_fraction=float(np.mean(np.diff(base,axis=-1)<0)) if len(self.model.quantiles)>1 else None,
                caveat='target equals native forecast on gated context by definition; not evidence of internal controllability'))
            interventions.append(dict(site=site,request=asdict(request),prediction_length=h,context_length=len(row)))
            bases.append(base);forecasts.append(forecast)
        bases=np.stack(bases);forecasts=np.stack(forecasts)
        return DynamicSteeringResult(bases,forecasts,forecasts-bases,diagnostics,interventions,self.model.quantiles.copy())

    def _proposal(self,j,target):
        return trust_region_direction(j,target.flatten(),self.config.radius,ridge=self.config.ridge)[0]

    def fit(self,contexts,requests,*,prediction_length,validation_contexts=None):
        """Learn response maps and select sites using validation forecast error.

        If validation_contexts is omitted, reserve the final25% of contexts
        (at least one). Duplicate rows across fit/validation are rejected. No
        future targets or ground-truth labels are used. The returned report
        retains failed candidates and exact sampled row indices.
        """
        with self.model.lock:
            self._check_model()
            self._begin()
            x=np.asarray(contexts,dtype=np.float32)
            if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
                raise ValueError('fit requires at least two finite context rows')
            if validation_contexts is None:
                split=max(1,len(x)-max(1,len(x)//4));train,valid=x[:split],x[split:]
            else:
                train=x;valid=np.asarray(validation_contexts,dtype=np.float32)
                if valid.ndim!=2 or not len(valid) or valid.shape[1]!=x.shape[1] or not np.isfinite(valid).all():
                    raise ValueError('validation context geometry must match training')
            if set(row.tobytes() for row in train)&set(row.tobytes() for row in valid):
                raise ValueError('identical context rows cross the calibration/validation split')
            requests=[requests] if isinstance(requests, (SteeringRequest, CombinedRequest)) else list(requests)
            if not requests or not all(isinstance(r, (SteeringRequest, CombinedRequest)) for r in requests):
                raise ValueError('requests must contain SteeringRequest instances')
            ti=np.linspace(0,len(train)-1,min(len(train),self.config.calibration_samples),dtype=int)
            vi=np.linspace(0,len(valid)-1,min(len(valid),self.config.validation_samples),dtype=int)
            train,valid=train[ti],valid[vi]
            return self._fit_selected(train,valid,requests,prediction_length,
                                      dict(selection='separate calibration/validation',calibration_indices=ti.tolist(),
                                           validation_indices=(vi+split).tolist() if validation_contexts is None else vi.tolist(),
                                           validation_source='reserved input tail' if validation_contexts is None else 'provided validation_contexts'))

    def _fit_selected(self,train,valid,requests,h,selection):
        bases=[self._forecast(row[None],h)[0] for row in train]
        repeated=self._forecast(train[:1],h)[0]
        if np.max(abs(repeated-bases[0]))/self._scale(train[0])>2e-5:
            raise SteeringUnavailable('predictor is stochastic; register a deterministic forecast summary or fixed sampling stream')
        valbases=[self._forecast(row[None],h)[0] for row in valid]
        targets_by_request={request.key(h):[self._target(x,h,base,request) for x,base in zip(valid,valbases)]
                            for request in requests}
        reports={}
        discovered=[]
        for request in requests:
            if self._direct_context(request,h):
                reports[request.key(h)]=dict(request=asdict(request),selection='explicit input relevance policy; no learned internal intervention',
                    caveat='full-horizon context policy executes directly; no calibration or optimized site is needed')
            else:discovered.append(request)
        if not discovered:return reports
        requests=discovered
        candidates,total,screening=self._candidates(train[0],h)
        measurements=[];failures=[]
        for site in candidates:
            space = None
            if site.label in self._readout_rows:
                layer = self.model.module.get_submodule(site.module)
                rows = layer.weight.detach()[self._readout_rows[site.label]].float().cpu().numpy().astype(float)
                space = ReadoutSpace(rows)
                if not len(space.axes):
                    space = None
            basis = space.axes[:1] if space is not None else self._basis(site)
            try:
                # Leave room to score measured candidates on actual validation
                # forecasts instead of consuming the whole budget on probes.
                self._budget_fraction=.8
                responses=[];details=[]
                for x,base in zip(train,bases):
                    response,info=self._jacobian(x,h,site,basis,np.zeros(len(basis)),base)
                    responses.append(response);details.append(info)
                mean = np.mean(responses, axis=0)
                if space is not None:
                    expected = space.forward[:, :1]
                    gains = [float(np.sum(j * expected) / np.sum(expected**2)) for j in responses]
                    if any(g <= 0 or np.linalg.norm(j-g*expected) > .02*np.linalg.norm(j)
                           for j, g in zip(responses, gains)):
                        if site.label in self._inferred_sites:
                            # Coordinate-wise rescaling preserves the identified rows.
                            # Measure the actual response after choosing a task-specific basis.
                            mean = None
                            self._basis_sources[site.label] = "inferred forecast row space with measured local response"
                        else:
                            failures.append(dict(site=site.label, reason="Declared readout disagreed with signed probes"))
                            basis = self._basis(site)
                            space = None
                            measured = [self._jacobian(x,h,site,basis,np.zeros(len(basis)),base)
                                        for x,base in zip(train,bases)]
                            mean = np.mean([j for j, _ in measured], axis=0)
                            details = [info for _, info in measured]
                    else:
                        mean = float(np.mean(gains))
                        self._basis_sources[site.label] = "active forecast readout, calibration-target subspace"
                measurements.append((site,basis,mean,details,space))
            except _BudgetExhausted as error:
                failures.append(dict(site=site.label,reason=str(error)));break
            except (SteeringUnavailable,ValueError,RuntimeError) as error:
                failures.append(dict(site=site.label,reason=f'{type(error).__name__}: {error}'))
            finally:self._budget_fraction=1.
        for request in requests:
            targets=targets_by_request[request.key(h)]
            scored=[]
            candidates_by_label = {}
            calibration_targets = ([self._target(x,h,base,request) for x,base in zip(train,bases)]
                                   if any(m[-1] is not None for m in measurements) else [])
            for site,basis,response,details,space in measurements:
                errors=[];norms=[]
                try:
                    if space is not None:
                        basis, raw_response = space.basis(calibration_targets, self.config.rank)
                        if response is None:
                            response, measurement = self._jacobian(train[0], h, site, basis,
                                                                   np.zeros(len(basis)), bases[0])
                            details = [measurement]
                        else:
                            response = raw_response * response
                    candidates_by_label[site.label] = (site,basis,response,space is not None)
                    for x,base,target in zip(valid,valbases,targets):
                        if np.linalg.norm(target)==0:errors.append(0.);norms.append(0.);continue
                        local = self._local_readout_response(x,h,site,basis,response,base) if space is not None else response
                        z=self._proposal(local,target/self._scale(x))
                        prediction=self._evaluate(x[None],h,site,basis,z[None])[0]
                        errors.append(response_metrics(prediction-base,target)['relative_error']);norms.append(float(np.linalg.norm(z)))
                    scored.append(dict(site=site.label,error=float(np.mean(errors)),errors=errors,
                                       mean_coefficient_norm=float(np.mean(norms)),details=details))
                except _BudgetExhausted:break
                except (SteeringUnavailable,ValueError,RuntimeError) as error:
                    failures.append(dict(site=site.label,request=request.kind,reason=str(error)))
            if not scored:raise SteeringUnavailable('no candidate completed validation within the budget')
            best = self._select_candidate(scored)
            site,basis,response,local_readout=candidates_by_label[best['site']]
            report=dict(**selection,request=asdict(request),chosen=best,rank=len(basis),
                        local_readout=local_readout,
                        basis_source=self._basis_sources.get(site.label,'stored orthonormal basis'),
                        observed_internal_sites=total,searched_candidates=len(candidates),candidate_scores=scored,
                        screening=screening,
                        rejected_candidates=failures.copy(),model_calls=self.model.calls-self._start_calls,
                        elapsed_seconds=time.monotonic()-self._started,context_length=len(train[0]),prediction_length=h,
                        caveat='bounded search; best observed validation score, not global optimality or semantic disentanglement')
            key=request.key(h)
            self.profiles[key]=SteeringProfile(site,basis,response,request,h,len(train[0]),report)
            reports[key]=report
        return reports

    def steer(self,contexts,request,*,prediction_length,auto_discover=True):
        """Use one API for registered models, new tasks, horizons and context policies."""
        with self.model.lock:
            self._check_model()
            self._begin()
            x=np.asarray(contexts,dtype=np.float32)
            if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('need finite [batch,context]')
            if not isinstance(request, (SteeringRequest, CombinedRequest)):raise TypeError('request must be a SteeringRequest')
            if self._direct_context(request,prediction_length):
                return self._context_result(x,request,prediction_length)
            key=request.key(prediction_length)
            if key not in self.profiles or self.profiles[key].context_length!=x.shape[1]:
                if not auto_discover:raise SteeringUnavailable('no fitted profile for this request/horizon/context length')
                reusable=[p for p in self.profiles.values() if p.prediction_length==prediction_length and p.context_length==x.shape[1]]
                if self.config.reuse_profiles and reusable:
                    self._retarget(x[0],request,prediction_length,reusable)
                else:
                    self._fit_selected(x[:1],x[:1],[request],prediction_length,
                                       dict(selection='online same-context discovery; not held-out validation',calibration_indices=[0],validation_indices=[0]))
            profile=self.profiles[key]
            forecasts=[];baselines=[];targets=[];diagnostics=[];interventions=[]
            for context in x:
                self._begin()
                base=self._forecast(context[None],prediction_length)[0]
                target=self._target(context,prediction_length,base,request)
                actual,info,z=self._control(context,profile,base,target)
                baselines.append(base);forecasts.append(actual);targets.append(target)
                diagnostics.append(info)
                interventions.append(dict(profile_key=key,site=asdict(profile.site),coefficients=z.tolist()))
            return DynamicSteeringResult(np.stack(baselines),np.stack(forecasts),np.stack(targets),diagnostics,interventions,self.model.quantiles.copy())

    def _retarget(self,x,request,h,profiles):
        """Try learned response maps on a new dose/target before remeasuring them.

        Target-independent responses are reusable. Site selection on this
        input is explicitly online, not new held-out validation evidence.
        """
        unique={}
        for profile in profiles:
            signature=(profile.site.label,hashlib.sha256(profile.basis.tobytes()).hexdigest())
            unique[signature]=profile
        base=self._forecast(x[None],h)[0];target=self._target(x,h,base,request)
        scored=[]
        for profile in unique.values():
            response = (self._local_readout_response(x,h,profile.site,profile.basis,profile.response,base)
                        if profile.report.get("local_readout") else profile.response)
            z=self._proposal(response,target/self._scale(x))
            forecast=self._evaluate(x[None],h,profile.site,profile.basis,z[None])[0]
            error=response_metrics(forecast-base,target)['relative_error'] if np.linalg.norm(target) else 0.
            scored.append((error,float(np.linalg.norm(z)),profile))
        error,norm,profile=min(scored,key=lambda r:r[:2])
        report=dict(profile.report)
        report.update(selection='online transfer from fitted responses; not held-out validation',request=asdict(request),
            transfer_from_request=asdict(profile.request),chosen=dict(site=profile.site.label,error=error,mean_coefficient_norm=norm),
            candidate_scores=[dict(site=p.site.label,error=e,mean_coefficient_norm=n) for e,n,p in scored],
            model_calls=self.model.calls-self._start_calls,elapsed_seconds=time.monotonic()-self._started)
        self.profiles[request.key(h)]=SteeringProfile(profile.site,profile.basis,profile.response,request,h,len(x),report)

    def _local_readout_response(self, x, h, site, basis, response, base):
        """Measure this context's activation norm and inverse-normalization gain together.

        For an affine readout these multiply every row by the same scalar.
        A failed full-model direction check falls back to local finite differences.
        """
        column = int(np.argmax(np.linalg.norm(response, axis=0)))
        reference = response[:, column]
        step = min(self.config.probe_step, self.config.radius)
        probe = np.zeros((1, len(basis)))
        probe[0, column] = step
        changed = self._evaluate(x[None], h, site, basis, probe)[0]
        measured = (changed-base).ravel() / (step*self._scale(x))
        power = float(reference @ reference)
        gain = float(measured @ reference / power) if power else 0.
        if gain > 0 and np.linalg.norm(measured-gain*reference) <= .02*np.linalg.norm(measured):
            return gain*response
        return self._jacobian(x,h,site,basis,np.zeros(len(basis)),base)[0]

    def _control(self,x,profile,base,target):
        rank=len(profile.basis);z=np.zeros(rank);current=base.copy();history=[];j=profile.response
        norm=float(np.linalg.norm(target));scale=self._scale(x)
        for iteration in range(1+self.config.refinement_steps):
            if norm==0:break
            try:
                if iteration:
                    j,measurement=self._jacobian(x,profile.prediction_length,profile.site,profile.basis,z)
                elif profile.report.get("local_readout"):
                    j = self._local_readout_response(x, profile.prediction_length, profile.site,
                                                     profile.basis, profile.response, base)
                    measurement = {"method": "forecast readout calibrated on current context"}
                else:measurement={'method':'frozen mean response warm start'}
                effective=(base+target-current).flatten()/scale+j@z
                proposed,linear=trust_region_direction(j,effective,self.config.radius,ridge=self.config.ridge)
                accepted=False;old=float(np.linalg.norm(current-base-target))
                for rate in (1.,.5,.25,.125):
                    candidate=z+rate*(proposed-z)
                    prediction=self._evaluate(x[None],profile.prediction_length,profile.site,profile.basis,candidate[None])[0]
                    if np.linalg.norm(prediction-base-target)<old:
                        z=candidate;current=prediction;accepted=True;break
                error=float(np.linalg.norm(current-base-target)/norm)
                history.append(dict(iteration=iteration,accepted=accepted,relative_error=error,
                                    coefficient_norm=float(np.linalg.norm(z)),measurement=measurement,linear_solve=linear))
                if error<=self.config.tolerance:break
            except _BudgetExhausted:
                history.append(dict(iteration=iteration,accepted=False,budget_exhausted=True,
                                    relative_error=float(np.linalg.norm(current-base-target)/norm)))
                break
        metric=response_metrics(current-base,target) if norm else dict(gain=None,collateral=0.,relative_error=0.)
        terms = profile.request.terms if isinstance(profile.request, CombinedRequest) else (profile.request,)
        mask = np.logical_or.reduce([window_mask(t.forecast_window, len(base)) for t in terms])
        info=dict(**metric,achieved=metric['relative_error']<=self.config.tolerance,
                  zero_target=norm==0,budget_exhausted=any(v.get('budget_exhausted',False) for v in history),
                  protected_rms=float(np.sqrt(np.mean((current[~mask]-base[~mask])**2))) if (~mask).any() else 0.,
                  protected_relative_error=float(np.linalg.norm(current[~mask]-base[~mask])/norm) if norm else 0.,
                  coefficient_norm=float(np.linalg.norm(z)),radius=self.config.radius,
                  site=asdict(profile.site),selection=profile.report['selection'],
                  validation_error=profile.report['chosen']['error'] if profile.report['selection']=='separate calibration/validation' else None,
                  selection_error=profile.report['chosen']['error'],history=history,
                  model_calls=self.model.calls-self._start_calls,
                  crossing_fraction=float(np.mean(np.diff(current,axis=-1)<0)) if current.shape[1]>1 else None,
                  baseline_crossing_fraction=float(np.mean(np.diff(base,axis=-1)<0)) if base.shape[1]>1 else None,
                  context_policy=profile.request.kind.startswith('context_'))
        return current,info,z

    def replay(self,contexts,result):
        """Reapply saved coefficients through the model to audit actual interventions."""
        with self.model.lock:
            self._check_model()
            if np.asarray(contexts).ndim!=2 or len(contexts)!=len(result.interventions):
                raise ValueError('replay contexts must match the number of interventions')
            outputs=[]
            for x,intervention in zip(np.asarray(contexts),result.interventions):
                if intervention['site']['source']=='context_policy':
                    if len(x)!=intervention['context_length']:raise ValueError('context policy replay length changed')
                    changed=context_counterfactual(x,SteeringRequest(**intervention['request']))
                    outputs.append(self.model.forecast(changed[None],intervention['prediction_length'])[0])
                    continue
                profile=self.profiles[intervention['profile_key']]
                outputs.append(self.model.evaluate(x[None],profile.prediction_length,profile.site,profile.basis,
                                                    np.asarray(intervention['coefficients'])[None])[0])
            return np.stack(outputs)

    def save(self,path):
        """Persist steering profiles, not model weights or original training contexts."""
        self._check_model()
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        metadata=dict(version=2,model_id=self.model.model_id,fingerprint=self.model.fingerprint(),
                      quantiles=self.model.quantiles.tolist(),config=asdict(self.config),profiles=[])
        arrays={}
        for i,(key,profile) in enumerate(self.profiles.items()):
            arrays[f'basis_{i}']=profile.basis;arrays[f'response_{i}']=profile.response
            metadata['profiles'].append(dict(key=key,site=asdict(profile.site),request=asdict(profile.request),
                prediction_length=profile.prediction_length,context_length=profile.context_length,report=profile.report))
        np.savez_compressed(path/'profiles.npz',**arrays)
        metadata['arrays_sha256']=hashlib.sha256((path/'profiles.npz').read_bytes()).hexdigest()
        (path/'profiles.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path,model):
        """Restore only against the same registered model and exact accessible weights."""
        path=Path(path);metadata=json.loads((path/'profiles.json').read_text())
        if metadata['version'] not in (1,2) or metadata['fingerprint']!=model.fingerprint(legacy=metadata['version']==1):
            raise SteeringUnavailable('saved profile/model identity or weights differ; refit steering')
        if hashlib.sha256((path/'profiles.npz').read_bytes()).hexdigest()!=metadata['arrays_sha256']:
            raise ValueError('profile array checksum mismatch')
        result=cls(model,SearchConfig(**metadata['config']))
        with np.load(path/'profiles.npz',allow_pickle=False) as arrays:
            for i,row in enumerate(metadata['profiles']):
                request=request_from_dict(row['request'])
                canonical_key=request.key(row['prediction_length'])
                # Early profiles preserved e.g. period=24 rather than 24.0.
                # Compare semantics, then migrate to the canonical numeric key.
                if json.loads(canonical_key)!=json.loads(row['key']):raise ValueError('profile request key mismatch')
                basis=arrays[f'basis_{i}'];response=arrays[f'response_{i}'];site=InterventionSite(**row['site'])
                if basis.shape[1]!=site.width or response.shape!=(row['prediction_length']*len(model.quantiles),len(basis)) or not np.isfinite(basis).all() or not np.isfinite(response).all():
                    raise ValueError('invalid saved profile dimensions or nonfinite values')
                if not np.allclose(basis@basis.T,np.eye(len(basis)),atol=1e-6):raise ValueError('profile basis is not orthonormal')
                result.profiles[canonical_key]=SteeringProfile(site,basis,response,request,row['prediction_length'],row['context_length'],row['report'])
        return result
