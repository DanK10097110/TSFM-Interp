"""Model-independent forecast requests and explicit counterfactual context policies.

Targets describe desired changes, not improved forecast accuracy. A forecast
window protects all other forecast coordinates. Context policies define what
ignore/emphasize means by constructing an observable input counterfactual;
they make no claim about directly changing attention weights.
"""

from dataclasses import asdict, dataclass, field
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d


TASKS = (
    'level', 'trend', 'detrend', 'amplitude', 'seasonality', 'deseasonalize',
    'smooth', 'roughness', 'uncertainty', 'asymmetry', 'quantile', 'anchor',
    'mean_revert', 'bounds', 'delay', 'context_ignore', 'context_emphasize',
    'context_focus', 'context_recency',
)


@dataclass(frozen=True)
class SteeringRequest:
    """One behavioral request; windows are half-open forecast/context step ranges.

    forecast_window: all/short/long or (start, stop); short/long are disjoint
    halves, with the extra odd step in long. Explicit negative indices count
    from the end. Context recent/old are likewise disjoint halves. Strength is
    in context-SD units for level/trend/seasonality/quantile; uncertainty
    and amplitude are fractional changes. Anchor/bounds values are original units.
    """
    kind: str
    strength: float = .2
    forecast_window: object = 'all'
    context_window: object = 'recent'
    period: float | None = None
    phase: float | None = None
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    quantile: float = .9
    smoothing: float = 2.
    delay: float = 1.
    outside_weight: float = .2

    def __post_init__(self):
        if self.kind not in TASKS:
            raise ValueError(f'unknown request {self.kind!r}; choose from {TASKS}')
        for name in ('strength','quantile','smoothing','delay','outside_weight'):
            if not np.isfinite(getattr(self,name)):
                raise ValueError(f'{name} must be finite')
        for name in ('period','phase','value','lower','upper'):
            if getattr(self,name) is not None and not np.isfinite(getattr(self,name)):
                raise ValueError(f'{name} must be finite when supplied')
        if self.smoothing <= 0 or self.outside_weight < 0:
            raise ValueError('smoothing must be positive; outside_weight nonnegative')
        if self.kind in ('seasonality','deseasonalize') and (self.period is None or self.period <= 2):
            raise ValueError('seasonal requests require period > 2')
        if self.kind in ('amplitude','uncertainty') and self.strength <= -1:
            raise ValueError('fractional changes must be greater than -1')
        if self.kind in ('detrend','deseasonalize','smooth','mean_revert','anchor','context_ignore') and not 0 <= self.strength <= 1:
            raise ValueError('this request uses a strength between zero and one')
        if self.kind == 'context_emphasize' and self.strength < 0:
            raise ValueError('context_emphasize needs nonnegative strength')
        if self.kind == 'anchor' and self.value is None:
            raise ValueError('anchor requires a value in original series units')
        if self.kind == 'bounds' and (self.lower is None and self.upper is None):
            raise ValueError('bounds requires lower and/or upper')
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError('lower must not exceed upper')
        for name in ('strength','quantile','smoothing','delay','outside_weight','period','phase','value','lower','upper'):
            if getattr(self,name) is not None:object.__setattr__(self,name,float(getattr(self,name)))
        for name in ('forecast_window','context_window'):
            value=getattr(self,name)
            if isinstance(value,(list,tuple)) and all(isinstance(v,(int,np.integer)) for v in value):
                object.__setattr__(self,name,tuple(int(v) for v in value))

    def key(self, prediction_length):
        """Bind learned policies to the exact behavioral request and horizon."""
        return json.dumps(dict(request=asdict(self),prediction_length=int(prediction_length)),sort_keys=True)


@dataclass(frozen=True)
class CombinedRequest:
    """Sum forecast changes relative to one baseline and fit one shared edit."""
    terms: tuple[SteeringRequest, ...]
    kind: str = field(default="combined", init=False)

    def __post_init__(self):
        terms = tuple(self.terms)
        if len(terms) < 2 or any(not isinstance(t, SteeringRequest) or t.kind.startswith("context_") for t in terms):
            raise ValueError("combined steering needs at least two forecast requests")
        object.__setattr__(self, "terms", terms)

    def key(self, prediction_length):
        return json.dumps(dict(request=asdict(self), prediction_length=int(prediction_length)), sort_keys=True)


def request_from_dict(item):
    if not isinstance(item, dict):
        raise ValueError("each steering request must be a mapping")
    if item.get("kind") == "combined":
        if set(item) != {"kind", "terms"}:
            raise ValueError("combined requests take a list of terms, each with its own settings")
        terms = item["terms"]
        if not isinstance(terms, (list, tuple)) or not all(isinstance(t, dict) for t in terms):
            raise ValueError("combined terms must be a list of forecast requests")
        return CombinedRequest(tuple(SteeringRequest(**term) for term in terms))
    return SteeringRequest(**item)


def window_mask(window, length, *, context=False):
    """Resolve explicit or named half-open windows without silently clipping typos."""
    if length < 1:
        raise ValueError('window length must be positive')
    if isinstance(window,str):
        names={'all':(0,length),'short':(0,max(1,length//2)),'long':(length//2,length)}
        if context:
            names.update(old=(0,max(1,length//2)),recent=(length//2,length))
        if window not in names:
            raise ValueError(f'unknown window {window!r}')
        start,stop=names[window]
    else:
        if not isinstance(window,(list,tuple)) or len(window)!=2:
            raise ValueError('window must be a name or (start,stop)')
        start,stop=window
        if not isinstance(start,(int,np.integer)) or not isinstance(stop,(int,np.integer)):
            raise ValueError('window boundaries must be integer steps')
        start=start+length if start<0 else start
        stop=stop+length if stop<0 else stop
        if not 0 <= start < stop <= length:
            raise ValueError('window is empty or outside the available steps')
    result=np.zeros(length,dtype=bool);result[start:stop]=True
    return result


def context_counterfactual(context, request):
    """Apply a documented relevance gate using references outside ignored regions.

    ignore=1 replaces selected values by interpolation from unselected values;
    this reference is independent of the ignored window. focus retains the
    selected window and shrinks the rest toward its mean. Emphasis extrapolates
    selected deviations from the outside interpolation. These are input policies,
    not claims about learned causal semantics or attention weights.
    """
    x=np.asarray(context,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all():
        raise ValueError('context must be a finite vector')
    selected=window_mask(request.context_window,len(x),context=True)
    if request.kind=='context_focus':
        reference=np.full(len(x),float(x[selected].mean()))
        weight=np.where(selected,1.,request.outside_weight)
    elif request.kind=='context_recency':
        reference=np.full(len(x),float(x.mean()))
        weight=np.exp(np.clip(request.strength*np.linspace(-1.,0.,len(x)),-20,20))
    else:
        outside=np.flatnonzero(~selected)
        if not len(outside):
            raise ValueError('ignore/emphasize needs at least one outside context value')
        reference=np.interp(np.arange(len(x)),outside,x[outside])
        if request.kind=='context_ignore':
            weight=np.where(selected,1.-request.strength,1.)
        elif request.kind=='context_emphasize':
            weight=np.where(selected,1.+request.strength,1.)
        else:
            raise ValueError('not a context-policy request')
    return reference+weight*(x-reference)


def requested_change(baseline, context, quantiles, request, *, counterfactual_forecast=None):
    """Compile 19 kinds into a complete [horizon,channels] desired change.

    Zero entries outside the forecast window remain in the optimizer objective.
    Point-only forecasts support all requests except uncertainty/asymmetry and
    unavailable quantiles. No future observations are used.
    """
    if isinstance(request, CombinedRequest):
        return sum(requested_change(baseline, context, quantiles, term) for term in request.terms)
    y=np.asarray(baseline,dtype=float);x=np.asarray(context,dtype=float)
    q=np.asarray(quantiles,dtype=float)
    if y.ndim!=2 or y.shape[1]!=len(q) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError('need finite forecast [horizon,channels] and context')
    median=int(np.argmin(abs(q-.5)))
    if abs(q[median]-.5)>1e-7:
        raise ValueError('forecast must include a median/point channel at 0.5')
    m=y[:,median];h=len(m);t=np.linspace(-.5,.5,h)
    scale=max(float(np.std(x)),1e-6)
    strength=request.strength;kind=request.kind
    add=np.zeros(h);delta=None
    if kind=='level':add[:]=strength*scale
    elif kind=='trend':add=strength*scale*t
    elif kind=='detrend':
        slope=np.dot(t,m-m.mean())/max(np.dot(t,t),1e-12);add=-strength*slope*t
    elif kind=='amplitude':add=strength*(m-m.mean())
    elif kind=='seasonality':
        tc=np.arange(len(x));tf=np.arange(len(x),len(x)+h)
        if request.phase is None:
            design=np.column_stack([np.ones(len(x)),tc-tc.mean(),np.sin(2*np.pi*tc/request.period),np.cos(2*np.pi*tc/request.period)])
            coeff=np.linalg.lstsq(design,x,rcond=None)[0][-2:]
            if np.linalg.norm(coeff)<1e-10*max(1.,scale):
                raise ValueError('no resolvable context harmonic; supply an explicit phase')
            phase=np.arctan2(coeff[1],coeff[0])
        else:phase=request.phase
        add=strength*scale*np.sin(2*np.pi*tf/request.period+phase)
    elif kind=='deseasonalize':
        tf=np.arange(h);wave=np.column_stack([np.sin(2*np.pi*tf/request.period),np.cos(2*np.pi*tf/request.period)])
        design=np.column_stack([np.ones(h),t,wave]);coef=np.linalg.lstsq(design,m,rcond=None)[0]
        add=-strength*(wave@coef[-2:])
    elif kind=='smooth':add=strength*(gaussian_filter1d(m,request.smoothing)-m)
    elif kind=='roughness':add=strength*(m-gaussian_filter1d(m,request.smoothing))
    elif kind=='uncertainty':
        if len(q)<3:raise ValueError('uncertainty steering requires quantile outputs')
        delta=strength*(y-m[:,None])
    elif kind=='asymmetry':
        if len(q)<3:raise ValueError('asymmetry steering requires quantile outputs')
        delta=abs(y-m[:,None])*strength
    elif kind=='quantile':
        matches=np.flatnonzero(np.isclose(q,request.quantile,atol=1e-7,rtol=0))
        if len(matches)!=1:raise ValueError('requested quantile is not available')
        delta=np.zeros_like(y);delta[:,matches[0]]=strength*scale
    elif kind=='anchor':add=strength*(request.value-m)
    elif kind=='mean_revert':
        reference=float(x.mean()) if request.value is None else request.value
        add=strength*np.linspace(0,1,h)*(reference-m)
    elif kind=='bounds':add=np.clip(m,request.lower,request.upper)-m
    elif kind=='delay':add=np.interp(np.arange(h)-request.delay,np.arange(h),m)-m
    elif kind.startswith('context_'):
        if counterfactual_forecast is None or np.shape(counterfactual_forecast)!=y.shape:
            raise ValueError('context requests require a matching counterfactual forecast')
        delta=np.asarray(counterfactual_forecast,dtype=float)-y
    else:raise ValueError('unknown steering kind')
    if delta is None:delta=np.repeat(add[:,None],len(q),axis=1)
    delta=np.array(delta,copy=True)
    delta[~window_mask(request.forecast_window,h)]=0.
    return delta
