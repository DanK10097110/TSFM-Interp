# Upstream steering findings

{
  "reg/widen": {
    "error": 0.06249883215409434,
    "success": 0.75,
    "lower_bound": 0.05482264028092486
  },
  "reg/season": {
    "error": 0.04787768848681566,
    "success": 1.0,
    "lower_bound": 0.06481230957511333
  },
  "reg/upper": {
    "error": 0.21464526511088217,
    "success": 0.0,
    "lower_bound": 0.34820696129351647
  },
  "shared/widen": {
    "error": 0.04944324519941623,
    "success": 1.0,
    "lower_bound": 0.046839269588133794
  },
  "shared/season": {
    "error": 0.055391661732003786,
    "success": 0.75,
    "lower_bound": 0.05319125612169551
  },
  "shared/upper": {
    "error": 0.25129245135807504,
    "success": 0.0,
    "lower_bound": 0.2757043542673796
  },
  "full/widen": {
    "error": 0.08628217182474288,
    "success": 0.75,
    "lower_bound": 0.03043320405154213
  },
  "full/season": {
    "error": 0.04530100077292547,
    "success": 1.0,
    "lower_bound": 0.03368320820270083
  },
  "full/upper": {
    "error": 0.2069713488530237,
    "success": 0.0,
    "lower_bound": 0.19417224975469347
  }
}

Errors are relative to the entire requested quantile/time change.
Bounds apply only to the first-order response in the declared token subspace.
