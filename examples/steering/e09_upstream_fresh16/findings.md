# Upstream steering findings

{
  "reg/widen": {
    "error": 0.06293648864400642,
    "success": 0.9375,
    "lower_bound": 0.043540762005248555
  },
  "reg/season": {
    "error": 0.03323933696388885,
    "success": 0.9375,
    "lower_bound": 0.03308312413370945
  },
  "reg/upper": {
    "error": 0.2570767497950713,
    "success": 0.0,
    "lower_bound": 0.4195139834384104
  },
  "shared/widen": {
    "error": 0.04446914917032621,
    "success": 0.875,
    "lower_bound": 0.03914741231796333
  },
  "shared/season": {
    "error": 0.026455233029194745,
    "success": 1.0,
    "lower_bound": 0.027224891452679822
  },
  "shared/upper": {
    "error": 0.242367760175313,
    "success": 0.0,
    "lower_bound": 0.31967113879732045
  },
  "full/widen": {
    "error": 0.06313741179374613,
    "success": 0.8125,
    "lower_bound": 0.023040874773126874
  },
  "full/season": {
    "error": 0.03146859625393418,
    "success": 1.0,
    "lower_bound": 0.012548748814985168
  },
  "full/upper": {
    "error": 0.27089506942840846,
    "success": 0.0,
    "lower_bound": 0.21209865207669704
  }
}

Errors are relative to the entire requested quantile/time change.
Bounds apply only to the first-order response in the declared token subspace.
