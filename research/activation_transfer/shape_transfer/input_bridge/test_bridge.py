import numpy as np
from scipy.fft import idct

from research.activation_transfer.shape_transfer.input_bridge.probe import basis, edit
from research.activation_transfer.shape_transfer.input_bridge.run import bounded, minimum, scramble


def test_public_input_directions_have_known_decoder_and_touch_only_final_32():
    axes = idct(np.eye(32), norm="ortho", axis=1)[:12]
    np.testing.assert_allclose(basis(), axes, atol=1e-15)
    x = np.arange(80., dtype=float).reshape(2, 40)
    coeff = np.zeros((2, 12))
    coeff[0, 0], coeff[1, 4] = .01*np.sqrt(32), -.08*np.sqrt(32)
    obs = edit(x, coeff, basis())-x
    np.testing.assert_array_equal(obs[:,:8], 0)
    np.testing.assert_allclose(obs[:,-32:], coeff@axes, atol=9e-15)
    np.testing.assert_allclose(np.sqrt(np.mean(obs[:,-32:]**2, axis=1)), [.01,.08])


def test_weighted_minimum_uses_each_history_and_correct_source_goal_for_both_models():
    # Non-diagonal alternating source/recipient designs; two separate histories and goals.
    rng=np.random.default_rng(8764)
    a=rng.normal(size=(2,16,12))
    g=rng.normal(size=(2,8,3))
    targets=np.concatenate((g,g),axis=1)
    pred=minimum(a,targets)
    for i in range(2):
        gram=a[i].T@a[i]
        ridge=.001*np.linalg.eigvalsh(gram)[-1]
        ref=np.linalg.solve(gram+ridge*np.eye(12),a[i].T@targets[i]).T
        np.testing.assert_allclose(pred[i],ref,atol=8e-13,rtol=5e-13)
    np.testing.assert_array_equal(minimum(np.zeros_like(a),targets),0)


def test_normalized_tail_budget_and_scramble_preserve_physical_size():
    x=np.arange(1.,25.).reshape(2,12)
    x[0]*=.002
    y,pre=bounded(x)
    np.testing.assert_allclose(pre,np.linalg.norm(x,axis=1))
    np.testing.assert_allclose(y[0],x[0])
    np.testing.assert_allclose(y[1],.25*np.sqrt(32)*x[1]/np.linalg.norm(x[1]))
    sham=scramble(x)
    wrong=np.stack([np.array([v[(j-5)%12]*(1 if j%3==0 else -1) for j in range(12)]) for v in x])
    np.testing.assert_allclose(sham,wrong)
    np.testing.assert_allclose(np.linalg.norm(bounded(sham)[0],axis=1),np.linalg.norm(y,axis=1))
