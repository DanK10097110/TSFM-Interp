"""Runnable registration example; no model download and no architecture adapter.

The tiny random model demonstrates API mechanics, not forecasting quality.
Run: python examples/dynamic_steering/custom_model.py
"""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tsfm_lens'))
import numpy as np
import torch
from tsfm_lens.steering import SteeringModel,DynamicSteerer,SteeringRequest


def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    your_model=torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.Tanh(),torch.nn.Linear(32,16)).eval()
    def your_predictor(contexts,prediction_length):
        if prediction_length>16:raise ValueError('this demo predictor supports at most16 steps')
        with torch.no_grad():return your_model(torch.from_numpy(contexts))[:,:prediction_length].numpy()

    registered=SteeringModel(your_predictor,module=your_model,quantiles=[.5],model_id='demo-fixed-seed7-v1')
    controller=DynamicSteerer(registered)
    contexts=np.random.default_rng(7).normal(size=(12,64)).astype('float32')
    request=SteeringRequest('level',.05,forecast_window='long')
    controller.fit(contexts[:8],request,prediction_length=16,validation_contexts=contexts[8:10])
    result=controller.steer(contexts[10:],request,prediction_length=16,auto_discover=False)
    for info in result.diagnostics:
        print('achieved:',info['achieved'],'relative error:',round(info['relative_error'],5),
              'protected movement:',round(info['protected_relative_error'],5),'selected site:',info['site'])
    assert all(info['achieved'] for info in result.diagnostics)
    assert np.allclose(controller.replay(contexts[10:],result),result.forecast)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'profile';controller.save(path)
        restored=DynamicSteerer.load(path,registered)
        again=restored.steer(contexts[10:],request,prediction_length=16,auto_discover=False)
        assert np.array_equal(again.forecast,result.forecast)
    focus=SteeringRequest('context_focus',context_window=(0,16),outside_weight=0.)
    focused=controller.steer(contexts[10:],focus,prediction_length=16)
    print('Context backend:',focused.diagnostics[0]['site']['source'])
    print('Profile reload and actual-model replay passed.')


if __name__=='__main__':main()
