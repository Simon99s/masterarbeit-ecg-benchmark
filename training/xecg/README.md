---
license: mit
pipeline_tag: other
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---


This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: https://github.com/dlaskalab/bench-xecg/
- Paper: [BenchECG and xECG: a benchmark and baseline for ECG foundation models](https://arxiv.org/abs/2509.10151)


To use this model, first copy to your code `xECG.py` and ensure the requirements in `requirements.txt` are inatalled and then:
```python
from xECG import xECG

model = xECG.from_pretrained("riccardolunelli/xECG_base_model_v1")
```

Important: when you pass the signal to the model ensure the leads respect the following order: `['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']`. 
For applications where less leads are available, just set them to zero.

This model returns both the set of final patches and a pooled representation.

We provide a class for ECG task with a head on the pooled representations, `downstream_models.xECGClassification`: use this for signal level class, e.g. classification, regression...

Moreover we provide another class where the head is appended to the each patch, `downstream_models.xECGFeatureClassification`: use this for task like segmentation (head output of the same size of the original patch, 25) or for heartbeat level classification.