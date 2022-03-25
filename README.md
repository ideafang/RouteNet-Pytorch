# RouteNet-Pytorch
Implementation of RouteNet, an effective way for Network Modeling and Optimization in SDN.

# Installation
1. Install Pytorch
2. `pip install networkx pytorch-lightning`
3. get dataset from [repo](https://github.com/BNN-UPC/NetworkModelingDatasets/tree/master/datasets_v0)

# Quick Start

```shell
python process.py
python main.py
```

# ToDo

- <del>figure out why train epoch is slow than eval...</del>
- add batch in training step and speed up training...

The code is baed on the [demo-RouteNet](https://github.com/knowledgedefinednetworking/demo-routenet) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)Project