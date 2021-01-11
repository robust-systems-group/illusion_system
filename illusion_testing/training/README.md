# Training DNNs testing for Illusion System

These scripts were used to train the DNNs under test. 

QPyTorch [1] was used to peform fixed precision training.

Torch, QPyTorch and torchvision all need to be installed.
The LSTM training is based on the script provided in ELL.

The raw data will need to be appropriately downloaded for the training scripts as they were too large to include in this repo.

[1] Zhang, T., Lin, Z., Yang, G. & De Sa, C. QPyTorch: a low-precision arithmetic simulation framework. Preprint at https://arxiv.org/abs/1910.04540 (2019).
[2] https://github.com/microsoft/ELL




The PyTorch model definitions can be found in `torch_models.py` and `KWS/keyword_spotters.py`.
