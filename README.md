# Testing Fourier Robustness for an Image Classification Model (PyTorch)

## Usage

CIFAR10
```
# ResNet56 (default)
python src/cifar10.py experiment=save weight=<weight_filename_in_./weights/>
```

## Output
Fourier Heat Map.
This is an error rate when each position fourier basis noise is added to image.

<img src=./fouriermap.png width=300>