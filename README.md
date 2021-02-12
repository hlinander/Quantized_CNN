# Quantized and pruned CNNS on SVHN

Train compressed (quantized and pruned) CNNs on SVHN (http://ufldl.stanford.edu/housenumbers/), as in https://arxiv.org/abs/2101.05108. Quantization is done through the Google Keras extension QKeras (https://arxiv.org/abs/2006.10159) provided at github.com/google/qkeras. pruning is done using the TensorFlow pruning AI (https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html).
A 10-fold cross validation training and validation is performed and metrics like ROC/error/accuracy use errors from cross validation. The repository also contains 

## Dependencies

For training you need: Python 3.6, TensorFlow version >= 2.1.0, Keras version: 2.2.4-tf, QKeras (https://github.com/google/qkeras).
QKeras is a sub-module of this repositry. A conda environment setup file is provided:
```
git clone https://github.com/thaarres/Quantized_CNN.git
cd Quantized_CNN/
conda env create -f environment.yml
conda activate cnns
```

## Training

To train, flags are set using absl.flags (https://abseil.io/docs/python/guides/flags). One flagfile is provided: svhn_cnn.cfg, use this, pass all parameters from commandline or create a new flag file. Specific architecture (see models.py), number of filters,kernel size, strides, loss function etc. can be set in flagfile or command line

To train use the command:

```
python train.py --flagfile=svhn_cnn.cfg #Runs normal full 10-fold training
python train.py --flagfile=svhn_cnn.cfg --epochs=1 #Runs normal full 10-fold training for 1 epoch
python train.py --flagfile=svhn_cnn.cfg --single #Runs normal training on one fold only
python train.py --flagfile=svhn_cnn.cfg --prune=True #Runs pruning
python train.py --flagfile=svhn_cnn.cfg --quantize #Runs quantization-aware training scanning bitwidths binary, ternary up to bit width 16 (mantissa quantization)
python train.py --flagfile=svhn_cnn.cfg --quantize --prune #Quantize and prune
```

Plot the average model accuracy accross the 10-folds. To reproduce plot in https://arxiv.org/abs/2101.05108, need both pruned and unpruned versioin of all models

```
python trainingDiagnostics.py -m 'models/quant_1bit;models/quant_2bit;models/quant_3bit;models/quant_4bit;models/quant_6bit;models/quant_8bit;models/quant_10bit;models/quant_12bit;models/quant_14bit;models/quant_16bit;models/latest_aq;models/full' --names 'B;T;3;4;6;8;10;12;14;16;AQ/\nAQP;BF/\nBP' --prefix 'quantized' --extra '' --pruned

```
Plot weights, calculate flops, profile weights and plot ROC with statistical uncertianties
```
python compareModels.py -m "models/full_0;models/pruned_full_0" --names "BF model;BP model" --kFold -w --doOPS
```

## Hyperparameter scan (Bayesian Optimisation)

### Optimize baseline

Hyperparameter scan using KerasTuner (https://www.tensorflow.org/tutorials/keras/keras_tuner) HyperBand and Bayesian Optimization (!WIP: some functionality has changed and this needs to be fixed (Error: attempt to get argmin of an empty sequence))

```
python kerasTuner.py 
```

### Run AutoQKeras

```
python runAutoQ.py 
```
