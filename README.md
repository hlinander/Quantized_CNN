# Quantized and pruned CNNS on SVHN

Train compressed (quantized and pruned) CNNs on SVHN (http://ufldl.stanford.edu/housenumbers/), as in https://arxiv.org/abs/2101.05108. Quantization is done through the Google Keras extension QKeras (https://arxiv.org/abs/2006.10159) provided at github.com/google/qkeras. pruning is done using the TensorFlow pruning AI (https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html).
A 10-fold cross validation training and validation is performed and metrics like ROC/error/accuracy use errors from cross validation. The repository also contains 

## Dependencies

For training you need: Python 3.6, TensorFlow version >= 2.1.0, Keras version: 2.2.4-tf, QKeras (https://github.com/google/qkeras).
QKeras is a sub-module of this repositry. A conda environment setup file is provided:
```
git clone --recurse-submodules https://github.com/thaarres/Quantized_CNN.git
cd Quantized_CNN/
git submodule update --recursive
conda env create -f environment.yml
conda activate cnns
cd qkeras/
pip install --user -e . #Or python -m pip install --user -e .
cd ../
```

## Training

To train, flags are set using absl.flags (https://abseil.io/docs/python/guides/flags). One flagfile is provided: svhn_cnn.cfg, use this, pass all parameters from commandline or create a new flag file. Specific architecture (see models.py), number of filters,kernel size, strides, loss function etc. can be set in flagfile or command line

To train use the command:

```
python3 train.py --flagfile=svhn_cnn.cfg #Runs normal full 10-fold training
python train.py --flagfile=svhn_cnn.cfg --epochs=1 #Runs normal full 10-fold training for 1 epoch
python3 train.py --flagfile=svhn_cnn.cfg --single #Runs normal training on one fold only
python3 train.py --flagfile=svhn_cnn.cfg --prune=True #Runs pruning
python3 train.py --flagfile=svhn_cnn.cfg --quantize #Runs quantization-aware training scanning bitwidths binary, ternary up to bit width 16 (mantissa quantization)
python3 train.py --flagfile=svhn_cnn.cfg --quantize --prune #Quantize and prune
```

Plot the average model accuracy accross the 10-folds. To reproduce plot in https://arxiv.org/abs/2101.05108, need both pruned and unpruned versioin of all models

```
python3 trainingDiagnostics.py -m 'models/quant_1bit;models/quant_2bit;models/quant_3bit;models/quant_4bit;models/quant_6bit;models/quant_8bit;models/quant_10bit;models/quant_12bit;models/quant_14bit;models/quant_16bit;models/latest_aq;models/full' --names 'B;T;3;4;6;8;10;12;14;16;AQ/\nAQP;BF/\nBP' --prefix 'quantized' --extra '' --pruned

```
Plot weights, calculate flops, profile weights and plot ROC with statistical uncertianties
```
python compareModels.py -m "models/full_0;models/pruned_full_0" --names "BF model;BP model" --kFold -w --doOPS
```

## Hyperparameter scan (Bayesian Optimisation)

### Optimize baseline

This method depends on the kerasTuner (https://www.tensorflow.org/tutorials/keras/keras_tuner)

```
pip install -q -U keras-tuner
```
Then you can do 
```
python kerasTuner.py 
```

### Run AutoQKeras

```
python runAutoQ.py 
```
