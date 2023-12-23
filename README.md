# Neurosymbolic-Causal-Effect-Esimator-NESTER-

After downloading the code, please follow the steps below to run the code. 

### Twins

Go to Twins->Nester and run the following command to reproduce the results of Twins dataset. One can tune the hyperparameters as given in the command below. For more hyperparameter settings, see 'argparse' section in train.py.

python -W ignore train.py --algorithm astar-near --exp_name twins --trial 1 --train_data ./ --test_data ./ --train_labels ./ --test_labels ./ --input_type "atom" --output_type "atom" --input_size 22 --output_size 1 --num_labels=0 --lossfxn "mseloss" --ite_beta 1  --batch_size 128 --symbolic_epochs 20 --max_depth 3 --neural_epochs 20


### Jobs

Go to Jobs->Nester and run the following command to reproduce the results of Twins dataset. One can tune the hyperparameters as given in the command below. For more hyperparameter settings see 'argparse' section in train.py.

python -W ignore train.py --algorithm astar-near --exp_name jobs --trial 1 --train_data ./ --test_data ./ --train_labels ./ --test_labels ./ --input_type "atom" --output_type "atom" --input_size 18 --output_size 1 --num_labels=0 --lossfxn "mseloss" --ite_beta 1  --batch_size 128 --symbolic_epochs 5 --max_depth 5 --neural_epochs 5

### Paper link: \[https://arxiv.org/pdf/2211.04370.pdf\]

If you use our work, please consider citing:
```
@article{nester, 
title={NESTER: An Adaptive Neurosymbolic Method for Causal Effect Estimation},  
journal={AAAI}, 
author={Abbavaram Gowtham Reddy, Vineeth N Balasubramanian}, 
year={2024}
}
```
