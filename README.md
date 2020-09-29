# Heterogeneous Molecular Graph Neural Network (HMGNN)

This is an implementation of the Heterogeneous Molecular Graph Neural Network (HMGNN) proposed in the paper:

**[Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties](https://arxiv.org/abs/2009.12710)**   
by Zeren Shui, George Karypis

To appear in ICDM 2020.

## Requirement

We implemented this software using the Deep Graph Library (DGL) with PyTorch backend. To run this code, you need

```
ase
tqdm
numpy
scipy>=1.4
pytorch>=1.4
dgl=0.4.3

```

## How to run
Since the QM9 dataset is too large to upload, we provide a script "Preprocess.py" to process the raw [QM9 dataset](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) and split the dataset to train/validation/test sets.
To run the preprocessing script you will need to download and unzip the 130k .xyz files of the QM9 dataset to a directory and run
```
python3 Preprocess.py --DATADIR=[raw data directory] --target_dir=[target directory] --evil_filename=uncharacterized.txt --cut_r=3.
```
where "uncharacterized.txt" contains the list of 3054 molecules which failed the geometry consistency check, cut_r is cut off distance. This file is also available on the QM9 website.

To train a HMGNN model to predict U0 (other available properties are mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv), run
```
python3 main.py --data_dir=[target directory] --train=1 --model_dir=[model directory] --prpty=U0 --cut_r=3.
```
Training a HMGNN model on a GPU (RTX 2070/RTX 2080/Titan V100) will cost around 3~4 days.

To test the trained model, run
```
python3 main.py --data_dir=[target directory] --train=0 --model_dir=[model directory] --prpty=U0
```

## Contact
Please contact shuix007@umn.edu or create an issue if you have any questions. We appreciate your feedbacks and comments.

## Cite
Please cite our paper if you find this model useful:

```
@misc{shui2020heterogeneous,
      title={Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties}, 
      author={Zeren Shui and George Karypis},
      year={2020},
      eprint={2009.12710},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
