 # A prefix and attention map discrimination fusion guided attention for biomedical named entity recognition.

 
## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dependencies

```
- numpy (1.21.4)
- torch (1.10.0)
- genism (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```

## 3. Dataset


The dataset is available on https://github.com/cambridgeltl/MTL-Bioinformatics-2016.

You should process the data like '/media/gyz/dc602a7c-be57-400f-a4ad-f22544ea1319/gzyCode/PAMDFGA/data/ncbi/train.json'


## 5. Training

```bash
>> python main.py --config ./config/ncbi.json
```
## 6. Acknowledgements
Thanks the code of Li et al. (https://arxiv.org/pdf/2112.10070.pdf)



