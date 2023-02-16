# MMSSL

PyTorch implementation for  WWW 2023 paper: Multi-Modal Self-Supervised Learning for Recommendation

<p align="center">
<img src="./MMSSL.png" alt="MMSSL" />
</p>


## Dependencies

- Python 3.6
- torch==1.5.0
- scikit-learn==0.24.2



## Dataset :

  ```
  ├─ MMSSL/ 
      ├── data/
      	├── baby/
        ├── sports/
        ├── tiktok/
        ├── allrecopes/
  ```


- We provide processed data at [dropbox](https://www.dropbox.com/s/qrrm94ezzr0koqg/data.zip?dl=0) 

## Usage

Start training and inference as:

```
cd MMSSL
python main.py --dataset {DATASET}
```



## Citation

If you want to use our codes in your research, please cite:

```

```

## Acknowledgement

The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE). Thank for their work.

