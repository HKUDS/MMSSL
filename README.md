# MMSSL: Multi-Modal Self-Supervised Learning for Recommendation

PyTorch implementation for WWW 2023 paper [Multi-Modal Self-Supervised Learning for Recommendation](https://arxiv.org/pdf/2302.10632.pdf).

<p align="center">
<img src="./MMSSL.png" alt="MMSSL" />
</p>

MMSSL is a new multimedia recommender system which integrates the generative modality-aware collaborative self-augmentation and the contrastive cross-modality dependency encoding. It achieves better performance than existing SOTA multi-model recommenders.


<h2>Dependencies </h2>

* Python >= 3.9.13
* [Pytorch](https://pytorch.org/) >= 1.13.0+cu116
* [dgl-cuda11.6](https://www.dgl.ai/) >= 0.9.1post1




<h2>Usage </h2>

Start training and inference as:

```
cd MMSSL
python ./main.py --dataset {DATASET}
```
Supported datasets:  `Amazon-Baby`, `Amazon-Sports`, `Tiktok`, `Allrecipes`


<h2> Datasets </h2>

  ```
  ├─ MMSSL/ 
      ├── data/
        ├── tiktok/
        ...
  ```
  |    Dataset   |   |  Amazon  |      |   |          |      |   |  Tiktok  |     |     |   | Allrecipes |    |
|:------------:|:-:|:--------:|:----:|:-:|:--------:|:----:|:-:|:--------:|:---:|:---:|:-:|:----------:|:--:|
|   Modality   |   |     V    |   T  |   |     V    |   T  |   |     V    |  A  |  T  |   |      V     |  T |
|   Embed Dim  |   |   4096   | 1024 |   |   4096   | 1024 |   |    128   | 128 | 768 |   |    2048    | 20 |
|     User     |   |   35598  |      |   |   19445  |      |   |   9319   |     |     |   |    19805   |    |
|     Item     |   |   18357  |      |   |   7050   |      |   |   6710   |     |     |   |    10067   |    |
| Interactions |   |  256308  |      |   |  139110  |      |   |   59541  |     |     |   |    58922   |    |
|   Sparsity   |   | 99.961\% |      |   | 99.899\% |      |   | 99.904\% |     |     |   |  99.970\%  |    |


- `2023.11.1 new multi-modal datastes uploaded`: 📢📢 🌹🌹 We provide new multi-modal datasets `Netflix` and `MovieLens`  (i.e., CF training data, multi-modal data including `item text` and `posters`) of new multi-modal work [LLMRec](https://github.com/HKUDS/LLMRec) on Google Drive. 🌹We hope to contribute to our community and facilitate your research~

- `2023.3.23 update(all datasets uploaded)`: We provide the processed data at [Google Drive](https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0?usp=share_link). 
- `2023.3.24 update`: The official website of the `Tiktok` dataset has been closed. Thus, we also provide many other versions of preprocessed [Tiktok](https://drive.google.com/drive/folders/1hLvoS7F0R_K0HBixuS_OVXw_WbBxnshF?usp=share_link).  We spent a lot of time pre-processing this dataset, so if you want to use our preprocessed Tiktok in your work please cite.

🚀🚀 The provided dataset is compatible with multi-modal recommender models such as [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), and [MICRO](https://github.com/CRIPAC-DIG/MICRO) and requires no additional data preprocessing, including (1) basic user-item interactions and (2) multi-modal features.

```
# part of data preprocessing
# #----json2mat--------------------------------------------------------------------------------------------------
import json
from scipy.sparse import csr_matrix
import pickle
import numpy as np
n_user, n_item = 39387, 23033
f = open('/home/weiw/Code/MM/MMSSL/data/clothing/train.json', 'r')  
train = json.load(f)
row, col = [], []
for index, value in enumerate(train.keys()):
    for i in range(len(train[value])):
        row.append(int(value))
        col.append(train[value][i])
data = np.ones(len(row))
train_mat = csr_matrix((data, (row, col)), shape=(n_user, n_item))
pickle.dump(train_mat, open('./train_mat', 'wb'))  
# # ----json2mat--------------------------------------------------------------------------------------------------


# ----mat2json--------------------------------------------------------------------------------------------------
# train_mat = pickle.load(open('./train_mat', 'rb'))
test_mat = pickle.load(open('./test_mat', 'rb'))
# val_mat = pickle.load(open('./val_mat', 'rb'))

# total_mat = train_mat + test_mat + val_mat
total_mat =test_mat

# total_mat = pickle.load(open('./new_mat','rb'))
# total_mat = pickle.load(open('./new_mat','rb'))
total_array = total_mat.toarray()
total_dict = {}

for i in range(total_array.shape[0]):
    total_dict[str(i)] = [index for index, value in enumerate(total_array[i]) if value!=0]

new_total_dict = {}

for i in range(len(total_dict)):
    # if len(total_dict[str(i)])>1:
    new_total_dict[str(i)]=total_dict[str(i)]

# train_dict, test_dict = {}, {}

# for i in range(len(new_total_dict)):
#     train_dict[str(i)] = total_dict[str(i)][:-1]
#     test_dict[str(i)] = [total_dict[str(i)][-1]]

# train_json_str = json.dumps(train_dict)
test_json_str = json.dumps(new_total_dict)

# with open('./new_train.json', 'w') as json_file:
# # with open('./new_train_json', 'w') as json_file:
#     json_file.write(train_json_str)
with open('./test.json', 'w') as test_file:
# with open('./new_test_json', 'w') as test_file:
    test_file.write(test_json_str)
# ----mat2json--------------------------------------------------------------------------------------------------
```


<h3> Multi-modal Datasets </h3>
🌹🌹 Please cite our paper if you use the 'netflix' dataset~ ❤️  

We collected a multi-modal dataset using the original [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) released on the [Kaggle](https://www.kaggle.com/) website. The data format is directly compatible with state-of-the-art multi-modal recommendation models like [LLMRec](https://github.com/HKUDS/LLMRec), [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO), and others, without requiring any additional data preprocessing.

 `Textual Modality:` We have released the item information curated from the original dataset in the "item_attribute.csv" file. Additionally, we have incorporated textual information enhanced by LLM into the "augmented_item_attribute_agg.csv" file. (The following three images represent (1) information about Netflix as described on the Kaggle website, (2) textual information from the original Netflix Prize Data, and (3) textual information augmented by LLMs.)
<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/textual_data1.png" alt="Image 1" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data2.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data'.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data3.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>LLM-augmented textual data.</figcaption> -->
  </figure>  
</div>
 
 `Visual Modality:` We have released the visual information obtained from web crawling in the "Netflix_Posters" folder. (The following image displays the poster acquired by web crawling using item information from the Netflix Prize Data.)
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/visiual_data1.png" alt="Image 1" style="width:690px;height:590px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>
 

<h3> Original Multi-modal Datasets & Augmented Datasets </h3>
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/datasets.png" alt="Image 1" style="width:480px;height:270px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>


<br>
<p>

<h3> Download the Netflix dataset. </h3>
🚀🚀
We provide the processed data (i.e., CF training data & basic user-item interactions, original multi-modal data including images and text of items, encoded visual/textual features and LLM-augmented text/embeddings).  🌹 We hope to contribute to our community and facilitate your research 🚀🚀 ~

- `netflix`: [Google Drive Netflix](https://drive.google.com/drive/folders/1BGKm3nO4xzhyi_mpKJWcfxgi3sQ2j_Ec?usp=drive_link).  [🌟(Image&Text)](https://drive.google.com/file/d/1euAnMYD1JBPflx0M86O2M9OsbBSfrzPK/view?usp=drive_link)



<h3> Encoding the Multi-modal Content. </h3>

We use [CLIP-ViT](https://huggingface.co/openai/clip-vit-base-patch32) and [Sentence-BERT](https://www.sbert.net/) separately as encoders for visual side information and textual side information.






<h2> Experimental Results </h2>

Performance comparison of baselines on different datasets in terms of Recall@20, Precision@20 and NDCG@20:

|    Baseline    |        Tiktok        |                      |                      |           |      Amazon-Baby     |                      |                      |           |     Amazon-Sports    |                      |                      |           |      Allrecipes      |                      |                      |
|:--------------:|:--------------------:|:--------------------:|:--------------------:|-----------|:--------------------:|:--------------------:|:--------------------:|-----------|:--------------------:|:--------------------:|:--------------------:|-----------|:--------------------:|:--------------------:|:--------------------:|
|                |         R@20         |         P@20         |         N@20         |           |         R@20         |         P@20         |         N@20         |           |         R@20         |         P@20         |         N@20         |           |         R@20         |         P@20         |         N@20         |
|     MF-BPR     |        0.0346        |        0.0017        |        0.0130        |           |        0.0440        |        0.0024        |        0.0200        |           |        0.0430        |        0.0023        |        0.0202        |           |        0.0137        |        0.0007        |        0.0053        |
|      NGCF      |        0.0604        |        0.0030        |        0.0238        |           |        0.0591        |        0.0032        |        0.0261        |           |        0.0695        |        0.0037        |        0.0318        |           |        0.0165        |        0.0008        |        0.0059        |
|    LightGCN    |        0.0653        |        0.0033        |        0.0282        |           |        0.0698        |        0.0037        |        0.0319        |           |        0.0782        |        0.0042        |        0.0369        |           |        0.0212        |        0.0010        |        0.0076        |
|       SGL      |        0.0603        |        0.0030        |        0.0238        |           |        0.0678        |        0.0036        |        0.0296        |           |        0.0779        |        0.0041        |        0.0361        |           |        0.0191        |        0.0010        |        0.0069        |
|       NCL      |        0.0658        |        0.0034        |        0.0269        |           |        0.0703        |        0.0038        |        0.0311        |           |        0.0765        |        0.0040        |        0.0349        |           |        0.0224        |        0.0010        |        0.0077        |
|      HCCF      |        0.0662        |        0.0029        |        0.0267        |           |        0.0705        |        0.0037        |        0.0308        |           |        0.0779        |        0.0041        |        0.0361        |           |        0.0225        |        0.0011        |        0.0082        |
|      VBPR      |        0.0380        |        0.0018        |        0.0134        |           |        0.0486        |        0.0026        |        0.0213        |           |        0.0582        |        0.0031        |        0.0265        |           |        0.0159        |        0.0008        |        0.0056        |
|  LightGCN-$M$  |        0.0679        |        0.0034        |        0.0273        |           |        0.0726        |        0.0038        |        0.0329        |           |        0.0705        |        0.0035        |        0.0324        |           |        0.0235        |        0.0011        |        0.0081        |
|      MMGCN     |        0.0730        |        0.0036        |        0.0307        |           |        0.0640        |        0.0032        |        0.0284        |           |        0.0638        |        0.0034        |        0.0279        |           |        0.0261        |        0.0013        |        0.0101        |
|      GRCN      |        0.0804        |        0.0036        |        0.0350        |           |        0.0754        |        0.0040        |        0.0336        |           |        0.0833        |        0.0044        |        0.0377        |           |        0.0299        |        0.0015        |        0.0110        |
|     LATTICE    |        0.0843        |        0.0042        |  0.0367  |           |  0.0829  |  0.0044  |  0.0368  |           |  0.0915  |  0.0048  |  0.0424  |           |        0.0268        |        0.0014        |        0.0103        |
|     CLCRec     |        0.0621        |        0.0032        |        0.0264        |           |        0.0610        |        0.0032        |        0.0284        |           |        0.0651        |        0.0035        |        0.0301        |           |        0.0231        |        0.0010        |        0.0093        |
|      MMGCL     |        0.0799        |        0.0037        |        0.0326        |           |        0.0758        |        0.0041        |        0.0331        |           |        0.0875        |        0.0046        |        0.0409        |           |        0.0272        |        0.0014        |        0.0102        |
|     SLMRec     |  0.0845  |  0.0042  |        0.0353        |           |        0.0765        |        0.0043        |        0.0325        |           |        0.0829        |        0.0043        |        0.0376        |           |  0.0317  |  0.0016  |  0.0118  |
|     MMSSL    |    0.0921   |    0.0046   |    0.0392   |           |   0.0962   |    0.0051   |    0.0422   |           |    0.0998   |    0.0052   |    0.0470   |           |    0.0367   |   0.0018   |    0.0135   |
| p-value | 1.28e-5 | 7.12e-6 | 6.55e-6 |  | 2.23e-6 | 7.69e-6 | 8.65e-7 |  | 7.75e-6 | 6.48e-6 | 6.78e-7 |           | 3.94e-4 | 5.06e-6 | 4.31e-5 |
|     Improv.    |        8.99%        |        9.52%        |        6.81%        |           |        16.04%       |        15.91%       |        14.67%       |  |        9.07%        |        8.33%        |        10.85%       |  |        15.77%       |        12.50%       |        14.40%       |



<h1> Citing </h1>

If you find this work helpful to your research, please kindly consider citing our paper.


```
@inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={790--800},
  year={2023}
}
```
<!-- or -->

<!-- @inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the Web Conference (WWW)},
  year={2023}
}
 -->


## Acknowledgement

## Acknowledgement

The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO). Thank them for their work.

