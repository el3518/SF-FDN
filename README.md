# SF-FDN
The implementation of "Source-Free Multi-Domain Adaptation with Fuzzy Rule-based Deep Neural Networks" in Python. 

Code for the TFS publication. The full paper can be found [here] https://ieeexplore.ieee.org/document/10128698

## Contribution

- We propose source-free multi-domain adaptation with fuzzy rule-based deep neural networks. To the best of
our knowledge, this is the first work adopting fuzzy rules to deal with source-free transfer learning.
- We develop an auxiliary learning mechanism to enhance the multi-domain performance of the private source models.
- We generate source anchors from source fuzzy rules to collect highly representative class features, which are employed to define an anchor-based alignment strategy to fit the pre-trained source model to the target domain while protecting the source data.
- We build a selection strategy in assistance with fuzzy outputs and nearest clustering to collect strong target samples to calculate clustering centers which we employ to predict pseudo labels with high confidence.

## Setup
Ensure that you have Python 3.7.4 and PyTorch 1.1.0

## Dataset
You can find the datasets [here](https://github.com/jindongwang/transferlearning/tree/master/data).

## Usage
Run "fuzz2lsrc-cen.py" to train source model on dataset Office-31. 
Run "fuzz2ltar-w.py" to adapt source models to the target domain.
(fuzz3lsrc-cen.py and fuzz3ltar.py for dataset Office-Home.)

## Results

| Task  | R | P  | C |  A | Avg  | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| SF-FDN  | 82.7  | 83.5  | 61.5 | 74.2 | 75.5 |


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{li2023source,
  title={Source-Free Multi-Domain Adaptation with Fuzzy Rule-based Deep Neural Networks},
  author={Li, Keqiuyin and Lu, Jie and Zuo, Hua and Zhang, Guangquan},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2023},
  publisher={IEEE}
}
