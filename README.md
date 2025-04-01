# AAAD
 The source code of auto-adversarial attack and defense （**The project will continue to be refined in future work**）

 Authors: Jialiang Sun, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

## Abstract
 Due to the urgent need of the robustness of deep neural networks (DNN),
  numerous existing open-sourced tools or platforms are developed to evaluate the robustness of DNN models by ensembling the majority of adversarial attack or defense algorithms. Unfortunately, current platforms can neither optimize the DNN architectures nor the configuration of adversarial attacks to further enhance the model robustness or the performance of adversarial attacks. To alleviate these problems, in this paper, we propose a novel platform called auto-adversarial attack and defense (A3D), which can help search for robust neural network architectures and efficient adversarial attacks.
   A3D integrates multiple neural architecture search methods to find robust architectures under different robustness evaluation metrics. 
   Besides, we provide multiple optimization algorithms to search for efficient adversarial attacks. In addition, we combine auto-adversarial attack and defense together to form a unified framework. Among auto adversarial defense, the searched efficient attack can be used as the new robustness evaluation to further enhance the robustness. In auto-adversarial attack, the searched robust architectures can be utilized as the threat model to help find stronger adversarial attacks. Experiments on CIFAR10, CIFAR100, and ImageNet datasets demonstrate the feasibility and effectiveness of the proposed platform.

arxiv: https://arxiv.org/abs/2203.03128


## Framework


<img src = 'https://github.com/Jialiang14/AAAD/blob/main/figures/AAAD_revisionv4.png?raw=true'/>

## The introduction of the core modules of A3D:
```
AAA: provide different search algorithms to search for the near-optimal attacks
optimizer_adv: provide different NAS algorithms to search for robust architectures under different evaluations
noise:the visualization of different noises
plot: the visualization of architectures
retrain: retrain the searched architectures
eval_robustness:evaluate the model robustness
```

<img src = 'https://github.com/Jialiang14/AAAD/blob/main/figures/noise.png?raw=true'/>

## Usage
a. Search for efficient attacks, e.g.,

```shell
cd AAAD/AAA/optimizer_attack/DE

python3 search.py
```

b. Search for robust architectures, e.g.,

```shell
cd AAAD/optimizer_adv/darts

python3 train_search_CAA.py
python3 train_search_Natural.py
python3 train_search_Quantific.py
python3 train_search_System.py
```

c. Retrain the searched architectures, e.g.,

```shell

cd AAAD/retrain/standard_train

python3 train.py
```

d. Evaluate the trained models, e.g.,

```shell
cd AAAD/eval_robustness

python3 robust_accuracy_Clean.py
python3 robust_accuracy_FGSM.py
python3 robust_accuracy_PGD.py
python3 robust_accuracy_Natural.py
python3 robust_accuracy_System.py
python3 robust_accuracy_Jacobian.py
```

e. Evaluate the searched attacks, e.g.,

```shell
cd AAAD/AAA/eval_attack

python3 eval_AAA.py
python3 eval_manual.py
```
## The performance of auto-adversarial attack and defense
```shell
The performance of searched architecture
```
<img src = 'https://github.com/Jialiang14/AAAD/blob/main/figures/aaa.png?raw=true'/>

```shell
The performance of searched attacks 
```

<img src = 'https://github.com/Jialiang14/AAAD/blob/main/figures/aaa2.png?raw=true'/>

### Our implementations are based on following papers:


[1] Jisoo Mok, Byunggook Na, Hyeokjun Choe, and Sungroh Yoon.
Advrush: Searching for adversarially robust neural architectures.
CoRR, abs/2108.01289, 2021.

[2] neural architecture search via proximal iterations. In Proceedings
of the AAAI Conference on Artificial Intelligence, volume 34, pages
6664–6671, 2020.

[3] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable
architecture search. arXiv preprint arXiv:1806.09055, 2018.

[4] Xiangning Chen and Cho-Jui Hsieh. Stabilizing differentiable
architecture search via perturbation-based regularization. In International conference on machine learning, pages 1554–1565. PMLR,
2020.

[5] Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, and Tie-Yan Liu.
Neural architecture optimization. Advances in neural information
processing systems, 31, 2018.

[6] Han Shi, Renjie Pi, Hang Xu, Zhenguo Li, James Kwok, and
Tong Zhang. Bridging the gap between sample-based and oneshot
neural architecture search with bonas. Advances in Neural
Information Processing Systems, 33:1808–1819, 2020.

[7] Liam Li and Ameet Talwalkar. Random search and reproducibility
for neural architecture search. In Uncertainty in artificial intelligence,
pages 367–377. PMLR, 2020.

[8] Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi,
Qi Tian, and Hongkai Xiong. Pc-darts: Partial channel connections
for memory-efficient architecture search. arXiv preprint
arXiv:1907.05737, 2019.

[9] Xiaofeng Mao, Yuefeng Chen, Shuhui Wang, Hang Su, Yuan He,
and Hui Xue. Composite adversarial attacks. In Thirty-Fifth AAAI
Conference on Artificial Intelligence, AAAI 2021, pages 8884–8892.
AAAI Press, 2021.
