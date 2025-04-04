# AddingVisualModule

Here is the official implementation of our paper **[Overcoming Language Priors in VQA via Added Visual Module](https://link.springer.com/article/10.1007/s00521-022-06923-0)**. (Accepted by Neural Computing and Applications)

> Visual Question Answering (VQA) is a new and popular research direction. Dealing with language prior problems has become a hot topic in VQA in the past two years. With the development of technologies relating to VQA, related scholars have realized that in VQA tasks, the generation of answers relies too much on language priors and considers less with visual content. Some of the previous methods to alleviate language priors only focus on processing the question, while the methods to increase visual acuity only concentrate on finding the correct region. To better overcome the language prior problem of VQA, we propose a method that will improve visual content further to enhance the impact of visual content on answers. Our method consists of three parts: the base model branch, the question-only model branch, and the visual model branch. Many experiments have been carried out on the three datasets VQA-CP v1, VQA-CP v2, and VQA v2, which proves the effectiveness of our method and further improves the accuracy of the different models.
![image](https://github.com/user-attachments/assets/24988288-988f-4bf2-aac3-6ebc5328be42)

____________
This repository contains code modified from [CSS](https://github.com/yanxinzju/CSS-VQA) and [SSL](https://github.com/CrossmodalGroup/SSL-VQA), many thanks!

Most of the code content is based on the above two links, the specific changes (**key codes**) are as follows:
## 1. CSS+V

### 1.1 File description
CSS+V Combine the first k words of the question with v in the basemodel.py file to form a new prediction value 
and add it to the original prediction in the vqa_debias_loss_functions.py file

### 1.2 Run command
CSS+V
```python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode q_v_debias --debias learned_mixin --topq 1 --topv -1 --qvp 5 --output [] --seed 0
```
LMH+V
```python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode updn --debias learned_mixin --topq 1 --topv -1 --qvp 5 --output [] --seed 0
```
updn+V
```python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode updn --topq 1 --topv -1 --qvp 5 --output [] --seed 0
```

### other loss

```python
lcloss
        
        # Add the bias in
        logits = 0.7*bias + 1.3*log_probs + 3*log_probsv

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        sap = torch.cosine_similarity(log_prob, labels, dim=1)
        san = torch.cosine_similarity(bias[:, :, 0], labels, dim=1)

        # Compute loss
        loss_lc = sap / elementwise_logsumexp(sap, san)
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0) + loss_lc.mean(0)

ladv

        # Add the bias in
        logits = bias + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        b = F.softplus(bias[:, :, 0])
        p = F.softplus(log_prob)
        entropy_p = -(torch.exp(p) * p).sum(1).mean(0)
        entropy_b = -(torch.exp(b) * b).sum(1).mean(0)
        loss_entropy = entropy_p - entropy_b
        # Compute loss
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0) + 0.05*loss_entropy

lccos

        # Add the bias in
        logits = bias + log_probs

        # Renormalize to get log probabilities
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss_cos = 1 - torch.cosine_similarity(log_prob, labels, dim=1)
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0) + 3*loss_cos.mean(0)
```

### 1.4 Result

the result of CSS+V in VQA-CP v2 dataset:  
| all | yn  |other|number|  
|59.48|88.40|46.31| 52.25|
The json file is in [here](https://pan.baidu.com/s/1IrR2We3YU7jOdo0Dil9vMA) with Extraction code：f5jb 

## 2. SSL+V
### 2.1 File description
We only changed the base_model.py file 

### 2.2 Run command
```python
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 3 --ml_loss
```

### 2.3 Result
the result of SSL+V in VQA-CP v2 dataset:  
| all |   yn | other | number|  
|59.34| 88.19| 50.15 | 37.74 |

## 3 Citation

```bibtex
@article{zhao2022overcoming,
  title={Overcoming language priors in VQA via adding visual module},
  author={Zhao, Jia and Zhang, Xuesong and Wang, Xuefeng and Yang, Ying and Sun, Gang},
  journal={Neural Computing and Applications},
  volume={34},
  number={11},
  pages={9015--9023},
  year={2022},
  publisher={Springer}
}
  ```

## Acknowledgments
Our code can build upon [CSS-CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Counterfactual_Samples_Synthesizing_for_Robust_Visual_Question_Answering_CVPR_2020_paper.html) and [SSL]([https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess](https://dl.acm.org/doi/abs/10.5555/3491440.3491591)). Thanks for their great works!

