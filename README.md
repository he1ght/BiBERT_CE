This is the repository for our paper "[Impact of Sentence Representation Matching in Neural Machine Translation](https://www.mdpi.com/2076-3417/12/3/1313)" and the original BiBERT Paper "[BERT, mBERT, or BiBERT? A Study on Contextualized Embeddings for Neural Machine Translation](https://arxiv.org/abs/2109.04588)".
```
@article{jung2022impact,
  title={Impact of Sentence Representation Matching in Neural Machine Translation},
  author={Jung, Heeseung and Kim, Kangil and Shin, Jong-Hun and Na, Seung-Hoon and Jung, Sangkeun and Woo, Sangmin},
  journal={Applied Sciences},
  volume={12},
  number={3},
  pages={1313},
  year={2022},
  publisher={MDPI}
  abstract = "Most neural machine translation models are implemented as a conditional language model framework composed of encoder and decoder models. This framework learns complex and long-distant dependencies, but its deep structure causes inefficiency in training. Matching vector representations of source and target sentences improves the inefficiency by shortening the depth from parameters to costs and generalizes NMTs with different perspective to cross-entropy loss. In this paper, we propose matching methods to derive the cost based on constant word embedding vectors of source and target sentences. To find the best method, we analyze impact of the methods with varying structures, distance metrics, and model capacity in a French to English translation task. An optimally configured method is applied to English from and to French, Spanish, and German translation tasks. In the tasks, the method showed performance improvement by 3.23 BLEU in maximum, 0.71 in average. We evaluated the robustness of this method to various embedding distributions and models as conventional gated structures and transformer network, and empirical results showed that it has higher chance to improve performance in those variety."
}
```

```
@inproceedings{xu-etal-2021-bert,
    title = "{BERT}, m{BERT}, or {B}i{BERT}? A Study on Contextualized Embeddings for Neural Machine Translation",
    author = "Xu, Haoran  and
      Van Durme, Benjamin  and
      Murray, Kenton",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.534",
    pages = "6663--6675",
    abstract = "The success of bidirectional encoders using masked language models, such as BERT, on numerous natural language processing tasks has prompted researchers to attempt to incorporate these pre-trained models into neural machine translation (NMT) systems. However, proposed methods for incorporating pre-trained models are non-trivial and mainly focus on BERT, which lacks a comparison of the impact that other pre-trained models may have on translation performance. In this paper, we demonstrate that simply using the output (contextualized embeddings) of a tailored and suitable bilingual pre-trained language model (dubbed BiBERT) as the input of the NMT encoder achieves state-of-the-art translation performance. Moreover, we also propose a stochastic layer selection approach and a concept of a dual-directional translation model to ensure the sufficient utilization of contextualized embeddings. In the case of without using back translation, our best models achieve BLEU scores of 30.45 for En→De and 38.61 for De→En on the IWSLT{'}14 dataset, and 31.26 for En→De and 34.94 for De→En on the WMT{'}14 dataset, which exceeds all published numbers.",
}
```
## Prerequisites
```
conda create -n bibert python=3.7
conda activate bibert
```
* [transformers](https://github.com/huggingface/transformers) >= 4.4.2
  ```
  pip install transformers
  ```
* Install our fairseq repo
  ```
  cd BiBERT
  pip install --editable ./
  ```
* [hydra](https://github.com/facebookresearch/hydra) = 1.0.3
  ```
  pip install hydra-core==1.0.3
  ```

### Training
The way to train a BiBERT model for translation is same with [BiBERT](https://github.com/fe1ixxu/BiBERT). Note that use `--concept_equalization` to use our proposed matching method in training. The method is only worked on the training session.
