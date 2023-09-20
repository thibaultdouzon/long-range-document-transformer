# long-range-document-transformer

## Models

Pre-trained models are available [here](https://drive.google.com/drive/folders/11A0-LVNaWPtmydXjCe70EICEfMC1kn6M?usp=drive_link).
For data privacy reasons (models were pre-trained with MLM task on private data), classification heads are removed from the models but the encoder remains.

### LayoutLM

LayoutLM[^1] was pre-trained in 3 flavours with a maximum sequence length of 518 tokens
The flavours differentiate by the 2D relative attention bias applied to the input.
These versions are referred to as SplitPage in the paper.

### Linformer

Linformer[^2] was pre-trained with 2048 sequence length.

### Cosformer

Cosformer[^3] was pre-trained in 3 flavours with a maximum sequence length of 2048 tokens.
The 2D relative attention biases are similar to those used for layoutLM but not exactly identical.

Cosformer is **not** compatible with fp16 inference or training. More investigation is needed to evaluate its compatibility with bf16.

[^1]: <https://arxiv.org/abs/1912.13318>

```code
@inproceedings{Xu_2020,
    doi = {10.1145/3394486.3403172},
    url = {https://doi.org/10.1145%2F3394486.3403172},
    year = 2020,
    month = {aug},
    publisher = {{ACM}},
    author = {Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    title = {{LayoutLM}: Pre-training of Text and Layout for Document    Image Understanding},
    booktitle = {Proceedings of the 26th {ACM} {SIGKDD} International Conference on Knowledge Discovery {\&}amp$\mathsemicolon$ Data Mining}
}
```

[^2]: <https://arxiv.org/abs/2006.04768>

```code
@misc{wang2020linformer,
    title={Linformer: Self-Attention with Linear Complexity}, 
    author={Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
    year={2020},
    eprint={2006.04768},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

[^3]: <https://arxiv.org/abs/2202.08791>

```code
@misc{qin2022cosformer,
    title={cosFormer: Rethinking Softmax in Attention}, 
    author={Zhen Qin and Weixuan Sun and Hui Deng and Dongxu Li and Yunshen Wei and Baohong Lv and Junjie Yan and Lingpeng Kong and Yiran Zhong},
    year={2022},
    eprint={2202.08791},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Example

Example comming soon :-)

## Cite this work

```code
@incollection{Douzon_2023,
    doi = {10.1007/978-3-031-41501-2_4},
    url = {https://doi.org/10.1007%2F978-3-031-41501-2_4},
    year = 2023,
    publisher = {Springer Nature Switzerland},
    pages = {47--64},
    author = {Thibault Douzon and Stefan Duffner and Christophe    Garcia and J{\'{e}}r{\'{e}}my Espinas},
    title = {Long-Range Transformer Architectures for~Document    Understanding},
    booktitle = {Document Analysis and Recognition {\textendash} {ICDAR} 2023 Workshops}
}
```
