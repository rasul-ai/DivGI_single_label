# DivGI

"DivGI: Delve into Digestive Endoscopy Image Classification"


## updated in 18 Feb. 2025
The training code is released with detail settings in `launch.json`.
Notably, our setting for Hyper-Kvasir, Upper-GI, and GastroVision is listed as follows: 
```
{
    "name": "hyperkvasir: ours 50epoch bit_resnext50_1",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "env": { "CUDA_VISIBLE_DEVICES": "3"},
    "args": [
        "--save_path", "hyperkvasir/ours/bit_resnext50_1",
        "--csv_train", "data/train_endo_split2.csv",
        "--data_path", "data/images",
        "--model_name", "bit_resnext50_1",
        "--lr", "0.002",
        "--do_mixup", "0.1",
        "--do_multigranularities", "true",
        "--n_step", "4",
        "--do_jigsaw", "false",
        "--do_multilabel", "true",
        "--multi_label_style", "hyper_kvasir_a",
        "--n_epochs", "50",
        "--metric", "mcc",
    ]
},
{
    "name": "uppergi: ours 50epoch bit_resnext50_1",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "env": { "CUDA_VISIBLE_DEVICES": "2"},
    "args": [
        "--save_path", "uppergi/ours/bit_resnext50_1",
        "--csv_train", "data/train_uppergi_split2.csv",
        "--data_path", "xxx/data/annotation_v00",
        "--model_name", "bit_resnext50_1",
        "--lr", "0.002",
        "--do_mixup", "0.1",
        "--do_multigranularities", "true",
        "--do_jigsaw", "false",
        "--do_multilabel", "true",
        "--multi_label_style", "upper_gi_a",
        "--n_epochs", "50",
        "--metric", "mcc",
    ]
},
{
    "name": "gastrovision: ours 50epoch bit_resnext50_1",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "env": { "CUDA_VISIBLE_DEVICES": "0"},
    "args": [
        "--save_path", "gastrovision/ours/bit_resnext50_1",
        "--csv_train", "data/train_gastrovision_split2.csv",
        "--data_path", "xxx/gastrovision/Gastrovision",
        "--model_name", "bit_resnext50_1",
        "--lr", "0.002",
        "--do_mixup", "0.1",
        "--do_multigranularities", "true",
        "--do_jigsaw", "false",
        "--do_multilabel", "true",
        "--multi_label_style", "gastrovision_tree",
        "--n_epochs", "50",
        "--metric", "mcc",
    ]
},
```

- real directory is omitted by 'xxx'.
