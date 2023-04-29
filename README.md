# CCRK: Improving Cross-Lingual Cross-Modal Retrieval Through 1-to-K Contrastive Learning

## Acknowledgement

About code, Our project is based on [CCLM](https://github.com/zengyan-97/CCLM).

About pretraining datasets, `zh,ja,de,fr,cs` texts in `cc3m` are translated by [UC2](https://github.com/zmykevin/UC2) while `zh,ja,de,fr,cs` texts in `sbu/coco/vg` are translated by [CCLM](https://github.com/zengyan-97/CCLM). For other languages `id,es,ru,tr`, we use `m2m_100_1.2B` model developed by [Meta AI](https://ai.facebook.com/research/) and [EasyNMT](https://github.com/UKPLab/EasyNMT) as a tool to translate all datasets from English.

Thanks for their great jobs!

## Requirements

- Install python3 environment

```shell
pip3 install -r requirements.txt
```

- Download data from corresponding websites
- If running pre-training scripts:
  - download pre-trained models for parameter initialization
    - image encoder: [swin-transformer-base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)
    - text encoder: [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
- Organize these files like this:

```
CCRK/
    data/
        xlm-roberta-large/...
        swin_base_patch4_window7_224_22k.pth
        finetune/
        	mscoco/...
        	multi30k/...
        	nlvr_en/...
    	
    iglue/
        datasets/...
    
    images/
        flickr30k-images/*.jpg
        coco/
            train2014/*.jpg
            val2014/*.jpg
            test2015/*.jpg
        image_data_train/
        	image_pixels/*.csv
        wit_test/
        	*.csv
```

## Pretrain

```shell
# CCRK 1M 6lan
python3 run.py --task "pretrain" --dist "1" --output_dir "output/CCRK-1m-6lan" --seed 42 --config configs/Pretrain_1m.yaml

# CCRK 2M 6lan
python3 run.py --task "pretrain" --dist "1" --output_dir "output/CCRK-2m-6lan" --seed 42 --config configs/Pretrain_2m.yaml --pret_para "--language_chosen zh,ja,en,de,fr,cs"

# CCRK 2M 10lan
python3 run.py --task "pretrain" --dist "1" --output_dir "output/CCRK-2m-10lan" --seed 42 --config configs/Pretrain_2m.yaml

# CCRK 3M 6lan
python3 run.py --task "pretrain" --dist "1" --output_dir "output/CCRK-3m-6lan" --seed 42 --config configs/Pretrain_3m.yaml --pret_para "--language_chosen zh,ja,en,de,fr,cs"

# CCRK 3M 10lan
python3 run.py --task "pretrain" --dist "1" --output_dir "output/CCRK-3m-10lan" --seed 42 --config configs/Pretrain_3m.yaml
```

For distributed training across nodes, see run.py for more details.

### Data

To facilitate research on multi-lingual multi-modal pre-training, we provide the text translation of [COCO+VG+SBU+CC3M](https://drive.google.com/drive/folders/1lkRMFKSdz9bXhpB0n8eELF0ztbVmcBp6?usp=share_link), which contains 10 language: zh/en/de/fr/ja/cs/id/tr/ru/es

**Please cite the corresponding papers appropriately and download the images from their websites.**

For more details, please read the code dataset/pretrain_dataset_multilingual.py (more specifically ImageMultiTextDataset) to see what format is needed.

### Checkpoints

We pretrain the model for only 30 epochs on 2 A100 GPUs. The batch size is set to 128.

|                                    Checkpoint                                     |               Dataset                |
|:---------------------------------------------------------------------------------:|:------------------------------------:|
| [CCRK_2M_10lan_30epoch](https://pan.baidu.com/s/1U4rNTD103N8T8MSSrI6bLA?pwd=cdkg) |          CC2M 10lan Version          |
| [CCRK_3M_10lan_30epoch](https://pan.baidu.com/s/1mb6JU_ZyvvkrDT5Zn69hRw?pwd=rt1r) |    CC2M+COCO+VG+SBU 10lan Version    |

Tips: 

* `All` : `CC2M+SBU+VG+COCO` , `SBU/VG/COCO` splits are borrowed from [CCLM](https://github.com/zengyan-97/CCLM).
* `10lan Version` : `zh,en,de,fr,ja,cs,id,es,ru,tr`
* `CC2M` : Because of many broken links, we only collect **1863804** images of [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) Dataset.

## Finetune

### Data: MSCOCO and Multi30K

Please download MSCOCO, Multi30K from the corresponding websites. We provide some links for reference.

- MSCOCO 

  - ja https://github.com/yahoojapan/YJCaptions
  - en https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

  - zh https://github.com/li-xirong/coco-cn

* Multi30k
  * https://github.com/multi30k/dataset

For these two datasets, you need to additionally reformulate the train json files like this:

```json
[
    {
        "caption": "A woman wearing a net on her head cutting a cake. ",
        "image": "coco/val2014/COCO_val2014_000000522418.jpg",
        "image_id": 522418
    }, ...
]
```

and the valid and test files like this:

```json
[
    {
        "image": "coco/val2014/COCO_val2014_000000391895.jpg",
        "caption": [
            "A man with a red helmet on a small moped on a dirt road. ",
            "Man riding a motor bike on a dirt road on the countryside.",
            "A man riding on the back of a motorcycle.",
            "A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ",
            "A man in a red shirt and a red hat is on a motorcycle on a hill side."
        ],
        "image_id": 391895
    }, ...
]
```

------

### Data: IGLUE

For IGLUE, you just need to clone [this repo](https://github.com/e-bug/iglue) and place it in the root path of our repo as follows. Our code works on the original annotations of IGLUE without any preprocess.

```
CCRK/
    iglue/
        datasets/...
```

For WIT, please download the `image_data_train.tar` and test images from its [kaggle](https://www.kaggle.com/c/wikipedia-image-caption/data) webpage, and extract them to `images` , `images/wit_test` seperately.

Tips for WIT:

- The download link of `image_data_train.tar` is in **Data Description**.
- You need to extract the files again in `images/image_data_train/image_pixels` and `iglue/datasets/wit/annotations/train_en.jsonl.zip`)

------

### Retrieval Tasks: Multi30K and MSCOCO

```shell
# English-only Fine-tune
## Multi30K
python3 run.py --dist 1 --task itr_multi30k --config configs/cclm-base-ft/Retrieval_multi30k_en_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th

## MSCOCO
python3 run.py --dist 1 --task itr_coco --config configs/cclm-base-ft/Retrieval_coco_en_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th


# Single-Language Fine-tune
## Multi30K, optional language: cs/de/fr
python3 run.py --dist 1 --task itr_multi30k --config configs/cclm-base-ft/Retrieval_multi30k_cs_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th

## MSCOCO, optional config: ja/zh
python3 run.py --dist 1 --task itr_coco --config configs/cclm-base-ft/Retrieval_coco_ja_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th


# All-Language Fine-tune
## Multi30K
python3 run.py --dist 1 --task itr_multi30k --config configs/cclm-base-ft/Retrieval_multi30k_all_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th

## MSCOCO
python3 run.py --dist 1 --task itr_coco --config configs/cclm-base-ft/Retrieval_coco_all_ft.yaml --output_dir output/path/to/save --bs 64 --seed 42 --epoch 10 --checkpoint pretrained_model.th
```

------

### IGLUE: Zero-Shot

We provide examples of fine-tuning on English train set and evaluating on the test sets of other languages.

```shell
# xFlickr&CO
python3 run.py --dist 1 --task xflickrco --output_dir output/path/to/save --checkpoint pretrained_model.th --bs 64 --seed 42

# WIT
python3 run.py --dist 1 --task wit --output_dir output/path/to/save --bs 80 --seed 42 --checkpoint pretrained_model.th
```

------

### IGLUE: Few-Shot

We also evaluate CCLM on IGLUE max-shot settings. **Note** that you need to finetune the pretrained model on English first, then load the checkpoints for few-shot learning.

```shell
# xFlickr&CO, optional language: de/es/id/ja/ru/tr/zh
python3 run.py --dist 1 --task xflickrco --output_dir output/path/to/save --checkpoint en_finetuned_model.th --bs 64 --seed 42 --fewshot de,100 --lr 1e-6
```

The value after language in `--fewshot` settings of xFlickr&CO is the number of few-shot samples, where we always use the maximum values.
