# Translate data to more language

For the large translation model m2m-100-lg, we have used a batch size of 16 on NVIDIA A100 with 11.2GB of memory.

For more usage, please see the code in `m2m_translate.py`

## Test Translation Model

```shell
python3 data_process.py --test --text "我爱你"
```

## Translate One File

```shell
# example
python3 m2m_translate.py --from_path /filepath/to/be/translated --to_path /filepath/to/save --device 'cuda:0' --bs 16 --trans_lanls ko,sw
```

## Translate Directory

Take our 10lan datasets as a example.Please download pretraining datasets and organise the file like this:

```
cc2m-10lan/*.data
vg-10lan/*.data
coco-10lan/*.data
sbu-10lan/*.data
```

Then execute the script below:

```shell
python3 m2m_translate.py --from_path cc2m-10lan/ --to_path cc2m-12lan/ --device 'cuda:0' --bs 16 --trans_lanls ko,sw
python3 m2m_translate.py --from_path vg-10lan/ --to_path vg-12lan/ --device 'cuda:0' --bs 16 --trans_lanls ko,sw
python3 m2m_translate.py --from_path coco-10lan/ --to_path coco-12lan/ --device 'cuda:0' --bs 16 --trans_lanls ko,sw
python3 m2m_translate.py --from_path sbu-10lan/ --to_path sbu-12lan/ --device 'cuda:0' --bs 16 --trans_lanls ko,sw
```

If you want to translate your own datasets, please transform every line of all files to the format as follows:

```
{"caption": {"en": "A group of men are playing soccer on a field."},"image_path": "COCO_train2014_000000056786.jpg"}
```