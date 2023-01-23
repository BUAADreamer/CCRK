import json
import time

import torch.cuda
from easynmt import EasyNMT

MODEL_TYPES = ["m2m-100-md", "m2m-100-lg", "opus-mt", "mbart50_m2m"]


def load_model(model_type, device):
    if model_type == "m2m-100-md":
        return EasyNMT("m2m_100_418M", device=device)
    elif model_type == "m2m-100-lg":
        return EasyNMT("m2m_100_1.2B", device=device)
    else:
        return EasyNMT(model_type, device=device)


def translate(model, sentences=None, source_lang='zh', target_lang='en', batch_size=0):
    if sentences == None:
        sentences = '我爱中国'
    if batch_size == 0:
        return model.translate(sentences, target_lang=target_lang)
    else:
        return model.translate_sentences(
            sentences,
            target_lang=target_lang,
            source_lang=source_lang,
            batch_size=batch_size,
        )


def translate_multi_p(model, sentences=None, source_lang='zh', target_lang='en', batch_size=0):
    process_pool = model.start_multi_process_pool()
    translations_multi_p = model.translate_multi_process(process_pool, sentences, source_lang='en', target_lang='de',
                                                         show_progress_bar=True)
    model.stop_multi_process_pool(process_pool)
    return translations_multi_p


def load_dataset():
    dataset2num = {
        "cc3m": 0,  # 256
        "coco": 0,  # 64
        "sbu": 0,  # 64
        "vg": 1  # 128
    }
    dataset2sentences = {
        "cc3m": [],
        "coco": [],
        "sbu": [],
        "vg": []
    }
    for k in dataset2num:
        num = dataset2num[k]
        path = f"/home/LAB/fengzc/pretrain_data/translated_4M/{k}/"
        for i in range(num):
            dataset2sentences[k].append([])
            file_path = path + "part_{:d}.data".format(i)
            if k == "cc3m":
                file_path = path + "part-{:05d}.data".format(i)
            lines = open(file_path).readlines()
            for line in lines:
                line = line.strip()
                if line != '':
                    data = json.loads(line)
                    dataset2sentences[k][i].append(data["caption"]["en"])
    return dataset2sentences


def test():
    cuda = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("m2m-100-lg", cuda)
    sentence = """Il existe de nombreux types de sources de logement pour les locations en France.
  Le premier est l'appartement des étudiants. Ils sont généralement situés à l'intérieur ou sur un campus universitaire. Parce que ces appartements universitaires sont moins chers à louer que d'habitude, ils sont très populaires auprès des étudiants universitaires. Le deuxième type est l'appartement privé pour étudiants universitaires. Le premier type d'appartement universitaire est en pénurie, de sorte que l'appartement universitaire privé se développe rapidement. Mais c'est plus cher qu'un appartement universitaire.
Le troisième type est l'internat familial. Il vous permet de vivre avec la famille du propriétaire, ce qui vous permet de mieux comprendre la culture française.
La France est un pays développé et très attrayant, avec une grande variété de logements. Les étudiants étrangers en France peuvent choisir en fonction de leurs besoins."""
    sentences = ['我爱中华', '我是中国人', '我是英国人', '这是一句中文']
    begin_time = time.time()
    print(translate(model, sentence, source_lang='fr', target_lang='zh', batch_size=4))
    print(time.time() - begin_time)


def translate_4M():
    dataset2sentences = load_dataset()
    dataset2res = {}
    for k in dataset2sentences:
        dataset2res[k] = []
    model = load_model("m2m-100-lg", "cuda")
    for k in dataset2sentences:
        all_begin_time = time.time()
        all_sentences = dataset2sentences[k]
        cnt = 0
        for i in range(len(all_sentences)):
            sentences = all_sentences[i]
            cnt += len(sentences)
            begin_time = time.time()
            dataset2res[k].append(translate(model, sentences, target_lang='tr', batch_size=64))
            delta_time = time.time() - begin_time
            print("\r", f"{i} done, use {delta_time}", end='', flush=True)
        all_delta_time = time.time() - all_begin_time
        all_delta_time = round(all_delta_time / 60, 2)
        print("\r", f"{k} done, use {all_delta_time} min, translate {cnt} sentence")


if __name__ == '__main__':
    # translate_4M()
    test()
else:
    # model = load_model("m2m-100-lg", "cuda:0")
    # print("model load done")
    pass
