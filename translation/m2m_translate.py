import argparse
import json
import os.path
import time
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
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
        sentences = '我爱你'
    if batch_size == 0:
        return model.translate(sentences, target_lang=target_lang)
    else:
        return model.translate_sentences(
            sentences,
            target_lang=target_lang,
            source_lang=source_lang,
            batch_size=batch_size,
            show_progress_bar=True
        )


def test(text):
    model = load_model("m2m-100-lg", args.device)
    for lan in args.trans_lanls:
        print(translate(model, text, source_lang=args.source_lang, target_lang=lan, batch_size=4), flush=True)


def translate_one(from_path, to_path, trans_lanls, model):
    print(f"translate {from_path} to {trans_lanls} begin", flush=True)
    with open(from_path) as f:
        lines = f.readlines()
    out_str = ''
    n = len(lines)
    en_text_ls = []
    trans_text_ls = {}
    data_ls = []
    for i in range(n):
        line = lines[i].strip()
        if line != '':
            data = json.loads(line)
            data_ls.append(data)
            en_text_ls.append(data["caption"]['en'])
    for lan in trans_lanls:
        print(f"{lan} begin", flush=True)
        begin_time = time.time()
        with torch.no_grad():
            if args.gpu_num == 1:
                trans_text_ls[lan] = translate(model, en_text_ls, source_lang=args.source_lang, target_lang=lan,
                                               batch_size=args.bs)
            else:
                trans_text_ls[lan] = model.translate_multi_process(process_pool, en_text_ls,
                                                                   source_lang=args.source_lang,
                                                                   target_lang=lan, show_progress_bar=True)

        delta_time = (time.time() - begin_time) / 60
        print(f"{lan} end using {delta_time} minutes", flush=True)
    n = len(data_ls)
    for i in range(n):
        data = data_ls[i]
        for lan in trans_lanls:
            data['caption'][lan] = trans_text_ls[lan][i]
        out_str += json.dumps(data, ensure_ascii=False) + "\n"
    with open(to_path, "w", encoding='utf-8') as f:
        f.write(out_str)
    print(f'save to {to_path} done', flush=True)


def translate_data(from_path, to_path, trans_lanls, model):
    assert os.path.exists(from_path)
    if os.path.isdir(from_path):
        if not os.path.exists(to_path):
            os.mkdir(to_path)
        for cnt, file in enumerate(os.listdir(from_path)):
            print(f"######  file {cnt} begin  ######", flush=True)
            translate_one(os.path.join(from_path, file), os.path.join(to_path, file), trans_lanls, model)
            print(f"######  file {cnt} end  ########", flush=True)
    elif os.path.isfile(from_path):
        translate_one(from_path, to_path, trans_lanls, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_path", type=str, help='translate from which path, e.g. vg-10lan/part_0.data',
                        default='in.data')
    parser.add_argument("--to_path", type=str, help='translate to which path, e.g. vg-11lan/part_0.data',
                        default='out.data')
    parser.add_argument("--trans_lanls", type=str,
                        help="language list to be translated using comma concat", default='id,ru,es,tr')
    parser.add_argument("--test", action='store_true', help='whether test the translation')
    parser.add_argument("--text", type=str, default='我爱你', help='for test only')
    parser.add_argument("--device", type=str, default='cpu', help='cpu or cuda:n')
    parser.add_argument("--source_lang", type=str, default='en', help='translate from which language')
    parser.add_argument("--bs", type=int, default=16, help='batch size to translate')
    parser.add_argument("--gpu_num", default=1, type=int)
    args = parser.parse_args()
    args.trans_lanls = args.trans_lanls.split(',')
    with torch.no_grad():
        if args.test:
            test(args.text)
        else:
            if args.gpu_num > 1:
                model = EasyNMT("m2m_100_1.2B")
                process_pool = model.start_multi_process_pool(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
                model.translate_multi_process(process_pool, ["hello world!!" for _ in range(10)], source_lang='en',
                                              target_lang='de',
                                              show_progress_bar=False)
            else:
                model = load_model("m2m-100-lg", args.device)

            translate_data(args.from_path, args.to_path, args.trans_lanls, model)
            if args.gpu_num > 1:
                model.stop_multi_process_pool(process_pool)
