import os
import json
from fasttext import FastText
from pathlib import Path 
import shutil

ft = FastText.load_model("prepair_dataset/lid.176.ftz")

def load_data_from_txt_file(txts_path):
    with open(txts_path, 'r') as buf:
        data = buf.readlines()
        
        json_data = []
        for line in data:
            json_data.append(json.loads(line))
        
        buf.close()        
    return json_data

def get_txts_from_annotations(annotations):
    txts = []
    for annotation in annotations:
        txts.append(annotation['text'])
        
    return txts
    

def detect_lang(txts, lang='vi'):
    labels = ft.predict(txts)[0]
    num_of_lang_vi = 0
    for label in labels:
        lang_pred = label[0].split('__')[-1]
        
        if lang_pred == lang:
            num_of_lang_vi += 1

    ratio = num_of_lang_vi / len(txts)
    return num_of_lang_vi, len(txts), ratio >= 0.4

def main():
    data_repair_dict = {
        'prepair_dataset/data/train.txt': 'train.txt',
        'prepair_dataset/data/test.txt': 'test.txt'
    }
    
    old_img_dir = 'prepair_dataset/data/raw/img'

    img_repaired_dir = Path('prepair_dataset/data_repaired/raw/img')
    img_repaired_dir.mkdir(parents=True, exist_ok=True)
    
    for txt_file_path, txt_save_path in data_repair_dict.items():
        json_datas = load_data_from_txt_file(txt_file_path)
       
        path_txt_dir= Path('prepair_dataset/data_vn')
        path_txt_dir.mkdir(parents=True, exist_ok=True)
        txt_save_path = path_txt_dir / txt_save_path
        
        buf = open(txt_save_path, 'a+') 
        for json_data in json_datas:
            txts = get_txts_from_annotations(json_data['annotations'])
            _, _, flag = detect_lang(txts)
            if flag:
                file_image = json_data['file_name']
                file_image = file_image.split('/')[-1]
                shutil.copy2(os.path.join(old_img_dir, file_image), os.path.join(img_repaired_dir, file_image))    
                
                json_data = str(json_data).replace('\'', '"')
                buf.write(json_data + '\n') 
        
        buf.close()
    
if __name__ == '__main__':
    main()