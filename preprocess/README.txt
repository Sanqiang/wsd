*** Process pipeline on MIMIC ***

MIMIC raw data processed by Rui:
    - WSD_DATA_FOLDER = /home/mengr/project/wsd/wsd_data/
    - UMLS sense inventory:
        WSD_DATA_FOLDER/mimic/final_cleaned_sense_inventory.json
    - A coarse longform detection on MIMIC (only tells if a longform appears in the text, location is not given):
        WSD_DATA_FOLDER/mimic/find_longform_mimic/

1. mimic_inventory.py根据你的abbr文件挑选合适的abbr和longform 配对，生成字典保存
        WSD_DATA_FOLDER/mimic/final_cleaned_sense_inventory.cased.processed.pkl
2. dataset/mimic.py对原文进行预处理，然后替换longform
        WSD_DATA_FOLDER/mimic/processed/
3. 最后mimic_vocab.py根据处理的文本，切分train&test，再生成train model的其他必要文件
        WSD_DATA_FOLDER/mimic/train, WSD_DATA_FOLDER/mimic/vocab etc.
4. 其他的dataset上的处理都是只预处理，生成model可以用的文本，没有找long form，也没有切分数据集