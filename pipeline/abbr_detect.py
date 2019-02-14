"""
Abbr Detection
"""
import re
import json
import pickle
import xml.etree.ElementTree as ET
import spacy

nlp = spacy.load("en")

NORMALIZE_PATTERN = re.compile(r"[\s\-_]+")


class SenseInventory:
    def __init__(self,sense_file_path):
        doc = ET.parse(sense_file_path)
        cache = self.cache = {}
        id_cache = self.id_cache = {}
        sense_cache = self.sense_cache = {}
        for abbr_elem in doc.getroot():
            abbr_name = abbr_elem.get("name")
            abbr_cache = cache[abbr_name] = {}
            for sense_elem in abbr_elem:
                id = sense_elem.get("id")
                sense = sense_elem.get("normalized_name")
                sense_info = abbr_cache[id]=(
                    sense, # normalized sense
                    id, # id
                    int(sense_elem.get("freq")), # freq
                    [(alternative_elem.text,int(alternative_elem.get("freq"))) for alternative_elem in sense_elem], #alternatives
                    abbr_name # abbreviation name
                )
                id_cache[id] = sense_info
                sense_cache[sense] = sense_info

    def get_by_id(self,id):
        return self.id_cache[id]

    def find_id_by_text(self,un_normalized_sense):
        normalized_name = " ".join([word.lemma_ for word in nlp(NORMALIZE_PATTERN.sub(" ", un_normalized_sense.lower()))])
        if  normalized_name in self.sense_cache:
            return self.sense_cache[normalized_name][1]
        else:
            return None

    def has_abbr(self,abbr):
        return abbr in self.cache


def load_abbr_from_sense_inventory(file_path):
    inventory = SenseInventory(file_path)
    abbr_set = set()
    for abbr in inventory.cache:
        if " " not in abbr:
            abbr_set.update([abbr])
    return abbr_set


def load_abbr_from_cleaned_sense_inventory(file_path):
    abbr_inventory = set()
    for line in open(file_path):
        obj = json.loads(line)
        abbrs = obj['ABBR']
        for abbr in abbrs:
            if " " not in abbr:
                abbr_inventory.update([abbr])
    return abbr_inventory


def load_abbr_from_lexicon_tool(file_path):
    # Abbrs from Lexicon Tool
    abbr_set_lexicon = set()
    with open(file_path, 'r') as file:
        for line in file:
            abbr = line.split("|")[1]
            if " " not in abbr:
                abbr_set_lexicon.update([abbr])
    return abbr_set_lexicon


if __name__ == '__main__':
    data_path = "/home/luoz3/wsd_data"
    lexicon_abbr_file = data_path + "/umls/LRABR"
    PATH_INVENTORY_JSON = data_path + '/final_cleaned_sense_inventory_with_testsets.json'

    abbr_set_sense_inventory = load_abbr_from_sense_inventory(data_path + "/normalized_abbr_sense2.xml")
    abbr_set_lexicon = load_abbr_from_lexicon_tool(lexicon_abbr_file)
    abbr_set_cleaned = load_abbr_from_cleaned_sense_inventory(PATH_INVENTORY_JSON)

    abbr_inventory = abbr_set_lexicon.union(abbr_set_cleaned)
    abbr_inventory = abbr_set_sense_inventory.union(abbr_inventory)

    pickle.dump(abbr_inventory, open(data_path + "/abbr_inventory.pkl", "wb"))
    print(len(abbr_inventory))
    print()
