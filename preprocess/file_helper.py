"""
Helper functions for file IO.

"""

import json
import codecs
import pickle


def txt_reader(txt_file_path, encoding='utf-8', ignore_errors=False):
    """
    Read txt file.

    :param txt_file_path: Text file path
    :param encoding: The name of the encoding used to decode the file.
    :param ignore_errors: whether to ignore unsupported characters
    :return: List of text strings
    """
    txt_list = []
    if ignore_errors:
        with codecs.open(txt_file_path, "r", encoding=encoding, errors='ignore') as file:
            for txt in file:
                txt_list.append(txt.rstrip("\n"))
    else:
        with open(txt_file_path, "r", encoding=encoding) as file:
            for txt in file:
                txt_list.append(txt.rstrip("\n"))
    return txt_list


def txt_writer(txt_list, txt_file_path):
    with open(txt_file_path, "w") as file:
        for row in txt_list:
            file.write(row + '\n')


def json_reader(json_file_path):
    return json.load(open(json_file_path, "r"))


def json_writer(dictionary, json_file_path, indent=True):
    """
    Write dictionary to json file.

    :param dictionary: Dictionary object to save
    :param json_file_path: Json file path
    :param indent: Whether to use indent in json file
    """
    with open(json_file_path, 'w') as file:
        if indent:
            json.dump(dictionary, file, indent=4)
        else:
            json.dump(dictionary, file)


def pickle_reader(pkl_file_path):
    return pickle.load(open(pkl_file_path, "rb"))


def pickle_writer(obj, pkl_file_path):
    with open(pkl_file_path, "wb") as file:
        pickle.dump(obj, file)
