'''
Author: jianzhnie
Date: 2021-11-29 18:28:44
LastEditTime: 2021-11-29 18:36:45
LastEditors: jianzhnie
Description:

'''
# Defined in Section 3.4.3

import re
import sys


def remove_empty_paired_punc(in_str):
    return in_str.replace('（）', '').replace('《》',
                                            '').replace('【】',
                                                        '').replace('[]', '')


def remove_html_tags(in_str):
    html_pattern = re.compile(r'<[^>]+>', re.S)
    return html_pattern.sub('', in_str)


def remove_control_chars(in_str):
    control_chars = ''.join(
        map(chr,
            list(range(0, 32)) + list(range(127, 160))))
    control_chars = re.compile('[%s]' % re.escape(control_chars))
    return control_chars.sub('', in_str)


def cleaning_wikidata(wiki_file):
    new_text = []
    with open(wiki_file, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if re.search(r'^(<doc id)|(</doc>)', line):
                print(line)
                continue
            line = remove_empty_paired_punc(line)
            line = remove_html_tags(line)
            line = remove_control_chars(line)
            new_text.append(line)

    return new_text


if __name__ == '__main__':
    wiki_file = sys.argv[1]
    clean_txt = cleaning_wikidata(wiki_file)
