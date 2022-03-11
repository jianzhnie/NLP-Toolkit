'''
Author: jianzhnie
Date: 2022-03-11 11:15:41
LastEditTime: 2022-03-11 12:08:31
LastEditors: jianzhnie
Description:

'''
from pprint import pprint

from datasets import load_dataset

if __name__ == '__main__':
    # 加载数据集
    imdb_dataset = load_dataset('examples/dataset_examples/imdb.py')
    # imdb_dataset = load_dataset('imdb')
    # 从Hugging Face GitHub 仓库或AWS bucket（如果尚未存储在库中）下载并在库中导入了imdb python处理脚本。
    # 运行imdb脚本以下载数据集。
    # 根据用户请求的拆分返回数据集。默认情况下，它返回整个数据集。
    print(imdb_dataset)

    imdb_train = load_dataset('imdb', split='train')
    imdb_valid = load_dataset('imdb', split='test')

    print('Length of training set: ', len(imdb_train))
    print('First example from the dataset: \n')
    pprint(imdb_train[0])
