from pprint import pprint

from datasets import load_dataset

if __name__ == '__main__':
    # datasets = list_datasets()
    # print("Number of datasets in the Datasets library: ", len(datasets),
    #       "\n\n")

    # # 数据集列表
    # # pprint(datasets, compact=True)

    # # 数据集属性
    # imdb = list_datasets(with_details=True)[datasets.index('imdb')]

    # # 调用python数据类
    # pprint(imdb.__dict__)

    # 加载数据集
    imdb_dataset = load_dataset('imdb')
    # 从Hugging Face GitHub 仓库或AWS bucket（如果尚未存储在库中）下载并在库中导入了imdb python处理脚本。
    # 运行imdb脚本以下载数据集。
    # 根据用户请求的拆分返回数据集。默认情况下，它返回整个数据集。
    print(imdb_dataset)

    imdb_train = load_dataset('imdb', split='train')
    imdb_valid = load_dataset('imdb', split='test')

    print('Length of training set: ', len(imdb_train))
    print('First example from the dataset: \n')
    pprint(imdb_train[0])
