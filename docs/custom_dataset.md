# HuggingFace  dataset  API

在前面Bert的微调中，加载数据的方式是

```python
from datasets import load_dataset, load_metric
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
```

打印的结构是：

```python
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
```

打印出来一个示例就是：

```python
datasets["train"][0]
--------------------------------------------------------------------
输出:
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},

'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',

'id': '5733be284776f41900661182',

'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',

'title': 'University_of_Notre_Dame'}
```

那么使用load_dataset方法加载本地数据呢？ 什么样的数据都可以加载吗？ 结论当然是可以的，仅仅需要自己写个脚本就可以了，下面先翻译HuggingFace Datasets关于 Writing a dataset loading script的官方教程，然后我会写一个自己的示例。

## 编写数据集加载脚本

Writing a dataset loading script

希望使用本地/私有数在前面Bert的微调中，加载数据的方式是据文件，而 CSV/JSON/txt的通用数据标识符(请参阅本地文件)对于您的用例来说是不够的。

在编写新的数据集加载脚本时，可以从[数据集加载脚本的模板](https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py)开始。你可以在 github 仓库的 templates 文件夹中找到这个模板。

下面是生成数据集所涉及的类和方法的一个快速概述:



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210302205825405.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzg4NzQy,size_16,color_FFFFFF,t_70#pic_center)



左边是在库内部创建datasets.Dataset实例的一般结构。在右边，是每个特定数据集的加载脚本。要创建一个新的数据集的加载脚本，一般需要在datasets.DatasetBuilder类指定三个方法:

- datasets.DatasetBuilder._info()
  - 负责将数据集的元数据指定为datasets.DatasetInfo数据类型，尤其是datasets.Features定义了数据集每一列的名字和类型
- datasets.DatasetBuilder._split_generator()
  - 负责加载或检索数据文件，通过 splits 和在需要时生成过程定义特定的参数组织文件
- datasets.DatasetBuilder._generate_examples()
  - 负责加载 split 过的文件和生成在features格式的示例

可选的，数据集加载脚本可以定义一个数据集要使用的配置，这个配置文件是datasets.DatasetBuilder从datasets.BuilderConfig继承的。这样的类允许我们自定义构建处理，例如，运行选择数据特定的子集或者加载数据集时使用特定的方法处理数据。

关于命名的注意:

- 数据集类(dataset class)应该是驼峰命名法(camel case, 首字母大小)，
- 数据集的名称使用蛇形命名法(snake case，小写加下划线 ‘_’)。例如：数据集book_corpus是 class BookCorpus(datasets.GeneratorBasedBuilder)

## 添加数据集元数据

 Adding dataset metadata

datasets.DatasetBuilder._info() 方法负责将数据集元数据指定为datasets.DatasetInfo数据类型。尤其是datasets.Features定义了数据集每一列的名称。datasets.DatasetInfo是一组预定义的数据，无法扩展。完整的属性列表可以在包引用(package reference)中找到。

需要指定的最重要属性:

- datasets.DatasetInfo.features:
  - datasets.Features 实例，用于定义数据集的每一列的名称和类型，以及示例的一般结构
- datasets.DatasetInfo.description:
  - 描述数据集的str，
- datasets.DatasetInfo.citation:
  - 包含数据集引用的 BibTex 格式的 str，引用该数据集的方式包含在通信中，
- datasets.DatasetInfo.homepage:
  - 一个包含数据集原始主页 URL 的 str。

### DatasetBuilder._info()

例如，这里的datasets.DatasetBuilder._info()是SQuAD数据集的示例，来着 squad数据集加载脚本

```python
def _info(self):
  return datasets.DatasetInfo(
    description=_DESCRIPTION,
    features=datasets.Features(
      {
        "id": datasets.Value("string"),
        "title": datasets.Value("string"),
        "context": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.features.Sequence(
          {
            "text": datasets.Value("string"),
            "answer_start": datasets.Value("int32"),
          }
        ),
      }
    ),
    # No default supervised_keys (as we have to pass both question
    # and context as input).
    supervised_keys=None,
    homepage="https://rajpurkar.github.io/SQuAD-explorer/",
    citation=_CITATION,
    task_templates=[
      QuestionAnsweringExtractive(
        question_column="question", context_column="context", answers_column="answers"
      )
    ],
  )
```

### Features

datasets.Features定义了每个示例的结构，并可以定义具有各种类型字段的任意嵌套对象。关于可用feature的更多细节可以在数据集特性指南和数据集包参考中找到。在 GitHub 存储库提供的各种数据集脚本中也可以找到许多特性的例子，甚至可以在数据集查看器(datasets viewer)上直接检查。

以下是SQuAD数据集的特性，例如，它取自SQuAD数据集加载脚本:

```python
features=datasets.Features(
  {
    "id": datasets.Value("string"),
    "title": datasets.Value("string"),
    "context": datasets.Value("string"),
    "question": datasets.Value("string"),
    "answers": datasets.features.Sequence(
      {
        "text": datasets.Value("string"),
        "answer_start": datasets.Value("int32"),
      }
    ),
  }
)
```

在上面的介绍中，这些特性基本上是不言而喻的。这里的一个特定行为是给“ answers”中的 Sequence 字段提供了一个子字段字典。正如上面提到的，在这种情况下，这个特性实际上被转换为一个列表字典(而不是我们在这个特性中读到的字典列表)。这一点在SQuAD数据集加载脚本, 通过最后的生成方法产生的例子的结构中得到了证实:

```python
answer_starts = [answer["answer_start"] for answer in qa["answers"]]
answers = [answer["text"].strip() for answer in qa["answers"]]

yield id_, {
    "title": title,
    "context": context,
    "question": question,
    "id": id_,
    "answers": {"answer_start": answer_starts, "text": answers,},
}
```

这里的"answers" 相应地提供了一个列表的字典，而不是一个字典的列表。

让我们来看看另一个来自 Race([large-scale reading comprehension dataset Race](https://huggingface.co/datasets/race)) 的特征例子:

```python
features=datasets.Features(
    {
        "article": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "question": datasets.Value("string"),
        "options": datasets.features.Sequence({"option": datasets.Value("string")})
    }
)
```



## 下载数据文件并组织拆分

 Downloading data files and organizing splits

datasets.DatasetBuilder._split_generator()方法负责下载(或者检索本地数据文件)，根据分片(splits)进行组织，并在需要时生成过程中定义特定的参数

此方法以datasets.DownloadManager作为输入,这是一个实用程序，可用于下载文件（如果它们是本地文件或已经在缓存中，则可以从本地文件系统中检索它们）并返回一个datasets.SplitGenerator列表。datasets.SplitGenerator是一个简单的数据类型，包含split和关键字参数的名称DatasetBuilder._generate_examples() 方法将在下一部分中详细介绍。

这些参数可以特定于每个split，并且，通常至少包括要为每个拆分加载的数据文件的本地路径

- Using local data files:

  如果你的数据不是在线数据，而是本地数据文件，那么datasets.BuilderConfig特别提供了两个参数。这两个参数是data_dir和data_files可以自由地用于提供目录路径或文件路径。这两个属性可以在调用datasets.load_dataset()时使用相关关键字参数，例如:dataset = datasets.load_dataset('my_script', data_files='my_local_data_file.csv'),并且，这个值通过访问self.config.data_dir和self.config.data_files在datasets.DatasetBuilder._split_generator()

让我们来看datasets.DatasetBuilder._split_generator()方法的一个简单的示例。我们来看一个squad数据集加载脚本的例子:

```python
class Squad(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    _URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    _URLS = {
        "train": _URL + "train-v1.1.json",
        "dev": _URL + "dev-v1.1.json",
    }

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
```

正如您所看到的，此方法首先为 SQuAD 的原始数据文件准备 URL。这个字典然后提 datasets.DownloadManager.download_and_extract() 方法，它将负责下载或者从本地文件系统索引文件，并且返回具有相同类型和组织结构的对象(这里是字典) 。datasets.DownloadManager.download_and_extract() 能够获得输入的 URL/PATH 或者 URLs/paths字典，并返回具有相同结构的对象(单个 URL/路径、 URL/路径的列表或字典)和本地文件的路径。此方法还负责提取 tar、 gzip 和 zip 压缩文档。

> 注意: 除了datasets.DownloadManager.download_and_extract()和datasets.DownloadManager.download_custom(),
>
> datasets.DownloadManager类还通过几种方法提供了对下载和提取过程的更细粒度控制，这些方法包括:
>
> - datasets.DownloadManager.download(),
>
> - datasets.DownloadManager.extract() 和
> - datasets.DownloadManager.iter_archive()

请参考数据集上的包参考 datasets.DownloadManager 了解这些方法的详细信息。数据文件下载后, datasets.DatasetBuilder._split_generator（） 方法的下一个任务是对每个 datasets.DatasetBuilder._generate_examples() 方法调用的结果来准备使用 datasets.SplitGenerator。我们将在下一个会话中详细介绍。

datasets.SplitGenerator是一个简单的数据类型，包括:

- name(string):
  - 关于分割(split)的名称（如果可能），可以使用数据集中提供的标准分割名称。Split可以使用：datasets.Split.TRAIN，datasets.Split.VALIDATION和datasets.Split.TEST，

- gen_kwargs(dict):
  - 关键字参数(keywords arguments)提供给datasets.DatasetBuilder._generate_examples()方法生成分割中的示例。
  - 这些参数可以特定于每个分割，通常至少包含要为每个拆分加载的数据文件的本地路径，如上面的SQuAD示例所示。

## 在每个分割中生成样本

Generating the samples in each split

datasets.DatasetBuilder._generate_examples()是负责读取数据文件以进行分割，并产生示例，这些示例是在datasets.DatasetBuilder._info()设置的特定的feature格式。

datasets.DatasetBuilder._generate_examples()的输入参数是由gen_kwargs字典定义的由之前详细介绍的datasets.DatasetBuilder._split_generator()方法。

再一次，让我们以squad数据集加载脚本的简单示例为例：
------------------------------------------------
```python
def _generate_examples(self, filepath):
    """This function returns the examples in the raw (text) form."""
    logger.info("generating examples from = %s", filepath)
    with open(filepath) as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"].strip() for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    yield id_, {
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                        "answers": {"answer_start": answer_starts, "text": answers,},
                    }
```

输入的参数是datasets.DatasetBuilder._split_generator()方法返回的每个dataset.SplitGenerator的gen_kwargs中提供的文件路径。

该方法读取并解析输入文件，并生成一个由id_(可以是任意的，但应该是唯一的(为了向后兼容TensorFlow数据集)和一个示例组成的元组。该示例是一个具有与datasets.DatasetBuilder._info()中定义的特性相同的结构和元素类型的字典。

注意:由于生成数据集是基于python生成器的，因此它不会将所有数据加载到内存中，因此它可以处理相当大的数据集。但是，在刷新到磁盘上的数据集文件之前，生成的示例存储在ArrowWriter缓冲区中，以便分批写入它们。如果您的数据集的样本占用了大量内存(带有图像或视频)，那么请确保为数据集生成器类的_writer_batch_size类属性指定一个低值。我们建议不要超过200MB。

## 指定几个数据集配置

Specifying several dataset configurations

有时，希望提供对数据集的多个子集的访问，例如，如果数据集包含几种语言或由不同的子集组成，或者希望提供几种构建示例的方法。

这可以通过定义一个特定的datasets.BuilderConfig类，并提供这个类的预定义实例供用户选择来实现。
基本dataset.BuilderConfig类非常简单，只包含以下属性:

- name(str)是数据集配置的名字。例如，如果不同的配置特定于不同的语言，则使用语言名来配置,
- version可选的版本标识符,

- data_dir(str)用于存储包含数据文件的本地文件夹的路径,

- data_files(Union[Dict, List])可用于存储本地数据文件的路径,
- description(str) 可以用来对配置进行长篇描述.

datasets.BuilderConfig仅作为一个信息容器使用，这些信息可以通过在datasets.DatasetBuilder实例的self.Config属性中访问来在– datasets.DatasetBuilder中构建数据集。

有两种方法来填充datasets.BuilderConfig类或子类的属性:

可以在数据集的datasets.DatasetBuilder.BUILDER_CONFIGS属性中设置预定义的datasets.BuilderConfig类或子类列表。然后可以通过将其名称作为name关键字提供给datasets.load_dataset()来选择每个特定的配置，

当调用datasets.load_dataset()时，所有不是特定于datassets.load_dataset()方法的关键字参数将用于设置datasets.BuilderConfig类的相关属性，并在选择特定配置时覆盖预定义的属性。

让我们看一个从CSV文件加载[脚本](https://github.com/huggingface/datasets/blob/master/datasets/csv/csv.py)改编的示例。

假设我们需要两种简单的方式来加载CSV文件:使用“，”作为分隔符(我们将此配置称为’comma’)或使用“;”作为分隔符(我们将此配置称为’semi-colon’)。

我们可以用delimiter属性定义一个自定义配置:

```python
@dataclass
class CsvConfig(datasets.BuilderConfig):
    """BuilderConfig for CSV."""
    delimiter: str = None
```

然后在DatasetBuilder中定义几个预定义的配置:

```python
class Csv(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = CsvConfig
    BUILDER_CONFIGS = [CsvConfig(name='comma',
                                 description="Load CSV using ',' as a delimiter",
                                 delimiter=','),
                       CsvConfig(name='semi-colon',
                                 description="Load CSV using a semi-colon as a delimiter",
                                 delimiter=';')]
```

```python
def self._generate_examples(file):
    with open(file) as csvfile:
        data = csv.reader(csvfile, delimiter = self.config.delimiter)
        for i, row in enumerate(data):
            yield i, row
```
这里我们可以看到如何使用self.config.delimiter属性来控制读取CSV文件。

数据集加载脚本的用户将能够选择一种或另一种方式来加载带有配置名称的CSV文件，甚至可以通过直接设置分隔符属性来选择完全不同的方式。例如使用这样的命令:

```python
from datasets import load_dataset
dataset = load_dataset('my_csv_loading_script', name='comma', data_files='my_file.csv')
dataset = load_dataset('my_csv_loading_script', name='semi-colon', data_files='my_file.csv')
dataset = load_dataset('my_csv_loading_script', name='comma', delimiter='\t', data_files='my_file.csv')
```

在最后一种情况下，配置设置的分隔符将被指定为load_dataset参数的分隔符覆盖。

虽然在这种情况下使用配置属性来控制数据文件的读取/解析，但配置属性可以在处理的任何阶段使用，特别是:

要控制在datasets.DatasetBuilder._info()方法中设置的datasets.DatasetInfo属性，例如特性，
要控制在datasets.DatasetBuilder._split_generator()方法中下载的文件，例如根据配置定义的语言属性选择不同的url

在[Super-GLUE加载脚本](https://github.com/huggingface/datasets/blob/master/datasets/super_glue/super_glue.py)中可以找到一个带有一些预定义配置的自定义配置类的例子，该脚本通过配置提供了对SuperGLUE基准测试的各种子数据集的控制。另一个例子是[Wikipedia加载脚本](https://github.com/huggingface/datasets/blob/master/datasets/wikipedia/wikipedia.py)，它通过配置提供对Wikipedia数据集语言的控制。

指定默认数据集配置(Specifying a default dataset configuration)
当用户加载具有多个结构的数据集时，他们必须指定一个结构名，否则会引发ValueError。对于一些数据集，最好指定一个默认结构，如果用户没有指定，它将被加载。

这可以通过datasets.DatasetBuilder.DEFAULT_CONFIG_NAME属性完成。通过将此属性设置为一个数据集配置的名称，在用户没有指定结构名称的情况下，该结构将被加载。

这个特性是可选的，应该只在默认配置对数据集有意义的地方使用。例如，许多跨语言数据集对于每种语言都有不同的配置。在这种情况下，创建一个可以作为默认配置的聚合配置可能是有意义的。这实际上会默认加载数据集的所有语言，除非用户指定了特定的语言。有关示例，请参阅通晓多种语言的NER加载脚本。

测试数据集加载脚本(Testing the dataset loading script)
一旦你完成了创建或调整数据集加载脚本，你可以通过给出数据集加载脚本的路径在本地尝试它:

```python
from datasets import load_dataset
dataset = load_dataset('PATH/TO/MY/SCRIPT.py')
```

如果您的数据集有几个配置，或者需要指定到本地数据文件的路径，您可以相应地使用datasets.load_dataset()的参数:

```python
from datasets import load_dataset
dataset = load_dataset('PATH/TO/MY/SCRIPT.py', 'my_configuration',  data_files={'train': 'my_train_file.txt', 'validation': 'my_validation_file.txt'})
```



## 数据集参考脚本(Dataset scripts of reference)

数据集共享相同的格式是很常见的。因此，可能已经存在一个数据集脚本，您可以从中获得一些灵感来帮助您编写自己的数据集脚本。

这是一个在本地上的数据集，它的一个示例为：

```
webqa_data = json.loads(open('./data/WebQA.json', encoding='utf8').read())
```

```
webqa_data[0]
----------------------
{'passages': [{'answer': '', 'passage': '商王朝最后一个君王叫纣，最早以亳为都城。'},
  {'answer': '', 'passage': '纣：中国商代最后一位君主。'},
  {'answer': '纣王',
   'passage': '商王朝最后一个君王叫帝辛，也就是纣王，商王朝都成在殷商，商王朝灭亡的一场战役是牧野之战，商王朝灭了，建立了周朝。'},
  {'answer': '',
   'passage': '答：第一个是秦始皇，首称“朕”、并六国、统度量衡、焚书坑儒、造长城等、是一个暴君也是一个雄才伟略的大作为君王。最后一个是清朝宣统帝，爱新觉罗．溥仪，为日本人控制建立伪满政权，写了本自传。'},
  {'answer': '', 'passage': '鸣条一战，夏师败绩，夏桀奔南巢而死，成汤则成为商代的第一位君王。'},
  {'answer': '', 'passage': '最后一个君王叫杰朝都在殷商战役不知道商灭亡后建立周朝'},
  {'answer': '', 'passage': '他在位期间致力于修明政治，统一中国北方，政绩显著，是十六国时期许多封建帝王中最杰出君王。'},
  {'answer': '',
   'passage': '中国皇帝（君王）包括正统朝代和少数民族建立的政权，还有一些政变、夺权所建立的政权，再加上农民起义建立的政权，中国皇帝共有１０００多位呢！附：南越、东越、闽越、东瓯、匈奴、突厥、回纥（回鹘）、吐蕃、高昌、于阗、柔然、吐谷浑、渤海国（大震）、南诏（大蒙、大礼、大封民）、大长和、大天兴、大义宁、大理国（前理汉武帝刘彻、后理）、大中、东夏（大真）（以上不包括十六国时期和五代十国时期的少数民族政权）其中云南列朝自世隆以下【南诏（大蒙、大礼、大封民）、大长和、大天兴、大义宁、大理国（前理、后理）、大中】和东夏（大真）的君主称皇帝；南越（吕后时）、于阗（五代时）的君主一度称皇帝；南越、东越、闽越、东瓯、高昌、于阗、吐谷浑、渤海国（大震）作为中原王朝的藩属国，君主称王；匈奴的君主称单于；回纥（回鹘）、柔然的君主称可汗；吐蕃的君主称赞普。'},
  {'answer': '', 'passage': '中国第一个有记载有国号的王朝是夏，最后一个国王是桀。'},
  {'answer': '', 'passage': '周朝是中国第三个也是最后一个世袭分封制王朝'},
  {'answer': '', 'passage': '仲壬亦称中壬、燕壬、工壬、其壬、南壬，姓子名庸，是中国商朝的一位君王。'},
  {'answer': '',
   'passage': '在中国汉代的历史学家司马迁的着作《史记》中记载，商代最后一个国王纣的兄弟箕子在周武王伐纣后，带着商代的礼仪和制度到了朝鲜半岛北部，被那里的人民推举为国君，并得到周朝的承认。'},
  {'answer': '', 'passage': '君王已成了历史名词，中国再不会有皇帝，留下大量史料让后人去挖掘。'},
  {'answer': '',
   'passage': '后人称他在贞观年间的统治为“贞观之治”。【丰功伟绩】1、他不拘一格（敌人、穷人、坏人）的用人，对人材的使用及领导达到了极高的境遇；2、他独具慧眼，看到了个人力量的不足，充分认识到君王如石、良臣如匠，方有美玉问世，对大臣的各项进步之言豁达的予以采纳；3、不独断专行、初步确立了三权分立、互相监督的政治管理制度，规定法令甚至包括自己（影响国家政策的那一部份）旨意需门下省审查副署后方可生效发布，保证了政策的可行性、及时发现并纠正/杜绝了不良政策对国家及人民的违害与影响；4、认识到人命至重、不可妄杀的法政政策，规定死刑需三复奏（外地五复奏）复审批准后方可行刑，这就不难认人们想起贞观四年（630年---中国的丰年）全国叛死刑才29人、贞观六年（632年）全国死刑犯290人，太宗审查时令全部290人回家团年、待来年秋收后回来复刑，结果290人均准时到来、无一人逃亡（现在有人说那是太宗广布法网，那290人是跑不掉才回来受死的，我说这人真是不动脑子，想想那时的法网严还是现在的法网严，那现在逃狱是不是100%呢，那又是为什么呢！！！）。5、太宗朝武功之盛，除太高丽战争上没有取得战略胜利外都取得了辉煌的胜利（突厥、吐谷浑、高昌、安西四镇、漠北薛延陀等），这与当时的国力、军队战斗力、整体战略、用人选将与配合默契、过程协调一致等重要因素是分不开的，因此在中华历史上的名将名相中，贞观朝占有相当的比例，在中华军事史上，贞观朝的战例也多被引用；6、气吞天下的“天可汗”气质，李世民多次以少吓多，经典之役就是在渭水单骑吓退突厥10万精骑，就对比宋真宗在寇准一在坚持和请求下才免强在大军护卫下到达澶州南城，而又要战战兢兢的马上要回去是何等的天壤之别啊！7、胸怀大局、四海一统的民族和外交政策，太宗朝的民族和外交政策取得了辉煌的胜利，四海之内只要知道中国的均努力内附，以唐为荣，乐不思蜀，他们不但同唐人一样可以自由自在的生存，还可以做官，著名的少数民族将领阿史那社尔、执思失力、契毕何力乃至后世的高仙芝、李光弼等都为唐朝做出了杰出贡献，在他们身上正好反映出李世民民族政策的光辉，现在的唐人、唐人街也正时那时繁荣富强、威甲四海、文礼之邦的生动写照；8、完善科举制度、大力兴办学校、重视教育活动、普及官吏选聘、当时的国子学、太学之盛、地方也有不少学校，如此才不难想起当时的教化呢，同时当时的科举也规范化、考选公平，以进士科最为杰出，如此才有太宗见新科进士鱼贯而出，喜言“天下英雄'},
  {'answer': '',
   'passage': '中国皇帝（君王）包括正统朝代和少数民族建立的政权，还有一些政变、夺权所建立的政权，再加上农民起义建立的政权，中国皇帝共有１０００多位呢！'},
  {'answer': '',
   'passage': '答：中国皇帝（君王）包括正统朝代和少数民族建立的政权，还有一些政变、夺权所建立的政权，再加上农民起义建立的政权，中国皇帝共有１０００多位呢！附：南越、东越、闽越、东瓯、匈奴、突厥、回纥（回鹘）、吐蕃、高昌、于阗、柔然、吐谷浑、渤海...'},
  {'answer': '',
   'passage': '中国经历的漫长封建社会自公元前221年秦王赢政称"皇帝"始,至1912年最后一个封建皇帝溥仪在辛亥革命的炮火中宣布退位止.长达2132年.在这期间,封建皇帝总数为494人,在位时间最长的皇帝是清康熙帝和乾隆帝,在位时间最短的是金末帝完颜承麟,从即位到被杀,不足半日.'},
  {'answer': '',
   'passage': '皇朝的终结中国最后一个君主专制政府——清朝在1911年的辛亥革命中被推翻，取而代之的是共和政体中华民国（正式成立于1912年1月1日）。'},
  {'answer': '', 'passage': '桀：桀是夏朝最后一个国王，名履癸，是中国历史上有名的暴虐、荒淫的国君之一。'},
  {'answer': '', 'passage': '中国第一个王朝是夏，夏的最后一个皇帝是桀。'}],
 'question': '中国商代最后一个君王是谁?',
 'id': 'Q_IR_VAL_000000#TEST'}

```

先是一个列表(list)，列表中每个元素是一个字典，每个字典中有[‘passages’, ‘question’, ‘id’]三个关键字。

```
webqa_data[1].keys()
----------------------------------
dict_keys(['passages', 'question', 'id'])
```

passages中对应的是一个列表，列表中的元素是dict_keys([‘answer’, ‘passage’])

然后我自己写了一个脚本

为了增加可读性，把许可证和描述的关键字设置为空

```
from __future__ import absolute_import, division, print_function

import csv
import json
import os

import datasets
_CITATION =  “”    # 来自论文或arxiv
_DESCRIPTION = “”    # 任务描述
_HOMEPAGE = ”“    # 链接
_LICENSE = ”“    # 链接

_URLs = {    # 本地文件的路径
    'train': "./data/qatrain.json",
    'dev': "./data/qavalid.json"
}

```

下面是在模板中填写自己的数据集：

class Webqa(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Plain_text", version=VERSION, description="Plain text"),
    ]    # 对数据集的概述

    def _info(self):
        return datasets.DatasetInfo(
            # 这是将出现在“数据集”页面上的描述。
            description=_DESCRIPTION,
            # 这定义了数据集的不同列及其类型
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "passage": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),     # 这一部分定义了输出关键字的类型，和要输出的关键字
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # 这个方法用来下载/提取数据，依据configurations分割数据
        # 如果可能有几种配置(在BUILDER_CONFIGS中列出)，则用户选择的配置在self.config.name中

        # dl_manager is a datasets.download.DownloadManager 用来下载和抽取url
        # 它可以接受任何类型或嵌套的list/dict，并将返回相同的结构，也可以将url替换为本地文件的路径。
        # 默认情况下，将提取归档文件，并返回到提取归档文件的缓存文件夹的路径，而不是归档文件
        data_dir = dl_manager.download_and_extract(_URLs)
        print(data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples kwargs将会传参给_generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir["dev"]),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # 这个方法将接收在前面的' _split_generators '方法中定义的' gen_kwargs '作为参数。
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
        # 它负责打开给定的文件并从数据集生成元组(键，示例)
        # key是不重要的，更多的是为了传承

        # 这里就是根据自己的数据集来整理
        with open(filepath, encoding="utf-8") as f:
            data = json.loads(f)
            for questions in data:    # 读列表中的其中第一个字典
                id_ = questions['id'].strip()
                question = questions['question'].strip()
                passages = questions['passages']
                for passage_n in passages:
                    answer = passage_n['answer'].strip()
                    passage = passage_n['passage'].strip()

                    yield id_, {
                        "id": id_,
                        "question": question,
                        "passage": passage,
                        "answer":answer
                    }

## 结果演示

然后我们来演示这个读的效果

```python
from datasets import load_dataset
dataset = load_dataset("./qascript.py", data_files='./data/qatrain.json')
```

可以看到，在经过一番周折后，效果还是不错的。

```python
import datasets
from datasets.tasks import TextClassification


_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


class IMDBReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for IMDBReviews."""

    def __init__(self, **kwargs):
        """BuilderConfig for IMDBReviews.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IMDBReviewsConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Imdb(datasets.GeneratorBasedBuilder):
    """IMDB movie reviews dataset."""

    BUILDER_CONFIGS = [
        IMDBReviewsConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"text": datasets.Value("string"), "label": datasets.features.ClassLabel(names=["neg", "pos"])}
            ),
            supervised_keys=None,
            homepage="http://ai.stanford.edu/~amaas/data/sentiment/",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        archive = dl_manager.download(_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "test"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split("unsupervised"),
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train", "labeled": False},
            ),
        ]

    def _generate_examples(self, files, split, labeled=True):
        """Generate aclImdb examples."""
        # For labeled examples, extract the label from the path.
        if labeled:
            label_mapping = {"pos": 1, "neg": 0}
            for path, f in files:
                if path.startswith(f"aclImdb/{split}"):
                    label = label_mapping.get(path.split("/")[2])
                    if label is not None:
                        yield path, {"text": f.read().decode("utf-8"), "label": label}
        else:
            for path, f in files:
                if path.startswith(f"aclImdb/{split}"):
                    if path.split("/")[2] == "unsup":
                        yield path, {"text": f.read().decode("utf-8"), "label": -1}

```
