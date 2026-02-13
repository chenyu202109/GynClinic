"""Select guidelines and filter keywords and return only sources with keywords in main text"""

import os
import argparse
import shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from preprocess_logger import logger


from utils import read_jsonl, write_jsonl


# 实现过滤指定文件夹下的文件，并将过滤后的文件保存到指定文件夹下

# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #
# TODO REMOVE NAMES HERE
to_filter = ["meditron"]  # 文件夹下去过滤的文件名，如meditron.josnl

to_copy = ["asco", "esmo", "onkopedia_de", "onkopedia_en"] # 文件夹下去拷贝的文件名，如asco.jsonl

# 关键词列表 检测到这些关键字就可以将包含这些关键字的行给纳入到新的json文件中

keywords = [
    "adenocarcinoma",         # 腺癌
    "colorectal",             # 结直肠
    "colon cancer",           # 结肠癌
    "rectal cancer",          # 直肠癌
    "CRC",                    # 结直肠癌缩写
    "pancreatic cancer",      # 胰腺癌
    "pancreas cancer",        # 胰腺癌（另一种表达）
    "cholangiocellular",      # 胆管细胞
    "cholangiocarcinoma",     # 胆管癌
    "cholangio",              # 胆管相关
    "CCC",                    # 胆管细胞癌缩写
    "metastatic",             # 转移性
    "metastases",             # 转移灶
    "metastasis",             # 转移
    "HCC",                    # 肝细胞癌缩写
    "hepatocellular",         # 肝细胞
    "liver cancer"            # 肝癌
]

exclude = [] # 我们暂时没有需要排出的


# in_directory = ".../ProcessedData" # MODIFY THIS
in_directory = "D:\\code\\LLM\\agent\\LLM_RAG_Agent\\RAGent\\DSPY\\Data" # MODIFY THIS
out_directory = "complete_oncology_data" # MODIFY THIS

# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #


def filter_or_copy_data(in_directory, to_filter, keywords, exclude, to_copy, out_directory):

    all_paths = list(Path(in_directory).glob("*.jsonl"))
    filter_paths = [p for p in all_paths if any([x in str(p) for x in to_filter])]

    target_dir = Path(out_directory)
    target_dir.mkdir(exist_ok=True)

    # filter
    for file_path in filter_paths:

        keyword_counter = Counter()
        data = []
        unfiltered = read_jsonl(file_path)

        for article in tqdm(unfiltered):

            article_keywords = [
                keyword
                for keyword in keywords
                if keyword.lower() in article["clean_text"].lower()
            ]

            article_exclude = [exc for exc in exclude if exc.lower() in article["clean_text"].lower()]

            if article_keywords and not article_exclude: # article_exclude要为空 也就是没排除东西 非空即可进入下面的添加数据
                data.append(article) # 如果这一行存在关键字 我们就把他添加到数据中
                keyword_counter.update(article_keywords)

        logger.info(f"Filtered {len(unfiltered)} to {len(data)}")
        logger.info(f"Found the following # articles per keyword: {keyword_counter}")

        write_jsonl(target_dir / file_path.name, data)

    # copy 主要是拷贝其他文件 过滤文件和其他文件不能重复负责会报断言错误
    copy_paths = [p for p in all_paths if any([x in str(p) for x in to_copy])]
    assert set(filter_paths).isdisjoint(
        set(copy_paths)
    ), "Some files are in both to_filter and to_copy"

    for copy_path in copy_paths:
        shutil.copy(copy_path, target_dir / copy_path.name)

    logger.info(
        f"Filtered {len(filter_paths)} files and copied {len(copy_paths)} files to {target_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Process some oncology data.")

    parser.add_argument(
        "--to_filter",
        nargs="+",
        default=to_filter,
        help="List of sources to filter out, except for specific allowed sources.",
    )
    parser.add_argument(
        "--to_copy",
        nargs="+",
        default=to_copy,
        help="List of sources to always include without filtering.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=keywords,
        help="List of keywords to consider in the data processing.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=exclude,
        help="List of keywords to exclude in the data processing.",
    )
    parser.add_argument(
        "--in_directory",
        type=str,
        default=in_directory,
        help="Input directory containing the data to be processed.",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        default=out_directory,
        help="Output directory where the processed data will be saved.",
    )

    args = parser.parse_args()

    filter_or_copy_data(
        in_directory=args.in_directory,
        to_filter=args.to_filter,
        keywords=args.keywords,
        exclude=args.exclude,
        to_copy=args.to_copy,
        out_directory=args.out_directory,
    )

if __name__ == "__main__":
    main()
