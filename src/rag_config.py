import os
from dataclasses import dataclass, field
from typing import List, Dict

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from dotenv import load_dotenv
load_dotenv()

# openai修改url和key
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-3-large",api_base=os.environ.get("OPENAI_BASE_URL")
)

# might fail splitting metadata if values are set too small 
# 处理不同子块大小 单位为token
def get_sub_chunk_sizes():
    """Get the default sub chunk sizes for each source."""
    return {
        "esmo": [128, 256, 512],
        "asco": [128, 256, 512],
        "meditron": [512], # 先前仅是512  1024/512 = 2 个innodes
        "onkopedia_de": [128, 256, 512],
        "onkopedia_en": [128, 256, 512],
    } # MODIFY THIS

# MODIFY THIS
"""
    default_client_path: Chroma DB 客户端的默认路径，通常用于存储向量数据库。
    collection_name: 在 Chroma DB 中创建的集合名称，类似于数据库表名。
    distance_metric: 检索时使用的距离度量方式，比如“cosine”（余弦相似度）、“l2”（欧氏距离）、“ip”（内积）。
    data_paths: 存放原始数据（如 JSONL 文件）的目录路径。
    default_chunk_size: 文本分块的默认大小（以 token 数计），用于分割长文本。
    default_chunk_overlap: 分块时的重叠部分大小，保证上下文连续性。 文本分块时，相邻块之间的重叠 token 数，保持上下文连贯。
    default_sub_chunk_sizes: 文本块进一部分为子分块，子分块的默认大小列表，用于进一步细分文本块。
    retrieve_top_k: 每次检索时返回的文档数量。
    rerank_top_k: 第一次筛选重排后 重排序后保留的文档数量。
    final_rerank_top_k: 最终重排序后 保留的文档数量。
    check_citations: 是否启用 RAG 的引用检查功能。 为实现可追溯/可解释的RAG，检查返回内容是否有真实数据来源，用于提升结果可靠性。 打开吧默认是关闭的 可能有消耗
    default_embed_model: 默认的文本嵌入模型名称。
    default_rerank_model: 默认的重排序模型名称（这里是英文模型）。
    default_llm_name: 默认的大语言模型名称（如 gpt-4o）。
    llm_temperature: LLM 生成文本的温度参数，影响输出的多样性。
    llm_max_tokens: LLM 生成文本的最大 token 数。
    reference_node_chunk_size: 参考节点分块大小。 通常与default_chunk_size区分，供不同类型知识源使用。
    reference_node_chunk_overlap: 参考节点分块重叠大小。
    sub_chunk_sizes: 针对不同情况的子分块大小配置（通常是个字典）。
    embedding_function: 嵌入函数，用于将文本转为向量。
    __post_init__: dataclass 初始化后自动调用的方法，这里用于设置 LLM 参数和参考节点参数。

    注意 default_chunk_size 和 reference_node_chunk_size 的区别：
    - default_chunk_size: 用于处理原始数据的默认分块大小。 长文本 所以chunk_size 通常较大。
    - reference_node_chunk_size: 用于处理参考节点的分块大小，通常用于 RAG 模型的知识库构建。 
"""
@dataclass
class RAGConfig:
    """Default configurations for RAG model."""

    default_client_path: str = "./chroma_db_oncology"  # path to the Chroma DB client
    collection_name: str = "oncology_db"  # Name of the collection to be created
    distance_metric: str = "cosine"  # "cosine" or "l2" or "ip"
    data_paths: str = "complete_oncology_data"  # directory where the data (*.jsonl) is stored 上次默认deduplicated_oncology_data
    default_chunk_size: int = 1024 # default spit size for the text tokens
    default_chunk_overlap: int = 50 # how much overlap between splits
    default_sub_chunk_sizes: List[int] = field(default_factory=lambda: [128, 256, 512])
    retrieve_top_k: int = 10 # how many documents to retrieve at each collection.get call
    rerank_top_k: int = 5 # how many documents after reranking
    final_rerank_top_k: int = 30 # how many documents after final reranking
    check_citations: bool = False # wheter to use RAG citation checking

    default_embed_model: str = "text-embedding-3-large"
    default_rerank_model: str = "rerank-english-v3.0" # 这里的重排序模型是英文的 先前用的rerank-english-v2.0
    default_llm_name: str = "gpt-4o"  # rag默认的LLM
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    reference_node_chunk_size: int = 512
    reference_node_chunk_overlap: int = 20

    sub_chunk_sizes: Dict[str, List[int]] = field(
        default_factory=get_sub_chunk_sizes
    )

    embedding_function: EmbeddingFunction = embedding_function

    def __post_init__(self):
        self.llm_kwargs = dict(
            temperature=self.llm_temperature, max_tokens=self.llm_max_tokens
        )
        self.ref_node_conf = dict(
            chunk_size=self.reference_node_chunk_size,
            chunk_overlap=self.reference_node_chunk_overlap,
        )
