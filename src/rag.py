import json
import re
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Union

import chromadb
import dspy
import nltk
from dspy.predict import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.retrieve.qdrant_rm import QdrantRM
from icecream import ic
from llama_index.core.schema import (
    Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank # 这里
from llama_index.core.retrievers import RecursiveRetriever, VectorIndexRetriever
from llama_index.core.schema import IndexNode, NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

nltk.download("punkt")
from ast import literal_eval
from pathlib import Path
from typing import Literal, Optional

import openai
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from citations_utils import *
from llama_index.core.schema import NodeWithScore
from loguru_logger import logger
from nltk.tokenize import sent_tokenize
from rag_config import RAGConfig
from rag_utils import MetadataFields, deduplicate, is_list_valid, length_check, to_list
from signatures import *
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from utils import defaults, exists, file_len, read_jsonl_in_batches
from rag_config import RAGConfig
rag_config = RAGConfig()

PathLike = str | Path

# DEFAULT_LLM_NAME = "gpt-4o"  # Default LLM model name

DEFAULT_EMBED_MODEL = "text-embedding-3-large"

__all__ = ["CollectionWrapper", "RAG", "RAGLoader"]

# 配置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

def clean_string(string):
    return string.encode("utf-8", "replace").decode("utf-8", "replace")


def yield_nodes_in_batches(nodes, batch_size: int = 1000):
    """Yield nodes in batches."""
    for i in range(0, len(nodes), batch_size):
        yield nodes[i : i + batch_size]


def wait_rand_exp_retry_dec(
    max_retries: int = 7,
    min_secs: int = 20,
    max_secs: int = 120,
) -> Callable[[Any], Any]:
    w = wait_random_exponential(min=min_secs, max=max_secs)
    s = stop_after_attempt(max_retries)

    return retry(
        reraise=True,
        stop=s,
        wait=w,
        retry=(
            retry_if_exception_type(
                (
                    openai.APITimeoutError,
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APIStatusError,
                )
            )
        ),
        # before_sleep=logger.warning(f"Retrying after OpenAI API Refusal."),
    )

embed_retry_dec = wait_rand_exp_retry_dec()

class CollectionWrapper:

    """
    Lightweight wrapper around the ChromaDB collection.
    Handles the entire pipeline for adding documents from a file to the collection.
    There are no convenience functions for update, deletion etc, because this can be done with much less boilerplate code on the chromadb collection directly.
    """

    def __init__(
        self,
        *,
        chroma_client: chromadb.HttpClient = None,
        chroma_collection: Collection = None,
        rag_config: Optional[RAGConfig] = None,
        sub_chunk_sizes: List[int] = None,
    ) -> None:
        if not exists(rag_config):
            from rag_config import RAGConfig

            rag_config = RAGConfig()

        self._chroma_client = chroma_client
        self._collection = chroma_collection

        self._rag_config = rag_config

        self.node_parser = SentenceSplitter(
            chunk_size=rag_config.default_chunk_size, # 1024
            chunk_overlap=rag_config.default_chunk_overlap, # 50
        )
        # sub_chunk_sizes is a list of integers, e.g. [128, 256, 512]
        # SentenceSplitter will create a list of SentenceSplitter instances with different chunk sizes
        # [SentenceSplitter(128,50),SentenceSplitter(256,50),SentenceSplitter(512,50)]
        self.sub_node_parsers = [
            SentenceSplitter(
                chunk_size=c, chunk_overlap=rag_config.default_chunk_overlap
            )
            for c in sub_chunk_sizes
        ]

    def add_documents(self, data_file_path: Union[PathLike, List[PathLike]] = None):
        """Add documents from a file or list of files."""
        # 判断传入的是单个路径还是列表。如果是单个路径，会把它转成列表，方便后续统一处理。
        if isinstance(data_file_path, PathLike):
            data_file_path = [data_file_path]

        for file_path in data_file_path:
            logger.info(f"Adding documents from {file_path}.")

            with tqdm(total=file_len(file_path), unit="articles") as pbar: # 读到了3782条数据
                for total_count, data_batch in read_jsonl_in_batches( # 是个迭代器 逐条返回？
                    file_path, batch_size=100  # 100条（100本书）可能有点多 数据量小的话可以修改一下
                ):  
                    logger.info(
                        f"Currently embedding at position {total_count} of items."
                    )

                    # 数据转换
                    documents = self._create_documents(data_batch) # 返回[Document(text=..., metadata={...}), ...]
                    # 将文档解析为节点的层次结构。"
                    nodes = self._convert_documents_to_nodes(documents) # 把Document对象进一步切分成更小的“节点”（node），用于后续嵌入

                    # 分批嵌入上传
                    for nodes_batch in yield_nodes_in_batches( # 迭代器返回
                        nodes, batch_size=1000
                    ):  
                        try:
                            # 进行嵌入和上传到Chroma数据库 self._collection.add(text, metadatas, ids)
                            # 我们的embedding Function 在这个数据库里面进行了集成
                            self.create_and_add_embeddings(nodes_batch)  # batch_size=1000 
                            # 得到的嵌入向量会与原始文本块、元数据一起写入Chroma数据库。
                            # 这样后续可以通过向量检索实现语义搜索。
                        except Exception as e:  # TODO: Better error handling
                            logger.error(f"Error uploading nodes: {e}.")
                            raise e

                    pbar.update(len(data_batch)) # 更新进度条

            logger.info(
                f"Added {len(nodes)} nodes to the collection. Currently collection length is {self._collection.count()} nodes."
            )

        logger.info("Upload complete.")

    # 能够在这里就把embeding做了，然后添加到chromadb，就里面不用做了
    @embed_retry_dec
    def create_and_add_embeddings(self, nodes_batch: List) -> None:
        # 先加入embeddings试一试
        # 给每一条数据做embeddings=[embedding_function(node.text) for node in nodes_batch] 太长时间 就不能批量了
        embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-3-large",api_base=os.environ.get("OPENAI_BASE_URL"))
        self._collection.add( # 当有1000个node时候，才会做此操作，这个操作也是批量操作
            documents=[node.text for node in nodes_batch],
            metadatas=[node.metadata for node in nodes_batch],
            ids=[node.id_ for node in nodes_batch]
        ) # chromadb好像集结了embeding函数,这里不提供，因为可以实现默认的批量将这个1000个node做embeding，不然还要先做好embeding过程

    def _create_documents(self, data: List[Dict] = None): # 传入batch_size个数据
        """Create a list of documents from a list of articles.
        Parameters:
            - data: A list of dictionaries where each dictionary represents an article.
        Returns:
            - documents: A list of Document objects.
        """
        documents = []
        for article in data:
            node_metadata = MetadataFields[article["article_source"].upper()].value # 元数据映射
            # node_metadata 为 一个集合，包含了文章的元数据字段 (article_source,unique_article_uuid,id,source,title,url)
            document = Document( # 包含要检索的文本 元数据 和 其他信息 # 这里是一篇文章组个一个Document
                text=article["clean_text"],
                metadata={
                    key: article[key]
                    for key in node_metadata
                    if article.get(key, None) is not None
                }, # {article_source: article["article_source"], unique_article_uuid: article["unique_article_uuid"], ...}
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
            )

            documents.append(document)

        return documents

    def _convert_documents_to_nodes(self, documents: List[Document]):
        """Parse documents into a hierarchical structure of nodes.它是生成向量的基础单元"""

        # 是先要创建首节点？
        logger.info(f"Converting {len(documents)} documents to nodes.")
        base_nodes = self.node_parser.get_nodes_from_documents(documents, show_progress=True)# sentensplit(1024,50) 也就是chunk_size=1024 chunk_overlap=50
        """ node结构特征
        node.text：文本内容
        node.metadata：元数据（如文档来源、标题等）
        node.id_：唯一标识
        node.relationships：与前后节点的关系
        node.start_char_idx/end_char_idx：在原文中的起止位置
                """
        all_nodes = []

        for base_node in base_nodes: # 再将每个1024的文本切分成更小的chunk_size
            for n in self.sub_node_parsers: # 文本到处理token主要在这里 [SentenceSplitter(128,50),SentenceSplitter(256,50),SentenceSplitter(512,50)] 
                # 原来是每个SentenceSplitter都存一份 
                sub_nodes = n.get_nodes_from_documents([base_node], show_progress=True)
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            original_node = IndexNode.from_text_node(base_node, base_node.node_id)  # 再把原始的存一份
            all_nodes.append(original_node)

        return all_nodes

class RAG(dspy.Module):
    def __init__(
        self,
        retrieve_top_k: int = rag_config.retrieve_top_k,
        rerank_top_k: int = rag_config.rerank_top_k,
        final_rerank_top_k: int = rag_config.final_rerank_top_k,
        default_rerank_model: str = rag_config.default_rerank_model,
        ref_node_conf: Dict = None, # 这里默认
        check_citations: bool = True, # 默认是False但是我这里把他设置为True
    ) -> None:
        super().__init__()

        # RAGConfig
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.max_n = final_rerank_top_k
        self.ref_node_conf = defaults(
            ref_node_conf, dict(chunk_size=512, chunk_overlap=20)
        )

        # DSPY modules
        self.subquery_gen = dspy.ChainOfThought(Search)
        self.ask_for_more = dspy.ChainOfThought(RequireInput)
        self.generate_answer_strategy = dspy.ChainOfThought(AnswerStrategy)
        self.generate_cited_response = dspy.Predict(GenerateCitedResponse)
        self.generate_suggestions = dspy.Predict(Suggestions)

        self.rerank_model_name = default_rerank_model
        self.check_citations = check_citations

    def forward(
        self,
        question: str = None,
        patient_context: Optional[str] = None,
        tool_results: Optional[str] = None,
        agent_tools: List[str] = None,
        rerank_model: CohereRerank = None,
        name: str = None
    ):
        """Forward pass"""

        assert exists(question), "Question must be provided."

        patient_context = defaults(
            str(patient_context), "There is no relevant patient context."
        )  # instead we could also instruct the InputField(desc="Ignore if N/A")
        tool_results = defaults(str(tool_results), "No tools were used.")  # Same.
        # print("传过来到工具结果================================》\n")
        # print(tool_results)

        agent_tools = defaults(agent_tools, [])

        subqueries = self.subquery_gen(
            question=question, context=patient_context, tool_results=tool_results
        ).searches

        logger.info(f"Generated Subqueries: {subqueries}")

        flagged_invalid = False
        while not is_list_valid(subqueries):
            if not flagged_invalid:
                flagged_invalid = True
                logger.warning("Subqueries are not valid. Trying to fix them.")

            dspy.Suggest(
                is_list_valid(subqueries),
                f"Assert that your searches can be interpreted as valid python list using eval / ast.literal_eval.",
                target_module=Search,
            )

        subqueries = to_list(subqueries)
        
        assert isinstance(subqueries, List), "Subqueries must be a list of strings."

        context: List[RerankResult] = []

        # 好像设置了8个子查询
        # retrieve for every subquery
        count = 1
        for idx, search in enumerate(subqueries, start=1):
            
            logger.info(f"Searching # {idx}, Search: {search}")

            # 进行检索 选出 个 还有个文件chroma_db_retriever 这个开始放到了的Dspy的配置文件中 就是执行的这个文件
            passages = dspy.Retrieve(k=self.retrieve_top_k)(search).passages
            
            # print(passages)
            # print("====================================================")
            if count == 2: # 重排序模型不能一分钟接受10个 所以要降低 我做了引用检查 所以可能会多次进行访问向量库查询
                count = 1
                time.sleep(5)

            # 重排序 为10个 肯定有个重排序模型
            #TODO 这里先注释掉 因为我的rerankmodel的排序次数我怕会不足
            max_retries = 5
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    passages = rerank_model.rerank(
                        query=search,
                        documents=passages,
                        top_n=self.rerank_top_k,
                        model=self.rerank_model_name,
                    )
                    break  # 成功则跳出循环
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"连接失败，正在进行第 {attempt + 1} 次重试... 错误: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 增加等待时间
                    else:
                        raise e  # 最后一次尝试也失败，抛出异常
 
            # print(passages)
            # print("====================================================")

            # 去重且保证原有的顺序
            context = deduplicate(context + [p for p in passages]) #TODO 其实这个去重工作应该放到所有子查询结束之后再来做
            count  = count + 1

        # context = rerank_model.rerank(
        #     query=patient_context + "\n" + tool_results + "\n" + question,
        #     documents=[c.document for c in context],
        #     top_n=self.max_n,
        #     model=self.rerank_model_name,
        # )

        # logger.info(f"# Context nodes after Rerank: {context}")
        
        # 停一分钟
        # time.sleep(20)


        # 返回所有带编号的新节点列表。如：根据 Source 2 == 引用[2]
        context_nodes = create_reference_nodes(context, self.ref_node_conf)

        logger.info(
            f"# 参考节点长度==》Context nodes after splitting into reference nodes: {len(context_nodes)}"
        )
        
        # 生成参考文献列表 这里直接放上去可能不那么好看，可以进行下数据清洗
        for i in context_nodes:
            n = i.document["text"]
            with open(f"./results/reference/{name}.txt", "a", encoding="utf-8") as f:
                f.write(n + "\n\n")
            # logger.info(f"Context node: {n}")

        # 这里就有带着Source x的样式的文本 之后在citations_check(pred.response) 将引用进行了转化为[x]
        medical_context = [n.document["text"] for n in context_nodes]

        # print("================这是medical_context===================")
        # print(medical_context)

        agent_tools = "These tools are available to you:" + str(
            [tool["description"] for tool in agent_tools]
        )

        data = dict(
            context=medical_context, # 知识库中的信息
            patient="Patient:\n" + patient_context,
            tool_results="Tool:\n" + tool_results,
            tools=agent_tools, # tool的描述
            question="Question:\n" + question,
        )

        answer_strategies = self.generate_answer_strategy(**data) # CoT 这是AnswerStrategy Prompt，这里综合了所有的信息知识库、工具结果、病人和问题的结合，也是为最终答案奠定基础 这个是产生答案的关键
        logger.info("CoT to structure the answer: {}", answer_strategies)

        # data.pop("context")
        # 这里才用到了tools的信息 
        # ask_for_more = self.ask_for_more(**data) # 这里去除了病人的知识库信息，通过RequireInpu Prompt 再次概述相关工具的使用额外建议 这里给放到了最后建议生成
        # logger.info("Ask for more information: {}", ask_for_more)
        

        pred = self.generate_cited_response( # 通过GenerateCitedResponse Prompt 生成引用 []数据
            strategy=answer_strategies.response,  # + ask_for_more.response,
            context=medical_context,
            patient="Patient:\n" + patient_context,
            tool_results="Tool:\n" + tool_results,
            question="Question:\n" + question,
        )

        # pred = dspy.Prediction(
        #     response=pred.response, context=medical_context, context_nodes=context_nodes
        # )

        # 这里先设置为False 先不进行引用检查
        self.check_citations = False 
        if self.check_citations:
            dspy.Suggest(
                citations_check(pred.response), # 确保每1到2个句子都有引用
                f"Make sure every 1-2 sentences has correct citations. If any 1-2 sentences have no citations, add them in 'text... [x].' format.",
                target_module=GenerateCitedResponse,
            )

            _, invalid_responses = citation_faithfulness(pred, None)
            if invalid_responses:
                invalid_pairs = [
                    (
                        output["text"],
                        output.get("context"),
                        output.get("error"),
                        output.get("rationale"),
                    )
                    for output in invalid_responses
                ]

                logger.warning(
                    "Currently having: {} invalid pairs of response <-> references.",
                    len(invalid_pairs),
                )

                for _, context, error, rationale in invalid_pairs:
                    msg = (
                        f"Make sure your output is based on the following context: '{context}'."
                        if exists(context)
                        else f"Make sure your output does not produce the following error: '{error}'."
                    )
                    if exists(rationale):
                        msg += f"The mistake you made was: {rationale}"
                        logger.warning(
                            "The model made the following mistake when checking citations: {}",
                            msg,
                        )

                    dspy.Suggest(
                        len(invalid_pairs) == 0,
                        msg,
                        target_module=GenerateCitedResponse, 
                    )
            # Check citations

        # suggestions = self.generate_suggestions( # Suggestions Prompt 这里生成的建议是要求给工具更多的信息，如具体的分子说明，他能提供靶向治疗方案，反正就是突出工具的实用性
        #     response=pred.response, recommendations=ask_for_more.response # res+工具额外建议 = 建议
        # )

        # final_response = str(pred.response) + "\n\n" + str(suggestions.suggestions) # res + 建议 = final answer
        final_response = str(pred.response)
        
        pred = dspy.Prediction( # 最后将final answer + 知识库[] + 知识库 Source 进行结合产生最终答案 
            response=final_response,
            context=medical_context,
            context_nodes=context_nodes,
        )
        
        logger.info("Final response: {}", pred.response)

        return pred

def load_rag(
    retrieve_top_k: int = None,
    rerank_top_k: int = None,
    final_rerank_top_k: int = None,
    default_rerank_model: str = None,
    ref_node_conf: Dict[str, List[int]] = None,
    check_citations=True, # 默认是False 先设置为True
) -> RAG:
    rag_config = RAGConfig()

    rag = RAG(
        retrieve_top_k=defaults(retrieve_top_k, rag_config.retrieve_top_k),
        rerank_top_k=defaults(rerank_top_k, rag_config.rerank_top_k),
        final_rerank_top_k=defaults(final_rerank_top_k, rag_config.final_rerank_top_k),
        default_rerank_model=defaults(
            default_rerank_model, rag_config.default_rerank_model
        ),
        ref_node_conf=defaults(ref_node_conf, rag_config.ref_node_conf),
        check_citations=defaults(check_citations, rag_config.check_citations),
    )

    rag = assert_transform_module(rag.map_named_predictors(Retry), backtrack_handler)

    return rag
