import argparse
import os
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin
import argparse
import os
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.tools import FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from loguru_logger import logger
from skimage import io, transform
import torch.nn.functional as F
import uuid

from llama_index.core import download_loader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from llama_index.core.storage.storage_context import StorageContext   
from llama_index.core import KnowledgeGraphIndex   
from llama_index.core.graph_stores import SimpleGraphStore      
from pyvis.network import Network   

from segment_anything import sam_model_registry

import json
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.tools import FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from loguru_logger import logger
from skimage import io, transform
import torch.nn.functional as F
import uuid

from llama_index.core import download_loader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from llama_index.core.storage.storage_context import StorageContext   
from llama_index.core import KnowledgeGraphIndex   
from llama_index.core.graph_stores import SimpleGraphStore      
from pyvis.network import Network   

from segment_anything import sam_model_registry

DEFAULT_MODEL_NAME = "gpt-4o"
# DEFAULT_MODEL_NAME = "gpt-5-mini-2025-08-07"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


join = os.path.join
PathLike = str | Path

__all__ = ["openai_agent_tools", "tool_names"]

tool_names = {
    "google_search": "Google Search",
    "query_pubmed": "PubMed Query",
    "get_check_report":"Get Patient's Results of Examinations Items"
}

registered_functions = []


def register(func):
    """Decorator func to register a function as an AI agent tool."""
    registered_functions.append(func)
    return func
    

# This search_query needs to be specific and tailored to the patient's condition, chief complaints, and unique features of the condition to get the best results from google search.

@register
def google_search(patient_id: str, query: str) -> str:
    """Run a classical google search and return the top 10 results. Useful when timely new / up-to-date information is needed.
    Args:
    - patient_id (str): Unique identifier of the patient. Also it is System ID. You could find it on patient context.
    - search_query (str): Combining the given questions with the patient's background information, chief complaint, medical history, and examination results for inquiry, it must be a coherent and fluent sentence.
    Returns:
    - response (str): A structed text response that return 10 most relevent results of search_query.
    """
    ## 测试
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_BASE_URL")
    from openai import OpenAI
    # client = OpenAI()
    # 初始化客户端
    client = OpenAI(
        base_url=url,
        api_key=key,
        timeout=300
    )
    response = client.chat.completions.create(
        messages=[
        {"role": "system", "content": "You are a helpful assistant that can search the web for information. "},
        {"role": "user", "content": f"Please query for this question: {query}, and then return it to us with detailed results"},
        ],
        # 这里我们先用gpt-4o-mini-search-preview平替
        model="gpt-4o",
        seed=42,
        temperature=0.3,
    )
    model_output = response.choices[0].message.content
    print("Google 查询结果: ")
    print(model_output)
    return model_output
    #############################################################测试结束

#########################原生的搜索不要钱#########################################################
    # from llama_index.agent.openai import (
    #     OpenAIAgent,
    # )  # TODO: this is legacy in the new llama index
    
    # from llama_index.llms.openai import OpenAI
    # load_dotenv()
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # google_engine = os.getenv("GOOGLE_SEARCH_ENGINE")

    # tool_spec = GoogleSearchToolSpec(key=google_api_key, engine=google_engine, num=10)
    # resp = tool_spec.google_search(query=query)

    # with open(f"./results/reference/{patient_id}-tool.txt", "a", encoding="utf-8") as f:
    #     f.write(resp + "\n")

    # # TODO: 根据文章的链接，依次对链接进行网页爬虫，然后将返回的内容进行数据清洗，包装为content
    # return resp

#######################################################################################
    # TODO: refactor; this currently initializes another OpenAI agent ...
    # try:
    #     agent = OpenAIAgent.from_tools(
    #     tool_spec.to_tool_list(), llm=OpenAI(model=DEFAULT_MODEL_NAME),
    #     system_prompt="""You are a helpful assistant that can search the web for information. 
    #                     Please provide detailed and structured information as soon as possible.
    #                     Please search for detailed answers. 
    #                     For treatment methods and medication choices, be sure to specify their names, maintain accuracy and relevance, and be detailed. 
    #                     Pleasing remember, for surgical intervention, radiation therapy, and chemotherapy, please be sure to confirm the specific amount of radiation and chemotherapy, the specific drug selection, and the specific location of the surgery."""
    # )
    # except:
    #     pass

    # logger.info(f"Running google search for query: {query}")
    # response = agent.chat(query)
    # logger.info(f"Got google search reply: {response}")
    # return response
################################################################################################
    # from llama_index.core.agent.workflow import FunctionAgent
    # logger.info(f"Running google search for query: {query}")
    # print("############")
    # print(tool_spec.to_tool_list())
    # agent = FunctionAgent(
    #     tools=tool_spec.to_tool_list(),
    #     llm=OpenAI(model=DEFAULT_MODEL_NAME),
    #     system_prompt="You are a helpful assistant that can search the web for information.Please provide detailed and complete information.",
    # )
    # response = agent.run(user_msg=query)
    # # print(str(response))
    # resp = str(response)
    # logger.info(f"Got google search reply: {resp}")
    # return resp


    #########################这里使用jina搜索
    # import requests
    # url = f'https://s.jina.ai/?q={query}'
    # headers = {
    #     'Accept': 'application/json',
    #     'Authorization': 'Bearer jina_b20b5f1474a54817aa205ea79615196b5jYPUKMgD9Olqag-EXNj8GEcnFhP',
    #     'X-Respond-With': 'no-content'
    # }

    # response = requests.get(url, headers=headers)
    # res = "# Google Serach Results:\n\n"
    # resp_json = response.json()
    # if resp_json['code'] == 200:
    #     data = resp_json['data']
    #     for item in data:
    #         title = item['title']
    #         description = item['description']
    #         res1 = f"## {title}\n{description}\n\n"
    #         res += res1

    # else:
    #     print("Google search failed！")
    # # print(res)
    # return res
    #########################这里使用jina搜索


# For instance: ["colorectal cancer k-ras", "k-ras, n-ras, braf CRC", "k-ras targeted therapy colorectal cancer"]
# ["cervical cancer HPV type 31", "endometrial cancer hyperplasia", "HPV-associated squamous cell carcinoma metastasis",p53,p16,PD-L1]
# This example is for reference format only. The specific content inside needs to be determined based on the patient's real situation and tool results
# The response to the query must be very detailed.
@register
def query_pubmed(pubmed_search_terms: List[str], query: str) -> str:
    """
    Performs a PubMed search for articles using a list of initial search terms and a final query.
    Only the first three pubmed_search_terms are used to fetch articles from PubMed, so they should be the most relevant.

    Parameters:
        - pubmed_search_terms (List[str]): A list of initial search terms to use for fetching articles from PubMed. Should not be longer than three strings.Please combine with the patient's condition.
            For instance: ["Symptoms of Endometrial Hyperplasia","Differential diagnosis of abnormal uterine bleeding","endometrial cancer hyperplasia"].
        - query (str): Combining with the patient's condition query to fetch additional articles and to perform the final search against the indexed documents.

    Returns:
        - Response: A detailed response that answers the query based on the retrieved pubmed documents.
    """
    
    # import os
    # from dotenv import load_dotenv
    # load_dotenv()
    # key = os.environ.get("OPENAI_API_KEY")
    # url = os.environ.get("OPENAI_BASE_URL")
    # print(key)
    # print(url)
    
    # PubmedReader = download_loader("PubmedReader")
    # loader = PubmedReader()

    # documents = []
    # try: # 暂时仅检索5篇文章
    #     for pubmed_search_term in pubmed_search_terms[:2]: # 作为测试我就先放入一个关键字进行搜索
    #         # only take the first 3 search terms
    #         documents.extend(loader.load_data(search_query=pubmed_search_term,max_results=1))
    #     documents += loader.load_data(search_query=query,max_results=1)
    # except:
    #     print("发生错误! 痛失检索一篇文章。。。")
    #     pass
    

    # # service_context = ServiceContext.from_defaults(
    # #     llm=OpenAI(model=DEFAULT_MODEL_NAME),
    # #     embed_model=OpenAIEmbedding(model="text-embedding-3-large"),
    # # )

    # # pubmed_index = VectorStoreIndex.from_documents(
    # #     documents=documents,
    # #     service_context=service_context,
    # # )

    # # save_dir = f"./pubmed_index_{uuid.uuid4()}"
    # # pubmed_index.storage_context.persist(persist_dir=save_dir)

    # # logger.info(
    # #     f"Saved pubmed index with search terms {pubmed_search_terms} and query {query} to {save_dir}."
    # # )

    # # return pubmed_index.as_query_engine().query(query).response
    
    # ########################################知识图谱########################################
    # llm = OpenAI(model=DEFAULT_MODEL_NAME,timeout=300.0,
    #              system_prompt="You are a search assistant based on the PubMed knowledge graph, and your task is to answer user queries about PubMed. Users may ask some questions about treatment methods and medication choices. Please search for detailed answers. For treatment methods and medication choices, be sure to specify their names, maintain accuracy and relevance, and be detailed. Pleasing remember, for surgical intervention, radiation therapy, and chemotherapy, please be sure to confirm the specific amount of radiation and chemotherapy, the specific drug selection, and the specific location of the surgery. Don't break down, provide a structured and detailed statement.")
    # em_model =OpenAIEmbedding(model=DEFAULT_EMBED_MODEL)

    # Settings.llm = llm
    # Settings.embed_model = em_model
    # Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    # Settings.num_output = 256
    # Settings.context_window = 3900

    # #  接下来我们不使用向量数据库查询，而是使用知识图谱来进行查询
    # # 设置存储上下文   
    # graph_store = SimpleGraphStore()   
    # storage_context = StorageContext.from_defaults(graph_store=graph_store)      
    # pubmed_index = KnowledgeGraphIndex.from_documents(
    #     documents=documents,
    #     max_triplets_per_chunk=3,
    #     storage_context=storage_context,
    #     embed_model=em_model,
    #     include_embeddings=True,
    #     show_progress=True
    # )   
    # # max_triplets_per_chunk：指的是每个文本块中能够提取的最大三元组数量。降低这个数值可以提升处理效率，因为它减少了需要处理的三元组数量。
    # # include_embeddings：用于决定索引中是否包含嵌入项。默认设置为False，因为生成嵌入项在计算上可能相当耗费资源。
    # # 10个documents 要是256chunksize--》1024      1024-->91

    # # myid = uuid.uuid4()
    # # save_dir = f"./LOGS/query_pubmed/pubmed_index_{myid}"
    # # pubmed_index.storage_context.persist(persist_dir=save_dir)
    # # print(f"Saved pubmed index with search terms {pubmed_search_terms} and query {query} to {save_dir}.")

    # # g = pubmed_index.get_networkx_graph()   
    # # net = Network(notebook=True, cdn_resources="in_line", directed=True)   
    # # net.from_nx(g)   
    # # net.save_graph(save_dir+".html")   # 记得在write时候 设置encoding="utf-8"
    
    # # from IPython.display import HTML, display   
    # # HTML(filename="rag_graph.html")   

    # myquery = f"Please conduct a detailed search of the knowledge graph and query for this question: {query}, and then return it to us with detailed results"
    # # # 到知识图谱中进行查询 并且返回出qurry后的结果
    # return pubmed_index.as_query_engine(llm).query(myquery).response
    ########################################知识图谱########################################
    # # 测试
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_BASE_URL")
    from openai import OpenAI
    # client = OpenAI()
    # 初始化客户端
    client = OpenAI(
        base_url=url,
        api_key=key,
        timeout=300
    )
    response = client.chat.completions.create(
        messages=[
        {"role": "system", "content": "You are a search assistant based on the PubMed knowledge graph, and your task is to answer user queries about PubMed. Users may ask some questions about treatment methods and medication choices. Please search for detailed answers. For treatment methods and medication choices, be sure to specify their names, maintain accuracy and relevance, and be detailed. Pleasing remember, for surgical intervention, radiation therapy, and chemotherapy, please be sure to confirm the specific amount of radiation and chemotherapy, the specific drug selection, and the specific location of the surgery. Don't break down, provide a structured and detailed statement."},
        {"role": "user", "content": f"Please query for this question: {query}, and then return it to us with detailed results"},
        ],
        model="gpt-4o",
        max_tokens=4096,
        seed=42,
        temperature=0.3,
        
    )
    model_output = response.choices[0].message.content
    print("PubMed 查询结果: ")
    print(model_output)
    return model_output
    #############################################################测试结束


def call_llm_with_retry(prompt_template: str) -> Dict[str, Any]:
    """
    封装 API 调用，包含重试逻辑。
    用于从完整的检查报告中筛选出医生需要的项目。
    """
    # 加载环境变量 (确保 .env 中有 OPENAI_API_KEY)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("OPENAI_BASE_URL")
    from openai import OpenAI

    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=url,
        api_key=key,
        timeout=300
    )

    # 配置常量
    MAX_RETRIES = 3
    BASE_DELAY = 2  # 秒
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful medical data assistant. Your task is to filter a medical report JSON based on a list of requested items. You must perform semantic matching (e.g., 'Blood work' matches 'CBC'). Return ONLY valid JSON."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=0.3, # 设置为0保证结果确定性
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                tqdm.write(f"\n[Error] API调用失败: {e}")
                # 如果失败，为了不阻断流程，返回一个空字典或者错误提示
                return {"error": "Failed to extract data via LLM", "details": str(e)}
            
            wait_time = BASE_DELAY * (2 ** (retries - 1))
            tqdm.write(f"[Warning] API Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

@register
def get_check_report(patient_id: str, check_list_to_fuzhu: List[str]) -> str:
    """
    Function that gets a check report including imaging examinations and blood routine tests for patients.
    The doctor requests additional auxiliary examination items to further support the patient's diagnostic basis.
    Parameters:
    - patient_id (str): Unique identifier of the patient. Also it is System ID. You could find it on patient context.
    - check_list_to_fuzhu (List[str]): This is a list of auxiliary examination items that gynecologists need to supplement.
    Returns:
    - response (str): The examination content corresponding to the checklist that needs to be returned.
    """
    
    model_name = "ref-bu70"
    file_path = 'Imaging/data.json'
    # model_name = "Baichuan-M2-32B"
    
    
    print("==================进来检查了")
    print(check_list_to_fuzhu)

    # 1. 记录请求日志
    os.makedirs(f"results/examinations/{model_name}", exist_ok=True) # 确保目录存在
    with open(f"results/examinations/{model_name}/{patient_id}.txt", 'a', encoding='utf-8') as file:
        file.write(f"Requested: {str(check_list_to_fuzhu)}\n")

    print(f"辅助检查项目请求 -> Patient: {patient_id} | Items: {check_list_to_fuzhu}")

    # 2. 从本地数据库读取完整报告 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    full_check_report = {}
    
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            target_record = next((item for item in data if item.get('patient_id') == patient_id), None)
            
            if target_record:
                full_check_report = target_record.get('check_report', {})
            else:
                return json.dumps({"error": f"Patient ID {patient_id} not found"}, ensure_ascii=False)
                
    except Exception as e:
        return json.dumps({"error": f"Database error: {str(e)}"}, ensure_ascii=False)

    # 3. Python 计算基础统计量
    total_report_count = len(full_check_report)
    requested_count = len(check_list_to_fuzhu)
    
    # 提取现有报告的所有 Keys (只传 Keys 给 LLM)
    available_keys = list(full_check_report.keys())

    if not full_check_report:
        return json.dumps({
            "statistics": {
                "total_report_count": 0,
                "requested_count": requested_count,
                "matched_count": 0
            },
            "matched_content": {},
            "matched_string": ""
        }, ensure_ascii=False)

    # 4. 构建 Prompt (让 LLM 做选择题，而不是填空题)
    prompt = f"""
    Context:
        I have a list of available medical examination items (keys) from a patient's report.
        I also have a list of specific tests requested by a doctor.
    
    Task:
        Select the specific item names from the "Available Items" list that correspond to the "Requested Items".
        **Use extremely loose and broad semantic matching.**
    
    Matching Rules:
        1. **Keyword Priority**: Match primarily based on the **examination type/modality** keywords (e.g., "Ultrasound", "CT", "MRI", "Blood", "Hormone", "Panel", "Test") regardless of the specific body part or organ mentioned.
        2. **Broad Association**: 
           - If request contains "Ultrasound" (even if "Liver Ultrasound") and available item has ANY "Ultrasound" (e.g., "Pelvic Ultrasound"), MATCH IT.
           - If request implies "Blood", "Lab", "Routine", match ALL available blood panels (CBC, BMP, CMP, Hormone, etc.).
        3. **Maximize Recall**: It is better to include a result that shares the same checking method than to return nothing.
    
    Available Items (List):
    {json.dumps(available_keys, ensure_ascii=False)}
    
    Requested Items (List):
    {json.dumps(check_list_to_fuzhu, ensure_ascii=False)}
    
    Output Format:
    Return ONLY a valid JSON object with a single key "matched_keys" containing the list of strings found in "Available Items".
    Example: {{"matched_keys": ["Complete Blood Count (CBC)", "Obstetric Ultrasound"]}}
    """

    # 5. 调用 LLM 获取匹配的 Key 列表
    llm_response = call_llm_with_retry(prompt)
    
    matched_keys = []
    if isinstance(llm_response, dict):
        matched_keys = llm_response.get("matched_keys", [])
    elif isinstance(llm_response, list):
        matched_keys = llm_response # 容错处理
    
    print(f"匹配的键===》{matched_keys}")
    # 6. Python 后处理：根据 Key 回填内容 & 拼接字符串
    matched_content = {}
    matched_string_parts = []
    
    for key in matched_keys:
        # 确保 LLM 返回的 key 真的在报告里 (防止幻觉)
        if key in full_check_report:
            value = full_check_report[key]
            matched_content[key] = value
            matched_string_parts.append(f"{key}: {value}")
            
    matched_count = len(matched_content)
    matched_string = "; ".join(matched_string_parts)
    print("===========================")
    print(matched_string)
    # 7. 组装最终结果结构
    final_result = {
        "statistics": {
            "total_report_count": total_report_count,
            "requested_count": requested_count,
            "matched_count": matched_count
        },
        "matched_content": matched_content,
        "matched_string": matched_string
    }

    result_str = json.dumps(final_result, ensure_ascii=False, indent=4)
    
    # 8. 记录响应日志
    with open(f"results/examinations/{model_name}/{patient_id}.txt", 'a', encoding='utf-8') as file:
        file.write("\nResponse:\n" + result_str + "\n" + "="*20 + "\n")

    # 返回拼接好的字符串
    res = final_result['matched_string']
    return res

openai_agent_tools = []
for func in registered_functions:
    tool = FunctionTool.from_defaults(fn=func)
    openai_agent_tools.append(tool)
