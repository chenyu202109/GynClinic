import os
import dspy
import cohere
import re
from datetime import datetime
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# from cohere.responses.rerank import RerankResult

from dspy.predict import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

from typing import Dict, Optional, List
from tqdm import tqdm

from utils import pp, ppj, ppjf
from utils import exists, defaults
from citations_utils import create_reference_nodes
from rag_logger import logger
from rag_config import RAGConfig
from chroma_db_retriever import ChromadbRM
from rag_utils import * # change
from signatures import * # change
from rag import RAG, load_rag

from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from collections import namedtuple
from med_agent import MedOpenAIAgent
from get_mytools import get_tools_byname
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent import ReActAgent, FunctionCallingAgent

from dotenv import load_dotenv
load_dotenv()

rag_config = RAGConfig()

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name=rag_config.default_embed_model
)

retriever_model = ChromadbRM(
    collection_name=rag_config.collection_name, 
    persist_directory=rag_config.default_client_path, 
    embedding_function=embedding_function,
    k=rag_config.retrieve_top_k,
)

lm_model = dspy.OpenAI(model=rag_config.default_llm_name, **rag_config.llm_kwargs)

rerank_model = cohere.Client(api_key=os.environ.get('COHERE_API_KEY'))

# lm_model.inspect_history(n=1)

dspy.settings.configure(lm=lm_model, trace=[], temperature=rag_config.llm_kwargs["temperature"], rm=retriever_model)


tool_result_dir="./results/tool_results"
rag_result_dir="./results/rag_results"
history_dir = f"./results/dp_history/"
output_path = "./Imaging/result.json"

# --- 角色 A: 问诊医生 (Doctor Interviewer) ---
DOCTOR_INTERVIEW_PROMPT_TEMPLATE = """
You are a clinical doctor, and your task is to gradually obtain as complete medical record information as possible through multiple rounds of Q&A with patients.
Your behavior rules in the conversation are as follows:
    1. You cannot directly request patients to provide all medical record information at once, but should gradually inquire like in an outpatient clinic.
    2. The consultation process should follow clinical logic:
        - Inquire about the onset time, duration, nature, aggravating or relieving factors of the symptoms.
        - Further inquire about accompanying symptoms.
        - Gradually transition to current medical history, past medical history, menstrual and marital history, and family history.
    3. Ask only one specific question at a time to avoid asking multiple questions at once.
    4. If the patient's answer is vague or uncertain, you need to ask for details, such as "Can you roughly say how long ago it was."
    5. You need to remember the information that the patient has already answered and avoid asking repeated questions.
    6. When you feel that you have obtained sufficient complete information, you can confirm with the patient and summarize the entire medical record.
Please always communicate with patients as a professional clinical doctor.
Finally, after collecting enough information, use '<dialogue_over>Break</dialogue_over>' to end the conversation.

Current Dialogue History:
{history}

Based on the history above, please provide your next response or question to the patient:
"""

# --- 角色 B: 病历总结专家 (Summary Expert) ---
SUMMARY_INSTRUCTION_TEMPLATE = """
You are an experienced gynecological expert who excels in summarizing the communication records between hospital gynecologists and patients.

The following is a record of the communication between the doctor and the patient:
{history}

Please summarize the record into a standard medical report format:
    **Chief complaint**: write chief complaint.
    **Present medical history**: write patient's present medical history.
    **Menstrual and marital history**: write menstrual and marital history.
    **Past history**: write past history.
    **Family history**: write family history.

Please fill in the content strictly based on the communication records. Do not fabricate information.
Describe the patient in the third person (e.g., "The patient...").
"""

# --- 角色 D: 患者 (Patient - 保持 Agent 模式以维护隐藏状态) ---
PATIENT_CONTEXT_TEMPLATE = """
You are a patient visiting the outpatient department, and your complete medical record information is as follows: {patient_background}.

Your behavior rules in the conversation are as follows:
    1. You will not proactively tell the doctor the complete medical record at once, but will selectively answer when asked by the doctor.
    2. Your response style should conform to the habits of ordinary patients:
        - Use vague, colloquial, and subjective expressions instead of medical terminology.
        - It can be mixed with hesitation, difficulty remembering, etc.
    3. When a doctor asks a question unrelated to your medical record, you can answer "no" or "not very clear".
    4. Unless the doctor explicitly asks, you will not proactively provide additional information.
    5. You need to be patient and not give too much information at once.

Please always answer the doctor's questions in the voice of a real patient.
"""

# 加载RAG
rag = load_rag()
question1 = """
        Please read the patient information I provided in detail and investigate the patient's current condition. We are now going to discuss the outpatient diagnosis results of the patient at the diagnosis and treatment planning meeting. Please provide us with the outpatient diagnosis results of the patient first, listing 5 differential diagnosis results. The diagnosis results must be specific to a certain disease and not ambiguous, such as threatened abortion, missed abortion, endometrial hyperplasia, endometrial polyps, and uterine fibroids. Such a relatively concise diagnosis result, at the same time, you need to sort them according to the relevance of the patient's condition, with the first item being the most relevant condition.
        Then, regarding the differential diagnosis results you listed, what other physical examinations and auxiliary examination items do she need to do? Please carefully read the patient's context and list them, Must must list ** 8 ** examination items. The examination items should also be specific and not vague, and there must be detailed examination items and examination items. Please pay attention to strictly distinguishing the different examination items that can be done for different patients' conditions. For example, pregnant women cannot undergo radioactive scans such as CT and MRI, only ultrasound.
        The above content should be precise and sufficient so that we can make better decisions. We will supplement the results of the auxiliary examination items you listed later and make a precise diagnosis.
        The examination items you have listed may include the following common laboratory tests Complete Blood Count (CBC)、Basic Metabolic Panel (BMP)、Comprehensive Metabolic Panel (CMP)、Hormone Panel、Miscellaneous， Radiological examinations include: Ultrasound, MRI, CT
        But it is also necessary to assign examination items based on the specific situation of the patient.
        Please note that the patient is very young, and we need to find as many accurate diagnoses as possible for them. Before you make a diagnosis, you can search on Pubmed and Google.
        
        """
question1_outputformat = """
        Notice: The confidence range is 0-100%, and it should be noted that the total confidence of all diagnostic results is 100%.
        Please strictly follow the following format for output:
        # Differential diagnosis results (from high to low correlation with the patient)
            - Diagnosis 1[confidence]: Detailed reasons need to be combined with the patient's current situation.
            - Diagnosis 2[confidence]: ...
            ...
        --- 
        # Examination items
            - Physical examination: Detailed reasons need to be combined with the patient's current condition.
            - Auxiliary examination 1: Detailed reasons need to be combined with the patient's current condition.
            - Auxiliary examination 2: ...
            ...
        ---
        # Detailed diagnostic reasoning process
            This includes the reasoning process for distinguishing diagnostic results, prioritizing diagnostic results, and recommending inspection items. Explain the diagnostic thinking in these three aspects.

"""

def gen_rag_case(name: str,basic_msg:str,context_msg:str, physical_examination: str):

    # doctor_agent（GynAgent）初始化
    doctor_tools = get_tools_byname(need_tools_name = ['google_search','query_pubmed'])
    doctor_agent = MedOpenAIAgent.from_tools(tools=doctor_tools, rag=rag)
    
    # patient_agent初始化
    patient_context_str = PATIENT_CONTEXT_TEMPLATE.format(patient_background=context_msg)
    patient_agent = FunctionCallingAgent.from_tools(
        llm=OpenAI(model="gpt-4o",temperature=0.3),
        system_prompt=patient_context_str
    )

    # auxiliary_agent
    auxiliary_agent = FunctionCallingAgent.from_tools(
        tools = get_tools_byname(need_tools_name=["get_check_report"]),
        llm=OpenAI(model="gpt-4o",temperature=0.0),
        system_prompt="""You are an experienced gynecological expert, and you need to conduct corresponding examinations on patients through some examination items. After the inspection is completed, you only need to return the complete structured inspection results to the user.""",
        max_function_calls=3
    )

    # 问诊阶段
    print(f"################################## 开始问诊: {name} ###############################################")
    dialogue_history = []
    
    # 开场白
    doctor_res = "Hello, I am Dr. Zhang. what's wrong with you today?"
    dialogue_history.append({"doctor": doctor_res})
    
    max_turns = 20
    current_turn = 0

    while current_turn < max_turns:
        current_turn += 1
        
        # --- 1.1 Patient Answers (Agent) ---
        his_str = f"Below is the dialogue history.\n{dialogue_history}"
        patient_agent.system_prompt = patient_context_str + "\n" + his_str
        patient_msg = patient_agent.chat(doctor_res)
        patient_res = patient_msg.response
        dialogue_history.append({"patient": patient_res})

        # --- 1.2 Doctor Follow-up (Using the new complete method) ---
        
        # Format the prompt with the current history
        doctor_input = DOCTOR_INTERVIEW_PROMPT_TEMPLATE.format(
            history=json.dumps(dialogue_history, ensure_ascii=False)
        )
        
        # Call the lightweight complete method
        # This uses the logic we defined in Part 1
        doctor_res_raw = doctor_agent.complete(doctor_input)
        doctor_res = str(doctor_res_raw).strip()

        # --- 1.3 Check for Termination ---
        if "<dialogue_over>Break</dialogue_over>" in doctor_res:
            print("Doctor decided to end the interview phase.")
            # Optional: Ask the LLM to summarize the interview before breaking
            break
            
        dialogue_history.append({"doctor": doctor_res})
        print(f"Turn {current_turn} - Doctor: {doctor_res}")

    # 保存对话历史
    with open(f"{history_dir}/{name}.txt", "w", encoding="utf-8") as f:
        f.write(str(dialogue_history)+"\n")

    # ==========================================
    # Phase 2: 总结 (Doctor LLM 直接生成)
    # ==========================================
    print("################################## 生成病历总结 ###############################################")
    
    # 构造总结 Prompt
    summary_input = SUMMARY_INSTRUCTION_TEMPLATE.format(history=json.dumps(dialogue_history, ensure_ascii=False))
    
    # 调用 doctor_llm 生成总结
    summary_response = doctor_agent.complete(summary_input)
    summarized_content = str(summary_response)
    
    # 保存总结
    with open(f"{history_dir}/{name}.txt", "a", encoding="utf-8") as f:
        f.write(summarized_content)

    # 我们不希望医生agent直接拿到患者的背景信息，而是通过和患者之间沟通得到的信息 | patient_agent和doctor_agent初次交互了解患者信息，放入doctor_agent记忆力中
    # 组合 Context: 基础信息（患者年龄、性别、民族等） + 问诊总结

    context_for_diagnosis = basic_msg + "\n" + summarized_content
    print("##################################  医患交流记录总结（Get患者背景） ###############################################")
    print(context_for_diagnosis) 
    doctor_agent.memory.put(
        ChatMessage(
        content=context_for_diagnosis,
        role=MessageRole.USER
    ))
    
    # docktor_agent step1 出具诊断和检查项目
    doctor_step1 = doctor_agent.chat_ext(context=context_for_diagnosis, question=question1, outputformat = question1_outputformat, use_rag=True, name=name)

    # 把RAG的信息放到doctor_agent中
    # doctor_agent.memory.put(ChatMessage(
    #     content=doctor_step1_rag.response,
    #     role=MessageRole.ASSISTANT
    # ))

    # 辅助检查agent
    print("##################################  auxiliary_agent ###############################################")
    measure_instruct = f"""
        The patient's background information is {basic_msg}\n The patient's preliminary diagnosis information is {doctor_step1.response}\n.
        The doctor has made a preliminary outpatient diagnosis, but differential diagnosis is still needed, so there are some items that need to be checked in the diagnostic information. 
        Please conduct corresponding examinations on the patient according to the examination items provided by the doctor.  
        Please use the auxiliary inspection item list directly for querying, do not split the query.  
        And obtain a summary of the final report results and return it to me.
    """
    get_check_msg = auxiliary_agent.chat(measure_instruct)
    print("##################################  辅助检查结果 ############################################### \n")
    print(get_check_msg)

    doctor_agent.memory.put(ChatMessage(
        content=get_check_msg.response,
        role=MessageRole.USER
    ))

    #第二阶段#################################################################################################################################
    print("##################################  修正诊断（通过辅助检查报告和上一轮工具的信息） ###############################################")
    # docktor_agent step2 开始进行修正诊断和推荐治疗方案 | 收到辅助检查的信息，针对患者信息进行全面检索
    # [Preliminary Diagnosis Results]:{doctor_step1.response}\n.
    mz_context = f"""The patient has completed the physical examination and auxiliary examination items you added, and added them to the patient's background information. 
    The following is the patient's medical history, auxiliary examination results\n
    {context_for_diagnosis}
    [Physical Examination]:{physical_examination}\n 
    [Auxiliary Check Report]:{get_check_msg.response} \n 
    """

    xzzd_quesion = f"""
        Now the patient information is sufficient, please read the patient information and examination information carefully again. Based on the patient's current condition and a series of primary diagnostic results, it is necessary to combine the patient's current condition and background information to make a revised diagnosis. 
        A revised diagnosis should be made and specific reasons for revision should be provided. The diagnostic results should be sorted in order, with the higher ranking being more relevant to the patient's condition, ensuring priority, and then output as 5 results, as there were also 5 previous diagnostic results, which is very important.
        Please be aware that we need to analyze patients based on comprehensive information and not rely solely on previous results, as previous diagnostic findings may interfere with the results. We now need to combine real clinical diagnostic thinking to analyze the patient's disease condition based on their **Chief complaints**, **Current medical history**, **Past medical history**, **Menstrual and marital history**, **Physical examination**, and **Auxiliary examination** results, in order to determine the diagnosis. 
        At the same time, when outputting various diagnostic reasoning processes in the following format, you must also meet this requirement and combine multiple aspects.
    """
    xzzd_quesion_outputformat = """
                    Notice: The confidence range is 0-100%, and it should be noted that the total confidence of all diagnostic results is 100%.
                    Please output strictly in the following format:
                    # Summary of patient comprehensive information:
                        ...
                    ---
                    # Correction of diagnostic results (in order of relevance to the patient from front to back):
                    <final_answer>['Diagnosis 1 ','Diagnosis 2' ..]</final_answer>
                    <final_confidence>['Diagnosis 1 confidence percentage', 'Diagnosis 2 confidence percentage'..]</final_confidence>
                    ---
                    # Explanation of Revised Diagnosis Results
                        - Diagnosis 1: Very **Detailed** reasons need to be combined with the patient's current situation and  should include the patient's **Chief complaints**, **Current medical history**, **Past medical history**, **Menstrual and marital history**, **Physical examination**, and **Auxiliary examination** results for analysis.
                        - Diagnosis 2: ...
                        ...
                    --- 
    """
   
    # 防止出现最后LLM没输出结果或者输出空结果
    res = ""
    retry_count = 0
    while retry_count < 3:
        # 调用 doctor_llm 进行最终诊断
        doctor_step2 = doctor_agent.chat_ext(context=mz_context, question=xzzd_quesion, outputformat=xzzd_quesion_outputformat, use_rag=True, name=name)
        res = str(doctor_step2.response)
        if res.strip():
            break
        print("Response was empty, retrying...")
        retry_count += 1

    return res

import re
if __name__ == "__main__":
    file_path = './Imaging/data.json'   
    with open(file_path, 'r', encoding='utf-8') as f:
        patients = json.load(f)

    print(f"已读取 {file_path}，共包含 {len(patients)} 条记录。")

    for idx, item in enumerate(tqdm(patients, desc="处理病例中", unit="例"), start=1):
        name = item['patient_id']
        basic_msg = item.get('basic_msg', "")
        context_msg = item.get('context_msg', "") 
        physical_examination = item.get('physical_examination', "") 
        try:
            # 执行全流程
            patient_result = gen_rag_case(name=name, basic_msg=basic_msg, context_msg=context_msg, physical_examination = physical_examination)
            
            # 提取最终结果
            pattern = r'<final_answer>(.*?)</final_answer>'
            matches = re.findall(pattern, patient_result, re.DOTALL) 
            if matches:
                item['prediction'] = matches[-1].strip()
            else:
                item['prediction'] = "no final_answer tag found"

        except Exception as e:
            item['prediction'] = "extract error"
            print(f"Error processing {name}: {e}")

        # break # 调试用，处理完一条后停止

    # 写入结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=4)

    print(f"已将数据写入：{output_path}")
