import os
import json
import re
import time  # 新增：用于重试等待
import random # 新增：用于随机等待
from typing import Dict, Optional, List
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures  # 新增：用于多线程

# LlamaIndex Imports
from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Local Imports
from get_mytools import get_tools_byname

load_dotenv()

# ==========================================
# 1. Prompt 定义 (角色设定)
# ==========================================

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

# --- 角色 C: 诊断专家 (Diagnosis Expert) ---
DOCTOR_DIAGNOSIS_SYSTEM_PROMPT = """
# Role
    You are a medical diagnostic expert with a senior professional title, proficient in the diagnosis of gynecological diseases.
# Task
    Due to the complexity of gynecological disease diagnosis, understanding the sources and treatment methods of diseases requires the support of many tools and a wealth of knowledge. For example, in cases where the chief complaint is vaginal bleeding, we must first determine the patient's age and whether they are pregnant, as pregnancy may be caused by miscarriage, and the specific type of miscarriage needs to be determined through examination reports. If the patient is not pregnant, we need to further evaluate the patient's medical history and examination report for endometrial hyperplasia, endometrial polyps, cervical cancer, or other conditions. Next, please assist me in diagnosing the patient.
# Logic
    In this diagnosis, you will receive the patient's medical records and questions raised by the doctor. Please follow the following steps when diagnosing:
    1. Carefully read the patient's relevant medical information and check the available diagnostic tools.
    2. Develop diagnostic strategies and, if necessary, use multiple rounds of tools to collect and analyze more information. You can combine the output and input of the tool for analysis.
    3. Use all the tools you think are useful, use evidence-based methods to support clinical decision-making for gynecological diseases, and provide precise assistance to doctors.
# Attention
    1. All patient information is for reference only, and you should think and use tools to search based on the specific situation of the patient.
    2. Pathological examination is the gold standard for diagnosis, and all other information is an auxiliary tool for diagnosis. However, the medical records we provide to you do not include pathological examination data, so all diagnostic decisions are made in the context of outpatient diagnosis and do not involve pathological examination, so you need to carefully distinguish.
    3. The ultimate goal is to assist clinical doctors in providing more comprehensive diagnostic support and ensure that every decision is strictly based on the patient's specific situation.
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

# ==========================================
# 2. 主逻辑函数
# ==========================================

# gpt-4o
# qwen3-235b-a22b

# gpt-5-mini
# deepseek-v3-1-think-250821
# gemini-2.5-flash
# gpt-5-mini-2025-08-07

# model_name = "gpt-oss-20b"
model_name = "qwen2.5-72b-instruct"
res_result_dir=f"./results/0llm/{model_name}-eme-noaub"
output_path = f"./Imaging/result-llm-{model_name}-eme-noaub.json"
history_dir = f"./results/dp_history/{model_name}-eme-noaub"

if not os.path.exists(res_result_dir):
    os.makedirs(res_result_dir, exist_ok=True) # 修正：增加 exist_ok 防止多线程竞态条件报错
if not os.path.exists(history_dir):
    os.makedirs(history_dir, exist_ok=True)

def extract_medical_notes(file_path):
    """
    读取文件，去除开头的数组内容（对话记录），保留后续的病历文本。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 未找到。")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 核心逻辑：找到数组结束的标志 ']'
        # split(']', 1) 表示以第一个 ']' 为分隔符，只分割一次
        # 这样可以防止文本内容中如果也有 ']' 会被错误切割
        parts = content.split(']', 1)
        
        if len(parts) > 1:
            # parts[0] 是数组内容（丢弃）
            # parts[1] 是数组后的内容（保留）
            extracted_text = parts[1].strip() # strip() 去除开头多余的换行和空格
            return extracted_text
        else:
            # 如果没找到 ']'，可能文件格式不对，这里选择原样返回或报错
            print("警告：未检测到数组格式（未找到 ']'），返回原始内容。")
            return content.strip()

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def gen_rag_case(name: str, basic_msg: str, context_msg: str, physical_examination: str):
    
    # 1. 初始化核心 LLM (GPT-4o)
    # 这个 LLM 将扮演：问诊医生、总结员、诊断专家
    doctor_llm = OpenAILike(
        model=model_name, 
        context_window=400000,
        is_chat_model=True,
        is_function_calling_model=True,
        system_prompt=DOCTOR_DIAGNOSIS_SYSTEM_PROMPT, 
        temperature=0.3,
        max_retries=5,
        timeout = 300.0
    )

    # doctor_llm = OpenAILike(
    #     model=model_name,
    #     context_window=128000,
    #     is_chat_model=True,
    #     is_function_calling_model=True,
    #     system_prompt=DOCTOR_DIAGNOSIS_SYSTEM_PROMPT,
    #     temperature=0.3,
    #     max_tokens=4096,
    #     api_key="TOQZZOJONJYZZ8Q0MBQ0PEVUMNAA6UXRKVLTEAWL",
    #     api_base="https://ai.gitee.com/v1",
    # )

    # doctor_llm = OpenAILike(
    #     model=model_name, 
    #     context_window=400000,
    #     is_chat_model=True,
    #     is_function_calling_model=True,
    #     system_prompt=DOCTOR_DIAGNOSIS_SYSTEM_PROMPT, 
    #     temperature=0.3,
    #     max_retries=5,
    #     timeout = 300.0
    # )

    
    # doctor_llm = OpenAILike(
    #     model=model_name,
    #     temperature=0.3,
    #     context_window=128000,
    #     is_chat_model=True,
    #     is_function_calling_model=False,
    #     api_key="sk-b989aefb40235367ab36871b8ac1b397",
    #     api_base="https://api.baichuan-ai.com/v1/",
    #     system_prompt=DOCTOR_DIAGNOSIS_SYSTEM_PROMPT
    # )

    # 2. 初始化病人 Agent
    # 病人仍需使用 Agent 模式，因为通过 System Prompt 注入"隐蔽病历"，并需要根据对话历史动态反应
    patient_context_str = PATIENT_CONTEXT_TEMPLATE.format(patient_background=context_msg)
    patient_agent = FunctionCallingAgent.from_tools(
        llm=OpenAI(
            model="gpt-4o",
            temperature=0.3
        ),
        system_prompt=patient_context_str
    )

    # ==========================================
    # Phase 1: 问诊 (Doctor LLM vs Patient Agent)
    # ==========================================
    print(f"################################## 开始问诊: {name} ###############################################")
    
    dialogue_history = []
    
    # 开场白
    doctor_res = "Hello, I am Dr. Zhang. what's wrong with you today?"
    dialogue_history.append({"doctor": doctor_res})
    
    max_turns = 20
    current_turn = 0

    while current_turn < max_turns:
        current_turn += 1
        
        # --- 1.1 患者回答 (Agent) ---
        # 更新患者记忆
        his_str = f"Below is the dialogue history.\n{dialogue_history}"
        patient_agent.system_prompt = patient_context_str + "\n" + his_str
        
        # 调用病人 Agent
        patient_msg = patient_agent.chat(doctor_res)
        patient_res = patient_msg.response

        # print(f"Patient: {patient_res}") # 多线程下减少print，避免输出混乱
        dialogue_history.append({"patient": patient_res})

        # --- 1.2 医生追问 (Doctor LLM 直接生成) ---
        # 此时不使用 Agent，而是构造 Prompt 直接让 doctor_llm 补全下一句话
        
        # 构造医生视角的输入 Prompt
        doctor_input = DOCTOR_INTERVIEW_PROMPT_TEMPLATE.format(history=json.dumps(dialogue_history, ensure_ascii=False))
        
        # 调用 doctor_llm 生成回复
        doctor_response_obj = doctor_llm.complete(doctor_input)
        doctor_res = str(doctor_response_obj).strip()

        # print(f"Doctor: {doctor_res}")

        # --- 1.3 判断对话结束 ---
        pattern = r'<dialogue_over>(.*?)</dialogue_over>'
        match = re.search(pattern, doctor_res)
        
        should_break = False
        if match:
            predicts_str = match.group(1).strip()
            if predicts_str == "Break":
                should_break = True
                # 移除标签，保留文本
                clean_res = doctor_res.replace(match.group(0), "").strip()
                if clean_res:
                     dialogue_history.append({"doctor": clean_res})
            else:
                dialogue_history.append({"doctor": doctor_res})
        else:
            dialogue_history.append({"doctor": doctor_res})
            
        if should_break:
            break

    # 保存对话历史
    with open(f"{history_dir}/{name}.txt", "w", encoding="utf-8") as f:
        f.write(str(dialogue_history)+"\n")

    # ==========================================
    # Phase 2: 总结 (Doctor LLM 直接生成)
    # ==========================================
    # print("################################## 生成病历总结 ###############################################")
    
    # 构造总结 Prompt
    summary_input = SUMMARY_INSTRUCTION_TEMPLATE.format(history=json.dumps(dialogue_history, ensure_ascii=False))
    
    # 调用 doctor_llm 生成总结
    summary_response = doctor_llm.complete(summary_input)
    summarized_content = str(summary_response)
    
    # 保存总结
    with open(f"{history_dir}/{name}.txt", "a", encoding="utf-8") as f:
        f.write(summarized_content)

    # summarized_content = extract_medical_notes(f"{history_dir}/{name}.txt")

    # ==========================================
    # Phase 3: 初步诊断 (Doctor LLM)
    # ==========================================
    # print("################################## 开始鉴别诊断 (Step 1) ###############################################")

    # 组合 Context: 基础信息 + 问诊总结
    context_for_diagnosis = basic_msg + "\n" + summarized_content
    print("Context for diagnosis:\n", context_for_diagnosis)

    # 诊断 Prompt
    question1 = f"""
        {context_for_diagnosis}\n
        Please read the patient information I provided in detail and investigate the patient's current condition. We are now going to discuss the outpatient diagnosis results of the patient at the diagnosis and treatment planning meeting. Please provide us with the outpatient diagnosis results of the patient first, listing 5 differential diagnosis results. The diagnosis results must be specific to a certain disease and not ambiguous, such as threatened abortion, missed abortion, endometrial hyperplasia, endometrial polyps, and uterine fibroids. Such a relatively concise diagnosis result, at the same time, you need to sort them according to the relevance of the patient's condition, with the first item being the most relevant condition.
        Then, regarding the differential diagnosis results you listed, what other physical examinations and auxiliary examination items do she need to do? Please carefully read the patient's context and list them, Must list ** 8 ** items. The examination items should also be specific and not vague, and there must be detailed examination items and examination items. Please pay attention to strictly distinguishing the different examination items that can be done for different patients' conditions. For example, pregnant women cannot undergo radioactive scans such as CT and MRI, only ultrasound.
        The above content should be precise and sufficient so that we can make better decisions. We will supplement the results of the auxiliary examination items you listed later and make a precise diagnosis.
        The examination items you have listed may include the following common laboratory tests Complete Blood Count (CBC)、Basic Metabolic Panel (BMP)、Comprehensive Metabolic Panel (CMP)、Hormone Panel、Miscellaneous， Radiological examinations include: Ultrasound, MRI, CT
        But it is also necessary to assign examination items based on the specific situation of the patient.
        Please note that the patient is very young, and we need to find as many accurate diagnoses as possible for them. 
        Please note that the diagnostic result you output must be a disease name, not a description of a phenomenon. For example, you should output a diagnosis name using short terms such as Vulvar Abscess or Retained Products of Concept, rather than a lengthy description of the phenomenon.

        Notice: The confidence range is 0-100%, and it should be noted that the total confidence of all diagnostic results is 100%.
        Please strictly follow the following format for output:
        # Differential diagnosis results (from high to low correlation with the patient)
            - Diagnosis 1[confidence]: Detailed reasons need to be combined with the patient's current situation.
            - Diagnosis 2[confidence]: ...
            ...
        --- 
        # Examination items
            - Auxiliary examination 1 Name: Detailed reasons need to be combined with the patient's current condition.
            - Auxiliary examination 2 Name: ...
            ...
        ---
        # Detailed diagnostic reasoning process
            This includes the reasoning process for distinguishing diagnostic results, prioritizing diagnostic results, and recommending inspection items. Explain the diagnostic thinking in these three aspects.

        """

    # 调用 doctor_llm 进行诊断 (此时它回归了System Prompt定义的诊断专家角色)
    # doctor_llm = OpenAILike(
    #     model= model_name, 
    #     context_window=400000,
    #     is_chat_model=True,
    #     is_function_calling_model=True,
    #     system_prompt=DOCTOR_DIAGNOSIS_SYSTEM_PROMPT, 
    #     temperature=0.3,
    #     max_retries=5,
    #     timeout = 300.0
    # )

    doctor_step1 = doctor_llm.complete(question1)
    doctor_step1_str = str(doctor_step1)

    with open(f"{res_result_dir}/{name}-step1.txt", "w",encoding="utf-8") as f:
        f.write(doctor_step1_str)
    # print("初步诊断完成。")

    # ==========================================
    # Phase 4: 辅助检查 (Measure Agent)
    # ==========================================
    # print("################################## 执行辅助检查 ###############################################")
    
    # 这个 Agent 负责查工具，保留原逻辑
    measure_agent = FunctionCallingAgent.from_tools(
        tools = get_tools_byname(need_tools_name=["get_check_report"]),
        llm=OpenAI(
            model="gpt-4o",
            temperature=0.0
        ),
        system_prompt="""You are an experienced gynecological expert, and you need to conduct corresponding examinations on patients through some examination items. After the inspection is completed, you only need to return the complete structured inspection results to the user.""",
        max_function_calls=1
    )

    measure_instruct = f"""
        The patient's background information is {basic_msg}\n The patient's preliminary diagnosis information is {doctor_step1_str}\n.
        The doctor has made a preliminary outpatient diagnosis, but differential diagnosis is still needed, so there are some items that need to be checked in the diagnostic information. 
        Please conduct corresponding examinations on the patient according to the examination items provided by the doctor.  
        Please use the auxiliary inspection item list directly for querying, do not split the query.  
        And obtain a summary of the final report results and return it to me.
    """
    # 模拟触发工具调用
    check_report = measure_agent.chat(measure_instruct)
    
    # print("Auxiliary Check Report:\n", check_report)

    # ==========================================
    # Phase 5: 修正诊断 (Doctor LLM)
    # ==========================================
    # print("################################## 修正诊断 (Step 2) ###############################################")

    # 这边直接将体格检查返回,我们认为体格检查是个必做的项目，直接进行返回.

    mz_context = f"{context_for_diagnosis}\n[Physical Examination]{physical_examination}\n[Auxiliary Check Report]:\n{check_report}\n"
    # print(mz_context)

    xzzd_quesion = f"""
                    {mz_context}
                    The patient has completed the physical examination and auxiliary examination items you added, and added them to the patient's background information. 
                    Now the patient information is sufficient, please read the patient information and examination information carefully again. Based on the patient's current condition and a series of revised diagnostic results, it is necessary to combine the patient's current condition and background information to make a revised diagnosis. 
                    A revised diagnosis should be made and specific reasons for revision should be provided. The diagnostic results should be sorted in order, with the higher ranking being more relevant to the patient's condition, ensuring priority, and then output as 5 results, as there were also 5 previous diagnostic results, which is very important.
                    Please note that you should conduct a comprehensive search based on existing information, especially auxiliary examination information, combined with previous diagnostic stage results. Do not rely solely on previous results, as previous diagnostic results may interfere with the results.
                    Notice: The confidence range is 0-100%, and it should be noted that the total confidence of all diagnostic results is 100%.
                    Please note that the diagnostic result you output must be a disease name, not a description of a phenomenon. For example, you should output a diagnosis name using short terms such as Vulvar Abscess or Retained Products of Concept, rather than a lengthy description of the phenomenon.
                    
                    Please output strictly in the following format:
                    # Summary of patient comprehensive information:
                        ...
                    ---
                    # Correction of diagnostic results (in order of relevance to the patient from front to back):
                    <final_answer>['Diagnosis 1 ','Diagnosis 2' ..]</final_answer>
                    <final_confidence>['Diagnosis 1 confidence percentage', 'Diagnosis 2 confidence percentage'..]</final_confidence>
                    ---
                    # Explanation of Revised Diagnosis Results
                        - Diagnosis 1: Detailed reasons need to be combined with the patient's current situation
                        - Diagnosis 2: ...
                        ...
                    --- 
    """

    res = ""
    retry_count = 0
    while retry_count < 3:
        # 调用 doctor_llm 进行最终诊断
        doctor_step2 = doctor_llm.complete(xzzd_quesion)
        res = str(doctor_step2)
        if res.strip():
            break
        print("Response was empty, retrying...")
        retry_count += 1

    # 保存最终诊断结果
    with open(f"{res_result_dir}/{name}-step2.txt", "w",encoding="utf-8") as f:
            f.write(res)

    return res

# ==========================================
# 3. 多线程处理封装与执行
# ==========================================

def process_single_patient(item: Dict) -> Dict:
    """
    单个病例的处理函数，用于多线程调用。
    包含了重试机制：当发生异常时，会重试指定次数。
    """
    name = item['patient_id']
    basic_msg = item.get('basic_msg', "")
    context_msg = item.get('context_msg', "") 
    physical_examination = item.get('physical_examination', "") 
    
    # ---------------------------
    # 重试机制配置
    # ---------------------------
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 5  # 基础等待时间（秒）
    
    for attempt in range(MAX_RETRIES):
        try:
            # 执行全流程
            patient_result = gen_rag_case(name=name, basic_msg=basic_msg, context_msg=context_msg, physical_examination=physical_examination)
            
            # 提取最终结果
            pattern = r'<final_answer>(.*?)</final_answer>'
            matches = re.findall(pattern, patient_result, re.DOTALL) 
            if matches:
                item['prediction'] = matches[-1].strip()
            else:
                item['prediction'] = "no final_answer tag found"
            
            # 如果成功执行到这里，直接返回，跳出重试循环
            return item

        except Exception as e:
            # 捕获异常
            print(f"[Warning] Error processing {name} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                # 如果不是最后一次尝试，则等待一段时间后重试
                wait_time = RETRY_DELAY + random.random() * 3 # 添加一点随机扰动
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # 如果是最后一次尝试仍然失败，记录错误
                print(f"[Error] Final failure for {name}: {e}")
                item['prediction'] = "extract error"
                # 继续抛出或者返回带有错误标记的 item，这里选择返回 item 以便主线程继续保存
                return item
    
    return item

if __name__ == "__main__":
    file_path = './Imaging/data.json'   
    with open(file_path, 'r', encoding='utf-8') as f:
        patients = json.load(f)

    print(f"已读取 {file_path}，共包含 {len(patients)} 条记录。")
    print("开始多线程处理 (Max Workers: 20)...")

    # 存储处理完的结果
    processed_patients = []
    
    # 使用 ThreadPoolExecutor 进行多线程处理
    # max_workers=20 设置并发线程数为 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_patient, item): item for item in patients}
        
        # 使用 tqdm 显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(patients), desc="多线程处理进度", unit="例"):
            try:
                result = future.result()
                processed_patients.append(result)
            except Exception as e:
                print(f"线程执行异常: {e}")

    # 写入结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=4)

    print(f"已将数据写入：{output_path}")