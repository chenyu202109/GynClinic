import dspy

# 检查引用
class CheckCitationFaithfulness(dspy.Signature):
    __doc__ = f"""Verify that the text is truly based on the provided context. Statemtents must be true and reflected in the source text."""
    context = dspy.InputField(
        desc="May contain relevant information to answer the question."
    )
    text = dspy.InputField(desc="Maximum of 1 or 2 sentences.")
    faithfulness = dspy.OutputField(
        desc="Boolean value indicating if text is faithful to context"
    )

# 生成4个子查询
class Search1(dspy.Signature):
    """Strategize queries for retrieving medical information from databases containing guidelines and official medical recommendations for cancer, including >700 medical scores (e.g., ECOG, GCS, HAS-BLED).
    Utilize scores for accurate decision-making if appropriate. Formulate questions to gather comprehensive medical data, aiding in the final decision. Incorporate wider medical knowledge, exploring all possible treatment avenues mentioned in guidelines, and anticipate future options.
    Explore all treatment options (drugs, chemotherapy, surgery, radiotherapy etc) that could arise from the patient context and the tool results, given your medical expertise.
    Ensure your queries are precisely formulated to identify the appropriate therapy line, such as determining the second-line treatment for patients who have experienced progression on their initial therapy. Include searches for all possible treatment options like surgery, chemotherapy, radiotherapy or targeted therapy, and consider the patient's specific condition and test results.
    Leverage your broad medical expertise to consider additional options beyond those explicitly mentioned in the user's query. Be creative while grounded on medical expertise.
    Frame questions precisely, targeting specific information. For example, instead of asking broad questions about the role of immunotherapy in colorectal cancer, tailor the query to address precise scenarios relevant to the patient's condition, such as 'Which immunotherapy agent is advised for second-line treatment in MSI-H positive cases?'.
    Return queries as a string formatted as a python list.
    """

    question = dspy.InputField(desc="The medical query to be addressed.")
    context = dspy.InputField(desc="In-depth details of the patient's case.")
    tool_results = dspy.InputField(
        desc="Accurate results from supplementary patient assessments (tools)."
    )
    searches = dspy.OutputField(
        desc="""A string of maximum 8 targeted subqueries, formatted as a list, derived from the main question, patient case context (context), and additional test results (tool_results). Subqueries should be answerable using general medical literature, focusing solely on the medical query without patient-specific details unanswerable by standard medical documents.
        Ensure the list is properly formatted, avoiding special characters like newlines. Unacceptable format. [Here is a list:\n 1. ..., second ...]. Good example format: ['Optimal treatment for X cancer?', 'Criteria for using Y drug?']. No special characters, like newlines or excessive ''."""
    )
class Search(dspy.Signature):
    """Strategize queries for Retrieving medical information related to the diagnosis of gynecological diseases from databases containing gynecological disease guidelines and official medical recommendations, especially for queries targeting>700 medical scores (such as ECOG, GCS, HAS-BLED). Make accurate diagnostic decisions based on the patient's medical background and scoring results. Design questions to comprehensively collect patients' medical data and assist in making final diagnostic decisions. Combining broader medical knowledge, explore all possible diagnostic pathways mentioned in the guidelines and predict future diagnostic choices.
Based on your medical expertise, comprehensively evaluate the patient's background, test results, and symptoms to ensure accurate diagnosis and rule out other possibilities. Accurately construct queries that focus on determining appropriate diagnostic pathways, such as providing recommendations for further examinations for patients who exhibit symptoms of suspected gynecological diseases during initial screening. This includes searching for all possible diagnostic tools and testing methods, and considering the patient's specific situation and examination results.
Utilize your medical expertise to consider diagnostic options beyond the standard diagnostic tools mentioned in the query. In the diagnostic process, consider innovation or the latest medical knowledge.
Accurately construct problems targeting specific information. For example, instead of asking broad questions about the initial screening of gynecological diseases, it is better to customize queries to address specific situations related to the patient's current condition, 
such as 'What disease is causing the XX phenomenon?', 'What is the recommended examination for patients with vaginal bleeding? ', 'How to differentiate and diagnose patients who exhibit symptoms of vaginal bleeding? ', 'Considering the patient's age, medical history, and examination results, what are the possible differential diagnoses?' .
Please note that for these issues, it is necessary to design based on the patient's real situation, integrating the patient's background, examination results, etc. It is essential to integrate the patient's true information, not to be too general, and to be precise.
Returns a query as a string in Python list format.
    """

    question = dspy.InputField(desc="The medical query to be addressed.")
    context = dspy.InputField(desc="In-depth details of the patient's case.")
    tool_results = dspy.InputField(
        desc="Accurate results from supplementary patient assessments (tools)."
    )
    searches = dspy.OutputField(
        desc="""A string of maximum 4 targeted subqueries, formatted as a list, derived from the main question, patient case context (context), and additional test results (tool_results). Subqueries should be answerable using general medical literature, focusing solely on the medical query without patient-specific details unanswerable by standard medical documents.
        Ensure the list is properly formatted, avoiding special characters like newlines. Unacceptable format. [Here is a list:\n 1. ..., second ...]. 
        Good example format: ['What disease is causing the X phenomenon?, 'Differential diagnosis for Y cancer?', 'What is the recommended examination for patients with X phenomenon?', Auxiliary inspection method for check Z state of illness?']. 
        No special characters, like newlines or excessive. Please make sure to consider the patient's true situation. """
    )

# 回答问题答案
class AnswerStrategy1(dspy.Signature):
    """Develop a structured approach to address medical queries, starting with a comprehensive and detailed summary of the patient's condition and test results. Prioritize the accuracy of details such as percentages and numbers, and incorporate medical scores judiciously. Identify and address any potential discrepancies in information that may require future resolution.
    Ensure that the response comprehensively addresses all pertinent treatment modalities, such as chemotherapy, targeted therapies, surgical intervention, or radiotherapy, as applicable to the patient's case.
    """

    context = dspy.InputField(
        desc="High-quality medical guidelines and recommendations."
    )
    patient = dspy.InputField(desc="Comprehensive patient case details.")
    tool_results = dspy.InputField(
        desc="Accurate results from supplementary diagnostic tests."
    )
    question = dspy.InputField(
        desc="The specific medical query related to the patient's context that needs to be adressed."
    )
    response = dspy.OutputField(
        desc="""A structured strategy, presented as a string, to resolve the medical query using the provided data. Begin with a synthesis of patient information and test outcomes, followed by a targeted plan referencing applicable medical documents and guidelines.Medical documents and guidelines must indicate their specific names
        Enumerate considerations such as treatment options and protocols. Avoid vague strategies like 'Evaluate the current treatment plan, Consider alternative treatment options, ...'.
        Ensure responses are specific and relevant, such as 'First, synthesize patient details and test results. Next, assess the current treatment against liver tumor progression. Finally, explore alternative therapies like SIRT, RFA, or FOLFIRI with Bevacizumab, based on guideline recommendations. Future treatment plans might include anti EGFR therapies like Cetuximab or Panitumumab'.
        Ensure completeness and relevance to the question. Assert that all pillars of treatments (chemotherapy, targeted therapies, surgery or radiotherapy) are addressed if relevant and contain the necessary level of detail, e.g specific drug or procedure names.In these treatment methods, it is necessary to confirm the specific names, including the specific methods of radiotherapy and chemotherapy, the names of targeted therapy drugs, and the specific organ locations determined in the surgical methods. This can facilitate doctors to verify and establish specific clinical decision-making plans
        Ensure that you pay attention to every tool result in a structured manner.
        Highlight potential conflicts in the data that you recieve.
        Recognize the boundaries of your capabilities. Should a tool's output present ambiguities or lack clarity in its conclusions, communicate these uncertainties to the user. Offer contingent pathways—for instance, using 'if...then...' statements—to navigate potential outcomes. Suggest consulting a medical expert in the respective field and think which futher tools or information from the user could help you solve the issue. For instance, if the radiology report is unclear / not unambiguous, ask for more information that could help you resolve the problem."""
    )
class AnswerStrategy(dspy.Signature):
    """Develop a structured approach to address medical issues, starting with a comprehensive and detailed summary of the patient's condition and test results, ensuring accurate description of all relevant details, including key data such as percentages and numbers. Prioritize medical scores (such as ECOG, GCS, HAS-BLED, etc.) and examination results (such as imaging, laboratory reports, etc.), and wisely incorporate these scores. Ensure consistency between all data and patient medical history to avoid any mismatches or omissions.
    Identify and resolve potential differences in information that may require further clarification, such as inconsistencies between imaging reports and medical history or conflicts between examination results and clinical symptoms, to ensure that these differences are accurately identified for further investigation.
    Ensure comprehensive consideration of all relevant diagnostic possibilities through structured methods. Based on the patient's symptoms and examination results, develop clear differential diagnosis to avoid vague speculation and provide specific guidance for possible future examinations.
    """

    context = dspy.InputField(
        desc="High-quality medical diagnose guidelines and recommendations."
    )
    patient = dspy.InputField(desc="Comprehensive patient case details.")
    tool_results = dspy.InputField(
        desc="Accurate results from supplementary diagnostic tests."
    )
    question = dspy.InputField(
        desc="The specific medical query related to the patient's context that needs to be adressed."
    )
    response = dspy.OutputField(
        desc="""A structured strategy, presented as a string, to resolve the medical query using the provided data.Firstly, the patient's detailed information and test results are integrated, including all relevant medical scores (such as ECOG, GCS, HAS-BLED, etc.) as well as examination results (such as imaging, laboratory reports, etc.). 
        Then, refer to applicable medical documents and guidelines to develop specific differential diagnosis strategies. Please clearly indicate the specific names of the medical documents and guidelines referenced.
        Ensure that all relevant diseases or causes are considered in differential diagnosis, and provide specific disease names where appropriate. 
        Based on the patient's symptoms, may it be early-stage uterine cancer or other gynecological tumors
        Pay attention to the results of each tool and check to ensure that the responses are specific and relevant. Highlight potential conflicts in the data you receive, such as inconsistencies between imaging reports and medical history, or mismatches between clinical symptoms and examination results. If there are areas of uncertainty, please communicate these uncertainties and suggest emergency measures, such as using a 'if... then...' statement.
        If there are ambiguities or unclear conclusions that cannot be answered, it is recommended to further consult medical experts in the relevant field and guide users to provide more information. For example, if the imaging examination results are unclear, please ask for more details that can help you solve the problem.
    
    """
    )


# 请记住，你在负责引用知识库时候，请不要随意添加或修改内容的意思，然后请尽你最大的可能丰富内容的同时，保证执行上面给你的要求。此外，尽可能的穿插引用在文章中我给你的详细的引用内容。
class GenerateCitedResponse(dspy.Signature):
    """Create an in-depth, well-structured and detailed response with valid citations, adhering to a predefined strategy. Begin with a summary of patient details and test outcomes. Each section of the response should correspond to a strategic suggestion, filled with specific, patient-oriented advice derived from medical guidelines (context).
    Citations should be included, such as [3], [4], [Tool], or [Patient], without an accompanying reference list. Promptly request additional information if necessary to complete the analysis.
    """

    strategy = dspy.InputField(
        desc="The structured approach to be employed in crafting the comprehensive answer."
    )
    context = dspy.InputField(
        desc="Medical guidelines and documentation pertinent to the inquiry, prefaced with 'Source x:', where x is the citation number like [x]."
    )
    patient = dspy.InputField(
        desc="Comprehensive patient case details, to be cited as [Patient]."
    )
    tool_results = dspy.InputField(
        desc="Google、PubMed search and accurate results from further examinations, to be cited as [Tool]."
    )
    question = dspy.InputField(
        desc="The medical query to be addressed using the provided context."
    )
    response = dspy.OutputField(
        desc="""
        Provide a detailed and cited narrative that combines diagnostic strategies, medical guidelines, and patient specific background information. 
        Citations should be frequent, every 1-2 sentences, while ensuring the response omits a reference list. 
        
        Please consider the patient's condition based on the actual situation, especially in the clinical diagnosis process of gynecological diseases.
        The answer should avoid using vague terms and focus on specific diagnostic steps and analysis of the patient's condition. For example, avoid using generic expressions such as "other possible diagnoses" or "common gynecological diseases", and instead elaborate on specific symptoms or manifestations, such as "considering the possibility of endometriosis in the patient" or "the symptom may be related to polycystic ovary syndrome".
        For further data or clarification requests, it should be particularly clear, especially regarding the use of tools in diagnostic strategies. If more information is needed to confirm the diagnosis, please clearly indicate and specify the type of information or tool required.
        Each diagnostic step should have unique value and closely revolve around the patient's medical history and clinical presentation, ensuring a fully personalized diagnostic process. Each diagnostic point should be analyzed in detail to ensure that they are closely aligned with the patient's current health status and potential diagnostic pathways.
        Please note that in the process of answering, do not repeat the strategy content or give overly general diagnostic opinions. Instead, focus on specific clinical diagnostic analysis and make inferences based on the patient's specific symptoms, medical history, and other available information.
        When the output of the tool is vague or the conclusion is unclear, please express these uncertainties in a timely manner and provide further diagnostic paths. For example, if imaging reports cannot clearly determine certain symptoms or signs, it is recommended that patients provide more clinical information or confirm through further diagnostic tools such as pathology reports or further imaging examinations. Using the logic of 'if... then...', provide specific response strategies and propose possible further actions.
        Ensure that all diagnostic recommendations are entirely based on the patient's actual condition, avoiding generalization of any diagnostic pathway or recommendation. Each step should provide detailed information and ensure that there is no redundant content.
                    
        """
    )

# A very detailed, citation-rich narrative, integrating insights from the strategy, medical guidelines, and specific patient context. 
#         Citations should be frequent, every 1-2 sentences, while ensuring the response omits a reference list. 
#         Consider all information in the context of cancer.
#         The response should eschew generic terminology in favor of detailed treatment specifics, e.g., avoiding 'other antiangiogenic drugs' in favor of explicit drug names like 'bevacizumab, ramucirumab or aflibercept' or replacing 'other chemotherapy' with 'chemotherapies x, y and z should be explored'.
#         Requests for additional data or clarification should be clear and specific, particularly regarding the use of tools in the strategy.
#         The response must be meticulously crafted to avoid any duplication of content, ensuring each point adds unique value to the overall narrative.
#         Furthermore, it is imperative that the response be tailored and personalized to the specific patient case at hand, reflecting a deep understanding of the patient's unique medical history, current condition, and potential treatment pathways.
#         This level of customization should manifest in the precise alignment of recommendations with the patient's diagnostic data, treatment history, and any other relevant personal health information.
#         The response should cover ALL pertinent treatment options without redundancy or generalities, and without restating the strategy.
#         Recognize the boundaries of your capabilities. Should a tool's output present ambiguities or lack clarity in its conclusions, communicate these uncertainties to the user. Offer contingent pathways—for instance, using 'if...then...' statements—to navigate potential outcomes. Suggest consulting a medical expert in the respective field and think which futher tools or information from the user could help you solve the issue. For instance, if the radiology report is unclear / not unambiguous, ask for the tumor locations so you can use one of your tools to solve the problem.
#         Please remember that when you are responsible for referencing the knowledge base, please do not add or modify the meaning of the content at will. Then, please try your best to enrich the content while ensuring the implementation of the requirements given to you above. In addition, try to intersperse the detailed citation content I have given you in the article as much as possible.


# 生成后续治疗建议和需要用户提供哪些信息来进一步辅助诊疗
class Suggestions(dspy.Signature):
    """Refine provided recommendations to guide the user in supplying additional details or data crucial for leveraging specific analytical tools, elucidating the reasons for these requests.
    Given broad suggestions on which tools to use, prompt the user for the exact data inputs required for these tools, ensuring that the suggestions are seamlessly integrated with the initial response.
    All tools at your disposal are detailed in the 'tools' field.
    Ensure that these refined suggestions seamlessly extend the initial response, without reiterating previous content or introducing new concepts.
    """

    response = dspy.InputField(
        desc="A comprehensive and accurate response, thoroughly backed by relevant sources."
    )
    recommendations = dspy.InputField(
        desc="Initial recommendations prompting the user to provide further information or data as input for tool use."
    )
    suggestions = dspy.OutputField(
        desc="""Crafted suggestions that prompt the user for specific information or data essential for certain tools, while mirroring the depth of detail found in the initial recommendations.
        Do not introduce new concepts or results that we already have. Only rely on the tools mentioned in the recommendations.
        Frame these suggestions as a coherent extension of the initial response, avoiding direct mention of tool names. Instead, imply the utility of the tools by stating with something like 'Utilizing my array of resources, I can further assist by doing x, if you could provide my with y' where x succinctly describes the function or benefit of the tool and y the required input.
        Specify the exact data inputs needed from the user to enable the effective use of analytical tools.
        Do not ask for access to any of the tools, they are actually provided to you already - instead ask for the necessary inputs the user needs to provide, like the path to the images for the genetic modeling.
        Formulate your output as either polite questions or instructions to the user.
        Limit the entire response to 2-3 sentences to maintain conciseness and focus.
        If you encounter conflicts in the data (e.g. unclear reports or contradictory statements between the given information and your findings), state them clearly.
        """
    )


# 概述工具使用的额外信息，也就是可以通过额外输入填补数据空白，不做建议哦
class RequireInput(dspy.Signature):
    """Leverage the detailed patient information, including test results (tool_results), to identify gaps in data that could be filled with additional inputs.
    Prompt the user to supply specific information or data that could enable the use of analytical tools that are listed in 'tools' and that have not been used so far (they don't appear in the 'tool_results' field).
    Focus solely on requests that necessitate user action and are directly tied to the utilization of available tools (as listed in 'tools'). For instance, ask for precise clinical details to compute a particular score or request additional images for in-depth analysis.
    Avoid referencing tools not explicitly included in the 'tools' list or tools where tool results ('tool_results') are already available.
    Tools that are already used might be described in 'tool_results' differently to their names in the available 'tools', match them accordingly to prevent suggesting tools that have been used already.
    Here is a mapping of their names to short a description: {
        "gen_radiology_report": "Automated Radiology Report Generation",
        "check_mutations": "Genetic Modeling from microscopic images",
        "onco_kb": "OncoKB Query",
        "segment_image": "Image Segmentation",
        "divide": "Ratio Calculation",
        "calculate": "Simple Calculus",
        "google_search": "Google Search",
        "query_pubmed": "PubMed Query",
    }
    Refrain from suggesting general medical consultations that do not directly involve the use of tools.
    """

    patient = dspy.InputField(desc="Comprehensive case details of the patient.")
    tool_results = dspy.InputField(
        desc="Accurate outcomes from supplementary patient diagnostics."
    )
    tools = dspy.InputField(
        desc="An inventory of available analytical tools, detailing their input requirements and appropriate scenarios for their application."
    )
    question = dspy.InputField(
        desc="The medical inquiry that necessitates further context or data for resolution."
    )
    response = dspy.OutputField(
        desc="""A strategy, formatted as a string, outlining the additional information needed to employ relevant tools effectively. Specify each tool's name and the precise data required for its operation.
        Responses should be structured as a list within a string, e.g., 'Provide detailed clinical history for score X computation, Upload new imaging for Y analysis using tool Z.'
        Unacceptable answer: [Here is a list: 1. ..., second ...]. Valid answers are in this format: '.., ..., ...'.
        When refering to a specific tool, like 'gen_radiology_report' include the description of the tool in your output as given in tools.
        Avoid general recommendations and ensure no redundancy with previously utilized tools or data. The suggested tools must not be included in the 'tool_results' field!"""
    )


class Suggestions2(dspy.Signature):
    """Refine provided recommendations to guide the user in supplying additional details or data crucial for leveraging specific analytical tools, elucidating the reasons for these requests.
    Given broad suggestions on which tools to use, prompt the user for the exact data inputs required for these tools, ensuring that the suggestions are seamlessly integrated with the initial response.
    All tools at your disposal are detailed in the 'tools' field.
    Ensure that these refined suggestions seamlessly extend the initial response, without reiterating previous content or introducing new concepts.
    """

    response = dspy.InputField(
        desc="A comprehensive and accurate response, thoroughly backed by relevant sources."
    )
    tool_results = dspy.InputField(
        desc="Accurate outcomes from supplementary patient diagnostics."
    )
    tools = dspy.InputField(
        desc="An inventory of available analytical tools, detailing their input requirements and appropriate scenarios for their application."
    )
    suggestions = dspy.OutputField(
        desc="""Crafted suggestions that prompt the user for specific information or data essential for certain tools, while mirroring the depth of detail found in the initial recommendations.
        Frame these suggestions as a coherent extension of the initial response, avoiding direct mention of tool names. Instead, imply the utility of the tools by stating with something like 'Utilizing my array of resources, I can further assist by doing x, if you could provide my with y' where x succinctly describes the function or benefit of the tool and y the required input.
        Specify the exact data inputs needed from the user to enable the effective use of analytical tools.
        Do not ask for access to any of the tools, they are actually provided to you already - instead ask for the necessary inputs the user needs to provide, like the path to the images for the genetic modeling.
        Limit the entire response to 2-3 sentences to maintain conciseness and focus.
        If you encounter conflicts in the data (e.g. unclear reports or contradictory statements between the given information and your findings), state them clearly.
        """
    )
