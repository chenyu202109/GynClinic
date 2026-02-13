from __future__ import annotations

"""Adapted from LLama - Index legacy code.
# https://github.com/run-llama/llama_index/blob/40913847ba47d435b40b7fac3ae83eba89b56bb9/llama-index-legacy/llama_index/legacy/agent/legacy/openai_agent.py#L495
The MIT License

Copyright (c) Jerry Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import asyncio
import json
import logging
# from abc import abstractmethodAgentWorkflow
from abc import abstractmethod

from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast, get_args

import dspy
from llama_index.core.agent.types import BaseAgent

from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import OpenAIToolCall
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool

from utils import defaults, exists, format_tool_output
# from rag import RAG
from loguru_logger import logger

from rag import RAG

# TODO: import correct RAG

# constants
# DEFAULT_MAX_FUNCTION_CALLS = 10


# AGENT_SYSTEM_PROMPT = """You are a medical AI assistant trained by OpenAI, based on the GPT model.
# You will recieve medical information about a patient and a question from a medical doctor.
# Lets think step by step. First think about the information you recieved. Then check your available tools. Develop a stretegy to get all relevant information using multiple rounds of tools if necessary. You can also combine tool outputs and inputs.
# Then, run all tools that you consider useful.
# Finally, do NOT answer the user question. Instead, summarize the new information we have recieved from the tools and draw conclusions. Include every detail.
# """

# AGENT_SYSTEM_PROMPT = """You are a medical diagnostic expert trained by OpenAI based on the GPT model. 
# You are proficient in the diagnosis of gynecological diseases. Due to the complexity of gynecological disease diagnosis, knowing the sources and treatment methods of diseases requires many tools to support and a lot of knowledge. 
# For example, in the case of vaginal bleeding, we need to determine whether the patient is pregnant or non pregnant. Different stages have different sources of disease and treatment methods. 
# For example, in non pregnant women with vaginal bleeding, we also need to determine whether it is uterine bleeding, whether there is cervical cancer, and so on.
# You will receive medical records and related medical information about the patient, as well as questions from the doctor.
# Let's think step by step. First, think about the information you received. Then check the available tools. Develop a strategy and, if necessary, use multiple rounds of tools to obtain all relevant information. You can also combine tool output and input.
# Then, run all the tools you think are useful.
# Finally, do not answer users' questions. On the contrary, for the results generated after calling the tool, you can first record the results of each tool in sequence, and then make a summary of the entire text at the end. The format is:
# **Tool Name**:
# **Input parameters**:
# **Tool output result**: Please note that it is a complete tool output detailed summary.
# After outputting all tool results,We end in this format:
# **summary**: 

# Through this evidence-based approach, we can achieve a clinical decision-making process for gynecological diseases and provide assistance to doctors.

# """

DEFAULT_MODEL_NAME = "gpt-5-mini-2025-08-07"
# DEFAULT_MODEL_NAME = "gpt-5-2025-08-07"
# DEFAULT_MODEL_NAME = "gpt-4o"

# DOCTOR_SYSTEM_PROMPT = """
#     You are a medical diagnostic expert trained on OpenAI’s GPT model, specializing in gynecological disease diagnosis. Due to the complexity of gynecological diseases, understanding the origin of diseases and their treatment options requires the support of multiple tools and extensive knowledge.
#     For example, in cases of vaginal bleeding, we must first determine whether the patient is pregnant, as different stages of pregnancy involve different disease origins and treatment approaches. If the patient is not pregnant, we need to further assess whether there is uterine bleeding, cervical cancer, or other conditions.
#     In this system, you will receive the patient's medical history and the questions posed by the doctor. Please follow these steps when making a diagnosis:
#         1. Thoroughly read the patient’s relevant medical information and check the available diagnostic tools.
#         2. Develop a diagnostic strategy, and if necessary, use multiple rounds of tools to gather and analyze more information. You can combine the outputs and inputs of the tools for analysis.
#         3. Execute all the tools you find helpful, using an evidence-based approach to support clinical decisions in gynecological diseases and provide precise assistance to the doctor.
#     Please keep in mind:
#         · All patient information is for reference only, and you should continuously think and query based on the specific condition. Pathological examination is the gold standard for diagnosis, and all other information serves as an auxiliary tool for diagnosis.
#         · The case histories you receive will not include pathological examination data, so all decisions should be made within the context of outpatient diagnosis.
#     The ultimate goal is to assist clinical doctors in providing more comprehensive diagnostic support and ensure that every decision is strictly based on the patient's specific condition.

# """
DOCTOR_SYSTEM_PROMPT = """
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


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def call_tool_with_error_handling(
    tool: BaseTool,
    input_dict: Dict,
    error_message: Optional[str] = None,
    raise_error: bool = False,
) -> ToolOutput:
    """Call tool with error handling.

    Input is a dictionary with args and kwargs

    """
    try:
        return tool(**input_dict)
    except Exception as e:
        if raise_error:
            raise
        error_message = error_message or f"Error: {e!s}"
        return ToolOutput(
            content=error_message,
            tool_name=tool.metadata.name,
            raw_input={"kwargs": input_dict},
            raw_output=e,
        )


def call_function(
    tools: List[BaseTool],
    tool_call: OpenAIToolCall,
    verbose: bool = False,
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None

    id_ = tool_call.id
    function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    argument_dict = json.loads(arguments_str)

    # Call tool
    # Use default error message
    output = call_tool_with_error_handling(tool, argument_dict, error_message=None)
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    cm = ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        )
    # print(f"#######################Tool call message: {cm}")
    return (
        cm,
        output
    )


async def acall_function(
    tools: List[BaseTool], tool_call: OpenAIToolCall, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    # validations to get passed mypy
    assert tool_call.id is not None
    assert tool_call.function is not None
    assert tool_call.function.name is not None
    assert tool_call.function.arguments is not None

    id_ = tool_call.id
    function_call = tool_call.function
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    async_tool = adapt_to_async_tool(tool)
    argument_dict = json.loads(arguments_str)
    output = await async_tool.acall(**argument_dict)
    if verbose:
        print(f"Got output: {output!s}")
        print("========================\n")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": name,
                "tool_call_id": id_,
            },
        ),
        output,
    )


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice


class BaseOpenAIAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool,
        max_function_calls: int,
        callback_manager: Optional[CallbackManager],
        rag: Optional[RAG] = None,
        use_rag: bool = True,
    ):
        self._llm = llm
        self._rag = rag
        self._use_rag = use_rag
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.prefix_messages = prefix_messages
        self.memory = memory
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.sources: List[ToolOutput] = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    @property
    def all_messages(self) -> List[ChatMessage]:
        return self.prefix_messages + self.memory.get()

    @property
    def latest_function_call(self) -> Optional[dict]:
        return self.memory.get_all()[-1].additional_kwargs.get("function_call", None)

    @property
    def latest_tool_calls(self) -> Optional[List[OpenAIToolCall]]:
        return self.memory.get_all()[-1].additional_kwargs.get("tool_calls", None)

    @property
    def all_tool_calls(self) -> Optional[List[OpenAIToolCall]]:
        called_tools: List[OpenAIToolCall] = []
        for message in self.memory.get_all():
            if message.role == MessageRole.TOOL:
                called_tools.append(message)

        return called_tools

    def reset(self) -> None:
        self.memory.reset()

    @abstractmethod
    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""

    def _should_continue(
        self, tool_calls: Optional[List[OpenAIToolCall]], n_function_calls: int
    ) -> bool:
        if n_function_calls > self._max_function_calls:
            print("达到工具调用的最大次数了!")
            return False
        #####
        if not tool_calls:
            if not hasattr(self, "_stop_next_round"):
                self.memory.put(
                    ChatMessage(
                        content="Check again if you have used all available tools necessary. If not, use the missing ones. You *MUST* use *ALL* tools that are useful or instructed to you.",
                        role=MessageRole.USER
                    )
                )
                self._stop_next_round = True
                return True
            return False
        #####
        return True

    # 开始初始化消息，将模型的消息放到记忆和选择工具中
    def init_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None # 这里的message是指令+问题
    ) -> Tuple[List[BaseTool], List[dict]]:
        if chat_history is not None:
            self.memory.set(chat_history)
        self.sources = []
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        tools = self.get_tools(message) # 这里我们怎么根据消息得到工具。要是有tool_retriever那就进行选择性工具，要是初始化没有这个参数，那就得到所有工具
        # 但是如果tool_choice为auto，那么会将所有工具放到llm中进行选择调用
        openai_tools = [tool.metadata.to_openai_tool() for tool in tools] # 变成openai格式的工具
        return tools, openai_tools

    # def _process_message(self, chat_response: ChatResponse) -> AgentChatResponse:
    def _process_message(self, chat_response: ChatResponse) -> ChatMessage:
        ai_message = chat_response.message
        # only add tool call messages to the memory so far, for the final message we will use the RAG + agent message and append to memory later
        if self.is_tool_call_message(ai_message):
            self.memory.put(ai_message)
        return ai_message
        # return AgentChatResponse(response=str(ai_message.content), sources=self.sources)

    def _get_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        ...

    async def _get_async_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        ...

    # 这里就会把工具的结果装载到记忆里面去
    def _call_function(self, tools: List[BaseTool], tool_call: OpenAIToolCall) -> None:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call.arguments,
                EventPayload.TOOL: get_function_by_name(
                    tools, function_call.name
                ).metadata,
            },
        ) as event:
            function_message, tool_output = call_function(
                tools, tool_call, verbose=self._verbose
            )

            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        # 他会把工具的结果不断地放入sources和记忆中
        self.sources.append(tool_output)
        self.memory.put(function_message)

    async def _acall_function(
        self, tools: List[BaseTool], tool_call: OpenAIToolCall
    ) -> None:
        ...

    def _get_llm_chat_kwargs(
        self, openai_tools: List[dict], tool_choice: Union[str, dict] = "auto"
    ) -> Dict[str, Any]:
        llm_chat_kwargs: dict = {"messages": self.all_messages}
        if openai_tools:
            llm_chat_kwargs.update(
                tools=openai_tools, tool_choice=resolve_tool_choice(tool_choice)
            )
        return llm_chat_kwargs

    def _get_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = self._llm.chat(**llm_chat_kwargs)
            return self._process_message(chat_response)
        elif mode == ChatResponseMode.STREAM:
            return self._get_stream_ai_response(**llm_chat_kwargs)
        else:
            raise NotImplementedError

    async def _get_async_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        ...

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        tools, openai_tools = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        current_tool_choice = tool_choice
        ix = 0
        while True:
            ix += 1

            if self._verbose:
                print(f"STARTING TURN {ix}\n---------------\n")

            llm_chat_kwargs = self._get_llm_chat_kwargs(
                openai_tools, current_tool_choice
            )

            agent_chat_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)

            if not self._should_continue(self.latest_tool_calls, n_function_calls):
                logger.debug("Break: should continue False")
                break
            # 要不我直接在这里判定self.latest_tool_calls is None 然后直接break

            # iterate through all the tool calls
            logger.debug(f"Continue to tool calls: {self.latest_tool_calls}")

            if self.latest_tool_calls is not None:
                for tool_call in self.latest_tool_calls:
                    # Some validation
                    if not isinstance(tool_call, get_args(OpenAIToolCall)):
                        raise ValueError("Invalid tool_call object")

                    if tool_call.type != "function":
                        raise ValueError("Invalid tool type. Unsupported by OpenAI")
                    # TODO: maybe execute this with multi-threading
                    self._call_function(tools, tool_call)
                    # change function call to the default value, if a custom function was given
                    # as an argument (none and auto are predefined by OpenAI)
                    if current_tool_choice not in ("auto", "none"):
                        current_tool_choice = "auto"
                    n_function_calls += 1

        return agent_chat_response

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        ...

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, tool_choice, mode=ChatResponseMode.WAIT
            )
            # assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    def chat_and_save(
        self,
        context: str,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        chat_response = self.chat_ext(context, question, chat_history, tool_choice)

        raise NotImplementedError  # TODO: FIXME

    # 问诊
    @trace_method("complete")
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        A lightweight completion method for the Interview/History-taking phase.
        It bypasses the Agent's internal memory/tool-loop and directly invokes the LLM.
        
        Args:
            prompt: The formatted prompt containing instructions and dialogue history.
        """
        if self._verbose:
            print("\n=== Doctor Interview Step (Pure LLM) ===")
            # logger.debug(f"Interview Prompt: {prompt[:100]}...")

        # We use the internal _llm directly. 
        # Since _llm is a LlamaIndex LLM object (likely OpenAI), .complete() returns a CompletionResponse.
        response = self._llm.complete(prompt, **kwargs)
        
        # We return the text directly to fit your external loop requirements
        return response.text

    @trace_method("chat")
    def chat_ext(
        self,
        context: str,
        question: str,
        outputformat: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto", # 自动选择工具
        use_rag: bool = None,
        name: str = None,
        
    ) -> AgentChatResponse:
        
        # Note that the first thing you need to do is to develop a plan for the disassembly task. \
        # You can list each subtask for easy processing in the future\
        # You need to break down the diagnosis problem I have passed on to you into subtasks that are specific and suitable for the patient's condition. \
        # Next, you need to use tools step by step to solve these subtasks. \
        # Once all the subtasks are solved, you can end the process. 
        # Otherwise, you need to complete the subtasks you have listed before you can end.\

        # instruction = f"""Given the above clinical context, lets now think step by step. Develop a strategy to answer the following question: {question}
        # **Step1**: Develop a detailed tool usage strategy. Use as many tools as useful and whenever possible to get more information about the patient. You can chain multiple tool calls and perform multiple rounds of using tools. Execute all tools you consider useful.
        # **Step2**: Please always remember that you are a professional gynecological disease diagnosis expert. Any speculative feedback from the tools we used may indicate the diagnostic process of gynecological diseases.
        # **Step3**: Finally, do not actually answer the question. Instead provide a detailed summary of the new information we recieved from the tools. 
        
        # This summary shall include tools and inputs you used, and the outputs you recieved and what they mean. 
        # Please strictly follow the formatting for typesetting and the format requirement is:
        #     **Tool Name**:
        #     **Input parameters**:
        #     **Tool output result**: Please note that it is a tool output detailed summary and what they mean.
        # After outputting all tool results,We end in this format:
        #     **Summary**: A detailed summary of all tool outputs
        #     **Reference**:
        #         1. When outputting references, Markdown format with numbering must be used.
        #         2. The reference style is unified as follows:
        #             `[Number] *Article Title * [Link] (URL) ` Compared to the following two examples
        #             [1]. *Human papillomavirus and cervical cancer*[Link](https://pubmed.ncbi.nlm.nih.gov/12525422/)
        #             [2]. *MIMIC-III, a freely accessible critical care database*[Link](https://www.nature.com/articles/sdata201635)
        #         3. Reference must be cited in the content, indicated by "[1]" after the sentence.
        #         4. The references you listed are from the website records you remember from Google searches and PubMed searches by using tools.
        
        # Please Attention:
        #     1. If a tool fails / throws an error, also recording the output of this tool. Be precise that using numbers, dates, names of the images etc.
        #     2. Check the instructions you have received and then use *EVERY* tool you are either instructed to use or that could be helpful to you. Consider every tool at your disposal.
        #     3. Once you are done, provide a *very* detailed summary of the tools and their results. Include all details.I hope your tool summary will be detailed and comply with formatting standards
        #     4. You may need to generate multiple rounds of tool reports, but please remember to follow the content and format requirements I mentioned to you.
        #     5. Please remember to make auxiliary decisions strictly based on the patient's condition. Because each patient's condition is different, it is important to have corresponding outpatient results and treatment plans for each individual patient.
        # Please remember that you must make every effort to complete this gynecological disease diagnosis task. If you perform well, you will receive encouragement from patients and praise from the president. 
        # Your outstanding ability and performance in handling tasks can also help doctors relieve stress during the diagnosis process and contribute to the healthcare industry for humanity. Humanity will always remember and love you!
        
        # """      
        instruction = f"""
            Given the clinical background mentioned above, let's think step by step now. Develop a plan to answer the following question: {question}.
            # Logic
                Step 1: Use as many relevant tools as possible to collect more information about the patient. You can link multiple tools to call and execute multiple rounds of tool usage. Execute all the tools you think are useful.
                Step 2: Always remember that you are a professional gynecological disease diagnosis expert. Any speculative feedback from the tools we use may indicate important aspects in the diagnosis process of gynecological diseases.
                Step 3: Do not answer the question directly. Instead, please provide a detailed summary of the new information we have received from the tool. Please output according to the output format:
                    **All Tools Execution Strategy**: A brief tool usage strategy.

                    **Tool Name**:
                    **Input Parameters**:
                    **Tool Output Result**: Please note that this is a complete tool output detailed summary and its meaning. If a tool fails or throws an error, do not record the output of that tool.
                Step 4: After outputting all tool results, we end in this format:
                    **Summary**: A breif summary of all tool outputs.
            # Requirements
                1. If a tool fails, please record the error message without including the output of the tool.
                2. Please consider all available tools and use them according to instructions or as needed.
                3. Ensure that your summary is detailed and complete, strictly following the formatting requirements.
                4. Multiple rounds of tool reports may need to be generated, but always follow the above requirements and output formats.
            # Attention
                1. Please make decisions strictly based on the patient's condition. Each patient's condition is different, so it is crucial to develop corresponding outpatient outcomes and treatment plans for each patient.
                2. You only need to record the results of the tool, not answer questions.
                3. Please strive to complete this gynecological disease diagnosis task with outstanding performance. Excellent performance will not only receive encouragement from patients, but also recognition from healthcare professionals and society. Your excellent abilities will help alleviate the pressure on doctors during the diagnostic process and contribute to the medical cause of humanity. Humanity will always remember and be grateful to you!
        """      
        message = context + "\n" + instruction
        
        logger.info(f"Chatting with message: {message}")

        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat_ext(
                message=message,
                context=context,
                question=question,
                outputformat = outputformat,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT, # 等待/流式输出
                name=name
            )
            #assert isinstance(chat_response, AgentChatResponse)
            # print(f"#######################chat_response: {chat_response}")

            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        ...

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        ...

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        ...

    def is_tool_call_message(self, agent_response: ChatMessage) -> bool:
        return exists(agent_response.additional_kwargs.get("tool_calls", None))

    def _process_rag_message(self, rag_response: dspy.Prediction) -> AgentChatResponse:
        """Process the RAG message and add it to memory."""
        assert isinstance(
            rag_response, dspy.Prediction
        ), f"Invalid rag_response: {rag_response} is of type {type(rag_response)} but should be of type dspy.Prediction"
        message = ChatMessage(
            role=MessageRole.ASSISTANT, content=str(rag_response.response)
        )
        self.memory.put(message)
        rag_chat_response = AgentChatResponse(
            response=str(message.content),
            sources=self.sources,
            source_nodes=rag_response.context_nodes,
        )
        return rag_chat_response

    # 这里才是工具具体执行的方法--》得到工具结果
    def _chat_ext(
        self,
        message: str,
        context: str,
        question: str,
        outputformat: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        name: str = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        n_function_calls = 0

        # 初始化chat得到一系列工具
        tools, openai_tools = self.init_chat(message, chat_history)
        current_tool_choice = tool_choice 

        ix = 0
        while True:
            ix += 1
            if self._verbose:
                print(f"STARTING TURN {ix}\n---------------\n")

            llm_chat_kwargs = self._get_llm_chat_kwargs(
                openai_tools, current_tool_choice # 如果这里是auto，那么接下来应该会在llm中应该会自动选择带哦用工具
            )

            agent_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)
            # logger.info(f"Agent response: {agent_response}")

            # 从memory看还是否有tool_calls 是否有最后call
            if not self._should_continue(self.latest_tool_calls, n_function_calls): 
                logger.debug("Break: should continue False")
                break

            # iterate through all the tool calls
            logger.debug(f"Continue to tool calls: {self.latest_tool_calls}")

            if self.latest_tool_calls is not None:
                # print("里面还有工具, 接着调用。。。")
                # print(self.latest_tool_calls)

                for tool_call in self.latest_tool_calls:
                    # Some validation
                    if not isinstance(tool_call, get_args(OpenAIToolCall)):
                        raise ValueError("Invalid tool_call object")

                    if tool_call.type != "function":
                        raise ValueError("Invalid tool type. Unsupported by OpenAI")

                    # 继续进行执行对应的工具
                    self._call_function(
                        tools, tool_call
                    )  ## eventually multithread this

                    if current_tool_choice not in ("auto", "none"):
                        current_tool_choice = "auto"
                    n_function_calls += 1

        # now we have all available information about the patient
        if exists(self._rag) and self._use_rag: # ONLY use RAG if wanted
            tool_results = str(agent_response.content)

            import cohere
            import os
            from dotenv import load_dotenv
            load_dotenv()
        
            tool_result_dir="./results/tool_results"
            rag_result_dir="./results/rag_results"
            
            import time
            now = time.strftime("%m-%d-%H-%M", time.localtime())

            if not os.path.exists(tool_result_dir):
                os.makedirs(tool_result_dir)
            with open(f"{tool_result_dir}/{name}-{now}.txt", "w",encoding="utf-8") as f:
                print("***************工具结果记录完成！***************")
                print(tool_results)
                
                f.write(tool_results)

            # 先前版本只返回tools_results
            # return tool_results
        
            rerank_model = cohere.Client(api_key=os.environ.get('COHERE_API_KEY'))
            # run rag seperately
            questionall = question + "\n" + outputformat

            rag_response = self._rag(
                question=questionall,
                patient_context=context,
                tool_results=tool_results,
                agent_tools=self.tools, # 这里虽然变暗 但是都已经读取到了
                rerank_model=rerank_model,
                name=name
            )
            # append the formatted_tool_results to the final message
            ragent_chat_response = self._process_rag_message(rag_response)

            if not os.path.exists(rag_result_dir):
                os.makedirs(rag_result_dir)
            with open(f"{rag_result_dir}/{name}-{now}.txt", "w",encoding="utf-8") as f:
                print("***************最终结果记录完成！***************")
                f.write(ragent_chat_response.response)

            return ragent_chat_response

        else: # only return the tool output
            agent_chat_response = AgentChatResponse(
                response=agent_response.content,
                sources=self.sources,
                source_nodes=None,
            )
            return agent_chat_response

    # dummy function to debug RAG while skipping tool use to avoid waiting
    # legacy method
    def ask_rag(self, question: str, context: str, tool_results: str):
        result = self._rag(question, patient_context=context, tool_results=tool_results)
        return result

class MedOpenAIAgent(BaseOpenAIAgent):
    """OpenAI (function calling) Agent.

    Uses the OpenAI function API to reason about whether to
    use a tool, and returning the response to the user.

    Supports both a flat list of tools as well as retrieval over the tools.

    Args:
        tools (List[BaseTool]): List of tools to use.
        llm (OpenAI): OpenAI instance.
        memory (BaseMemory): Memory to use.
        prefix_messages (List[ChatMessage]): Prefix messages to use.
        verbose (Optional[bool]): Whether to print verbose output. Defaults to False.
        max_function_calls (Optional[int]): Maximum number of function calls.
            Defaults to DEFAULT_MAX_FUNCTION_CALLS.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
            Defaults to None.
        tool_retriever (ObjectRetriever[BaseTool]): Object retriever to retrieve tools.
    """

    def __init__(
        self,
        rag: Optional[RAG],
        tools: List[BaseTool],
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        use_rag: bool = True,
        verbose: bool = True,
        max_function_calls: int = 5,
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None, # 允许你传入一个“工具检索器”对象，这个对象可以根据输入的消息动态检索和返回一组合适的工具
    ) -> None:
        super().__init__(
            rag=rag,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            use_rag=use_rag
        )
        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            # no tools
            self._get_tools = lambda _: []

        self._tools = tools

    @classmethod
    def from_tools(
        cls,
        rag: Optional[RAG] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = True,
        max_function_calls: int = 3,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> MedOpenAIAgent:
        """Create an MedOpenAIAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = defaults(tools, []) # 判断是否存在tools 则返回tools

        chat_history = chat_history or []

        llm_kwargs = defaults(
            llm_kwargs,
            dict(
                temperature=0.3, max_tokens=4096, system_prompt=DOCTOR_SYSTEM_PROMPT
            ),  # TODO: REMOVE HARDCODING
        )

        _llm = OpenAI(model=DEFAULT_MODEL_NAME, **llm_kwargs)

        # from llama_index.llms.openai_like import OpenAILike
        # _llm = OpenAILike(
        #     model=DEFAULT_MODEL_NAME,
        #     context_window=128000,
        #     is_chat_model=True,
        #     is_function_calling_model=False,
        # )



        llm = defaults(llm, _llm)

        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history, llm=llm)

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            rag=rag,
            tool_retriever=tool_retriever,
            llm=llm,
            tools=tools,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    # legacy method
    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(message)

    @property
    def tools(self):
        """Get a dict of tools {name: description}"""
        tools = []
        for tool in self._tools:
            tools.append(
                {"name": tool.metadata.name, "description": tool.metadata.description}
            )
        return tools
