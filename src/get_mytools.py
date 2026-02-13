from agent_tools import openai_agent_tools

def get_tools_byname(need_tools_name = []):
    mytools = []
    for tool in openai_agent_tools:
        for i in need_tools_name:
            if getattr(tool, "metadata", None) and getattr(tool.metadata, "name", "") == i:
                mytools.append(tool)
    return mytools

# res = get_tools_byname(need_tools_name = ['query_pubmed','google_search','onco_kb'])
# print(res)
# for tool in res:
#         print(getattr(tool.metadata, "name", ""))