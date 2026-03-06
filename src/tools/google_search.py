"""Google Search tool spec."""

import urllib.parse
from typing import Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
import re

QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}"
)

class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""

    spec_functions = ["google_search"]

    def __init__(self, key: str, engine: str, num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        print("#######################进入Google开始查询##################")
        # url = QUERY_URL_TMPL.format(
        #     key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        # )

        # if self.num is not None:
        #     if not 1 <= self.num <= 10:
        #         raise ValueError("num should be an integer between 1 and 10, inclusive")
        #     url += f"&num={self.num}"

        # response = requests.get(url)
        # # print(response)
        # return [Document(text=response.text)]
        #########################################################################################
        num = 10
        url = QUERY_URL_TMPL.format(
            key=self.key,
            engine=self.engine,
            query=urllib.parse.quote_plus(query)
        )
        url += f"&num={num}"
        response = requests.get(url)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        web_snippets = []
        results_items = results.get("items", [])
        for idx, page in enumerate(results_items):
            source = ""
            if "displayLink" in page:
                source = "\nSource: " + page["displayLink"]

            snippet = ""
            if "snippet" in page:
                snippet_text = page["snippet"]
                # 正则去除开头日期（如 "Feb 13, 2025 ..."），如果有就去，没有就保留原文
                snippet_text = re.sub(r"^[A-Z][a-z]{2,8} \d{1,2}, \d{4} \.\.\. ?", "", snippet_text)
                snippet = snippet_text

            redacted_version = f"[{idx+1}]. [{page.get('title','')}]({page.get('link','')}){source}\nDescriptions: {snippet}"
            web_snippets.append(redacted_version)

        return("# Goolgle Search Results\n" + "\n\n".join(web_snippets))
