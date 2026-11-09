from dataclasses import replace

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from bidding_qa.chain import build_answer_chain
from bidding_qa.llm import build_chat_model
from bidding_qa.retriever import LocalHybridRetriever


def test_chat_model_targets_kimi_chat_completions(settings):
    configured = replace(settings, api_key="sk-test", model="kimi-k3.1-pro-highspeed")
    model = build_chat_model(configured)
    assert model.openai_api_base == "https://api.moonshot.cn/v1"
    assert model.model_name == "kimi-k3.1-pro-highspeed"
    assert model.reasoning_effort == "low"
    assert model.temperature is None
    assert model.use_responses_api is False
    assert model.openai_api_key.get_secret_value() == "sk-test"

    older = build_chat_model(replace(configured, model="kimi-k2.6"))
    assert older.reasoning_effort is None


def test_retriever_is_langchain_and_finds_faq(index):
    retriever = LocalHybridRetriever(index=index, k=3)
    assert isinstance(retriever, BaseRetriever)
    documents = retriever.invoke("如何成为易派客平台会员？")
    assert documents
    assert isinstance(documents[0], Document)
    assert documents[0].metadata["title"] == "如何成为易派客平台会员？"
    assert "会员中心" in documents[0].page_content


def test_compound_retriever_returns_both_topics(index):
    retriever = LocalHybridRetriever(index=index, k=4)
    documents = retriever.retrieve_many(
        [
            "投标用CA数字证书的初始密码是什么？过期超过6个月怎么处理？",
            "投标用CA数字证书的初始密码是什么？",
            "CA数字证书 过期超过6个月怎么处理？",
        ]
    )
    blob = " ".join(doc.page_content + doc.metadata["title"] for doc in documents)
    assert "初始密码" in blob
    assert "6个月" in blob


def test_answer_chain_is_a_runnable(settings):
    chain = build_answer_chain(settings, complete_fn=lambda messages: "好的")
    assert isinstance(chain, Runnable)
