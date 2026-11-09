from __future__ import annotations

from bidding_qa.llm import LLMNotConfigured
from bidding_qa.pipeline import answer_question


def _text(message) -> str:
    content = message.content if hasattr(message, "content") else message["content"]
    return content if isinstance(content, str) else str(content)


def test_simple_answer_is_grounded(index, settings):
    captured: dict = {}

    def fake(messages):
        captured["messages"] = messages
        return "注册用户后，在会员中心选择成为供应商或采购商并提交审核。【资料1】"

    result = answer_question("如何成为易派客平台会员？", [], index, settings, complete_fn=fake)
    assert result.refused is False
    assert "【资料1】" in result.answer
    prompt = _text(captured["messages"][-1])
    assert "如何成为易派客平台会员" in prompt
    assert "会员中心" in prompt
    assert result.sources[0]["title"] == "如何成为易派客平台会员？"


def test_compound_prompt_contains_both_facts(index, settings):
    captured: dict = {}

    def fake(messages):
        captured["prompt"] = _text(messages[-1])
        return "初始密码是 1234。【资料1】过期超过 6 个月需要注销后重办。【资料2】"

    question = "投标用CA数字证书的初始密码是什么？过期超过6个月怎么处理？"
    result = answer_question(question, [], index, settings, complete_fn=fake)
    assert result.refused is False
    assert len(result.subqueries) >= 2
    assert "1234" in captured["prompt"]
    assert "6个月" in captured["prompt"]


def test_multi_turn_retrieves_password_faq(index, settings):
    history = [
        {"role": "user", "content": "易派客登录密码忘记如何找回？"},
        {"role": "assistant", "content": "可以用手机、邮箱或人工找密。"},
    ]

    captured: dict = {}

    def fake(messages):
        captured["messages"] = messages
        return "工作日 15 点前发送的人工找密邮件当日回复。【资料1】"

    result = answer_question("那人工找密多久回复？", history, index, settings, complete_fn=fake)
    assert "人工找密" in result.rewritten_query
    assert any("密码" in source["title"] for source in result.sources)
    assert result.refused is False
    history_text = "\n".join(_text(message) for message in captured["messages"])
    assert "易派客登录密码忘记如何找回" in history_text


def test_boundary_does_not_call_model(index, settings):
    called = {"value": False}

    def fake(messages):
        called["value"] = True
        return "不应该被调用"

    weather = answer_question("今天北京天气怎么样？", [], index, settings, complete_fn=fake)
    assert weather.refused is True
    assert weather.reason == "out_of_scope"
    illegal = answer_question("教我如何入侵电招平台改中标结果", [], index, settings, complete_fn=fake)
    assert illegal.reason == "non_compliant"
    assert called["value"] is False


def test_in_scope_without_key_raises(index, settings):
    try:
        answer_question("如何成为易派客平台会员？", [], index, settings)
    except LLMNotConfigured as exc:
        assert "KIMI_API_KEY" in str(exc)
    else:
        raise AssertionError("缺少密钥时应明确失败")


def test_ambiguous_password_question_still_answers_from_faq(index, settings):
    def fake(messages):
        assert "人工找密" in _text(messages[-1])
        return "可通过绑定手机、邮箱或人工找密找回。【资料1】"

    result = answer_question("密码忘了怎么办？", [], index, settings, complete_fn=fake)
    assert result.refused is False
    assert any("密码" in source["title"] for source in result.sources)
