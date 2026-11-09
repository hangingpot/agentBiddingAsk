from bidding_qa.dialogue import decompose, is_followup, rewrite_query


def test_short_complete_question_is_not_a_followup():
    history = [{"role": "user", "content": "如何成为易派客平台会员？"}]
    assert is_followup("如何支付标书费？", history) is False


def test_rewrite_followup_keeps_previous_topic():
    history = [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "超出范围"},
        {"role": "user", "content": "易派客登录密码忘记如何找回？"},
        {"role": "assistant", "content": "可以人工找密。"},
    ]
    rewritten = rewrite_query("那人工找密多久回复？", history)
    assert "密码" in rewritten
    assert "人工找密" in rewritten
    assert "天气" not in rewritten


def test_decompose_compound_question_keeps_topic():
    parts = decompose("投标用CA数字证书的初始密码是什么？过期超过6个月怎么处理？")
    assert len(parts) == 2
    assert "初始密码" in parts[0]
    assert "6个月" in parts[1]
    assert "ca" in parts[1].lower() or "证书" in parts[1]


def test_single_question_stays_intact():
    text = "如何成为易派客平台会员？"
    assert decompose(text) == [text]
