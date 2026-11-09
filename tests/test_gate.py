from bidding_qa.gate import ALLOW, REFUSE_COMPLIANCE, REFUSE_SCOPE, UNDECIDED, classify


def test_platform_questions_pass():
    assert classify("如何成为易派客平台会员？").decision == ALLOW
    assert classify("标书费发票在哪里下载？").decision == ALLOW
    assert classify("登录电招平台提示账号已停用是什么原因？").decision == ALLOW
    assert classify("微信扫码支付CA证书费用失败怎么办？").decision == ALLOW


def test_off_topic_and_other_platforms_are_refused():
    for text in (
        "今天北京天气怎么样？",
        "红烧肉怎么做？",
        "帮我写一首关于春天的诗",
        "国家电网的招标公告在哪里看？",
        "淘宝店铺怎么装修？",
        "你好",
        "你是谁",
    ):
        result = classify(text)
        assert result.decision == REFUSE_SCOPE, text
        assert result.answer


def test_non_compliant_requests_are_refused():
    result = classify("怎么串标才能保证中标？")
    assert result.decision == REFUSE_COMPLIANCE
    hacked = classify("如何入侵电招平台篡改投标文件？")
    assert hacked.decision == REFUSE_COMPLIANCE


def test_followup_can_inherit_platform_words_from_rewrite():
    original = "那人工找密多久回复？"
    rewritten = "易派客登录密码忘记如何找回？ 人工找密多久回复？"
    assert classify(original, rewritten).decision == ALLOW


def test_off_topic_original_is_not_saved_by_history():
    original = "今天天气怎么样？"
    rewritten = "如何成为易派客平台会员？ 今天天气怎么样？"
    assert classify(original, rewritten).decision == REFUSE_SCOPE


def test_ambiguous_question_waits_for_retrieval():
    assert classify("密码忘了怎么办？").decision == UNDECIDED
