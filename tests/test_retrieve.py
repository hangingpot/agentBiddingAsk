from __future__ import annotations


def _titles(index, query: str, k: int = 3) -> list[str]:
    return [hit.chunk.title for hit in index.search(query, k=k)]


def test_exact_faq_is_first(index):
    titles = _titles(index, "如何成为易派客平台会员？")
    assert titles[0] == "如何成为易派客平台会员？"


def test_paraphrase_hits_password_faq(index):
    titles = _titles(index, "忘记易派客登录密码了怎么找回")
    assert any("密码" in title for title in titles[:2])


def test_company_code_and_disabled_account(index):
    assert any("石化编码" in title for title in _titles(index, "本企业的石化编码在哪里看"))
    assert any("账号已停用" in title for title in _titles(index, "登录电招平台提示账号已停用"))


def test_payment_failure_and_quality_report(index):
    assert any("扫码支付" in title for title in _titles(index, "微信扫码支付CA证书费用失败怎么办"))
    assert any("产品质量评价报告" in title for title in _titles(index, "产品质量评价报告在哪里查看"))


def test_compound_search_returns_both_ca_topics(index):
    hits = index.search_many(
        [
            "投标用CA数字证书的初始密码是什么？过期超过6个月怎么处理？",
            "投标用CA数字证书的初始密码是什么？",
            "CA数字证书 过期超过6个月怎么处理？",
        ],
        k=4,
    )
    blob = " ".join(hit.chunk.title + hit.chunk.text for hit in hits)
    assert "初始密码" in blob
    assert "6个月" in blob


def test_half_year_means_six_months_for_ca(index):
    titles = _titles(index, "CA过期超过半年了怎么办，费用多少")
    assert any("有效期" in title for title in titles[:2])


def test_unrelated_query_scores_below_answer_threshold(index):
    from bidding_qa.pipeline import MIN_CONFIDENT

    weather = index.search("今天北京天气怎么样", k=1)[0]
    recipe = index.search("红烧肉怎么做才好吃", k=1)[0]
    assert weather.score < MIN_CONFIDENT
    assert recipe.score < MIN_CONFIDENT

    member = index.search("如何成为易派客平台会员？", k=1)[0]
    assert member.score >= MIN_CONFIDENT
