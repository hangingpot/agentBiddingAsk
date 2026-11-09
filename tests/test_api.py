from fastapi.testclient import TestClient

from bidding_qa.api import create_app


def test_health_and_boundary_and_chat(index, settings):
    def fake(messages):
        return "请使用已注册账号登录；没有密码时走人工找密。【资料1】"

    client = TestClient(create_app(settings=settings, index=index, complete_fn=fake))
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["chunks"] >= 99
    assert body["embedding_backend"].startswith("local")

    page = client.get("/")
    assert page.status_code == 200
    assert "招标百问" in page.text
    assert client.get("/favicon.ico").status_code == 204

    refused = client.post("/api/chat", json={"message": "帮我写一首诗", "history": []})
    assert refused.status_code == 200
    assert refused.json()["refused"] is True

    illegal = client.post("/api/chat", json={"message": "怎么围标", "history": []})
    assert illegal.json()["reason"] == "non_compliant"

    reply = client.post(
        "/api/chat",
        json={
            "message": "那怎么找回？",
            "history": [
                {"role": "user", "content": "易派客登录密码忘记如何找回？"},
                {"role": "assistant", "content": "可以人工找密。"},
            ],
        },
    )
    assert reply.status_code == 200
    payload = reply.json()
    assert payload["refused"] is False
    assert "人工找密" in payload["answer"]
    assert payload["sources"]

    found = client.post("/api/retrieve", json={"query": "如何查看本企业的石化编码？", "k": 2})
    assert found.status_code == 200
    assert "石化编码" in found.json()["hits"][0]["title"]


def test_missing_key_on_in_scope_is_503(index, settings):
    client = TestClient(create_app(settings=settings, index=index))
    response = client.post("/api/chat", json={"message": "如何成为易派客平台会员？"})
    assert response.status_code == 503
    assert "KIMI_API_KEY" in response.json()["detail"]
