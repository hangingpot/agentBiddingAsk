from __future__ import annotations

import json
import zipfile
from pathlib import Path

from bidding_qa.corpus import load_corpus
from bidding_qa.index_store import build_index


def test_faq_loaded_once_and_keeps_report_question(corpus):
    questions = [chunk.question for chunk in corpus if chunk.kind == "faq"]
    assert len(questions) == len(set(questions))
    assert len(questions) >= 99
    assert "如何成为易派客平台会员？" in questions
    assert "如何查看产品质量评价报告？" in questions
    membership = [chunk for chunk in corpus if chunk.question == "如何成为易派客平台会员？"]
    assert len(membership) == 1
    assert "faq.json" in membership[0].source


def test_extra_files_are_picked_up_without_code_changes(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "补充说明.md").write_text(
        "电招平台夜间维护窗口为周日 0 点至 2 点，期间不能远程解密。\n",
        encoding="utf-8",
    )
    manual = {
        "items": [
            {"question": "测试用：标书副本存哪里？", "answer": "测试答案：保存在投标文件制作工具的本地工程目录。"}
        ]
    }
    (data / "extra.json").write_text(json.dumps(manual, ensure_ascii=False), encoding="utf-8")
    archive = data / "手册.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("inner/note.txt", "内部说明：WZ 开头是电招标编号，ZZ 开头是 EC 电子标编号。")

    chunks = load_corpus(data)
    blob = "\n".join(chunk.text for chunk in chunks)
    assert "夜间维护窗口" in blob
    assert "标书副本存哪里" in blob
    assert "WZ 开头是电招标编号" in blob
    assert any(chunk.source.endswith("手册.zip!inner/note.txt") for chunk in chunks)


def test_index_rebuilds_when_data_changes(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.json").write_text(
        json.dumps([{"q": "如何在电招平台解密？", "a": "到开标时间后在电招平台远程解密。"}], ensure_ascii=False),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    first = build_index(data, index_dir)
    assert "远程解密" in first.search("电招平台如何解密", k=1)[0].chunk.text
    cached = build_index(data, index_dir)
    assert len(cached.chunks) == len(first.chunks)

    (data / "b.md").write_text("补充说明：WZ 编号属于电招标。\n", encoding="utf-8")
    rebuilt = build_index(data, index_dir)
    blob = "\n".join(chunk.text for chunk in rebuilt.chunks)
    assert "WZ 编号属于电招标" in blob
    assert len(rebuilt.chunks) > len(first.chunks)
