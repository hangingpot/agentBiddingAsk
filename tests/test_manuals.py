import io
from pathlib import Path

from docx import Document as DocxDocument
from PIL import Image
from langchain_core.documents import Document

from bidding_qa.corpus import load_corpus
from bidding_qa.embeddings import HashEmbeddings
from bidding_qa.index_store import HybridIndex, build_index
from bidding_qa.vector_store import merge_vector_hits


def _png() -> bytes:
    image = Image.new("RGB", (320, 180), "white")
    pixels = image.load()
    for x in range(0, 320, 2):
        for y in range(180):
            pixels[x, y] = (x % 255, y % 255, 40)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_docx_keeps_body_screenshot_and_ocr(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    picture = tmp_path / "shot.png"
    picture.write_bytes(_png())
    document = DocxDocument()
    document.add_paragraph("电招平台上传投标文件的入口在物资业务菜单。")
    document.add_picture(str(picture))
    document.save(data / "上传标书.docx")

    def ocr(path: Path) -> str:
        return "点击上传投标文件"

    chunks = load_corpus(data, assets_dir=tmp_path / "assets", ocr=ocr)
    blob = "\n".join(chunk.text for chunk in chunks)
    assert "物资业务菜单" in blob
    assert "点击上传投标文件" in blob
    assert "/assets/" in blob
    assert any(chunk.images for chunk in chunks)

    index = HybridIndex(chunks)
    titles = [hit.chunk.text for hit in index.search("截图里如何上传投标文件", k=2)]
    assert any("点击上传投标文件" in text for text in titles)


def test_hash_chroma_retrieves_manual_sentence(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "说明.md").write_text("电招平台夜间维护窗口写在下载中心公告里。\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    index = build_index(
        data,
        index_dir,
        embedding_backend="hash",
        embedding_model="hash",
        assets_dir=index_dir / "assets",
    )
    lexical = [
        Document(
            page_content="无关",
            metadata={"id": "other", "title": "其他", "source": "x", "score": 0.05, "excerpt": "无关", "images": []},
        )
    ]
    merged = merge_vector_hits(
        lexical,
        ["电招平台夜间维护窗口"],
        index_dir=index_dir,
        backend="hash",
        model="hash",
        k=3,
        allow_new=True,
    )
    blob = " ".join(doc.page_content for doc in merged)
    assert "夜间维护窗口" in blob
    assert any(doc.metadata["id"].startswith("doc-") for doc in merged)
    assert index.chunks


def test_hash_embedding_is_stable():
    embedder = HashEmbeddings()
    first = embedder.embed_query("标书费怎么支付")
    second = embedder.embed_query("标书费怎么支付")
    assert first == second
    assert abs(sum(value * value for value in first) - 1) < 1e-6
