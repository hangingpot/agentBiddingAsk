"""命令行：索引、检索、单问、多轮对话、启动接口。"""

from __future__ import annotations

import argparse
import logging
import sys

from bidding_qa.config import load_settings
from bidding_qa.index_store import build_index
from bidding_qa.llm import LLMError, LLMNotConfigured
from bidding_qa.pipeline import answer_question


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="bidding_qa", description="招标百问：易派客 / 电招平台问答")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="把 data/ 编入本地索引")
    ingest.add_argument("--force", action="store_true", help="忽略缓存，强制重建")

    retrieve = sub.add_parser("retrieve", help="只检索，不调用模型")
    retrieve.add_argument("question")
    retrieve.add_argument("-k", type=int, default=5)

    ask = sub.add_parser("ask", help="回答一个问题")
    ask.add_argument("question")

    sub.add_parser("chat", help="多轮对话")

    serve = sub.add_parser("serve", help="启动本地网页和 HTTP 接口")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = load_settings()
    if settings.embedding_backend != "local":
        print(
            f"当前只实现了 EMBEDDING_BACKEND=local，收到的是 {settings.embedding_backend}。"
            "向量仍按本地混合索引构建。",
            file=sys.stderr,
        )

    if args.command == "ingest":
        index = build_index(settings.data_dir, settings.index_dir, force=args.force)
        print(f"索引完成，共 {len(index.chunks)} 条。目录：{settings.index_dir}")
        return 0

    if args.command == "retrieve":
        index = build_index(settings.data_dir, settings.index_dir)
        for hit in index.search(args.question, k=args.k):
            print(f"{hit.score:.3f}  tfidf={hit.tfidf:.3f}  bm25={hit.bm25:.3f}  title={hit.title_score:.3f}")
            print(f"  [{hit.chunk.id}] {hit.chunk.title}")
            print(f"  {hit.chunk.source}")
        return 0

    if args.command == "ask":
        index = build_index(settings.data_dir, settings.index_dir)
        return _print_answer(args.question, [], index, settings)

    if args.command == "chat":
        index = build_index(settings.data_dir, settings.index_dir)
        return _chat(index, settings)

    if args.command == "serve":
        import uvicorn

        from bidding_qa.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 1


def _print_answer(question: str, history: list[dict], index, settings) -> int:
    try:
        result = answer_question(question, history, index, settings)
    except LLMNotConfigured as exc:
        print(exc, file=sys.stderr)
        return 2
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 3
    print(result.answer)
    if result.sources:
        print("\n依据：")
        for source in result.sources:
            print(f"- [{source['id']}] {source['title']}（{source['source']}，{source['score']}）")
        if not result.refused:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.answer})
        return 0


def _chat(index, settings) -> int:
    print("招标百问。输入问题开始对话，/reset 清空上下文，/exit 退出。")
    history: list[dict] = []
    while True:
        try:
            question = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in {"/exit", "/quit", "exit", "quit"}:
            return 0
        if question == "/reset":
            history.clear()
            print("已清空对话。")
            continue
        code = _print_answer(question, history, index, settings)
        if code == 2:
            return code
    return 0
