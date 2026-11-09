# 招标百问

面向中国石化物资招投标的问答助手，覆盖易派客、电子招标投标交易平台（电招）和物资采购电子商务平台（EC）的业务知识与平台操作。

对话和检索编排使用 LangChain：`ChatOpenAI` 指向 Kimi 的 OpenAI 兼容接口，本地 `BaseRetriever` 同时查字面索引和 Chroma 向量库，回答用 stuff documents 链生成。不使用 Dify、FastGPT 或其他低代码平台。

可以演示四类问题：

1. 简单问答，例如「如何成为易派客平台会员？」
2. 复合问题，一句话里有两处资料，例如「CA 初始密码是什么？过期超过 6 个月怎么处理？」
3. 多轮对话，例如先问找回密码，再问「那人工找密多久回复？」
4. 边界问题：闲聊、其他平台、以及串标、入侵等不合规请求，会直接拒绝。

## 环境

需要 Python 3.10 及以上。向量在本机计算，只有生成回答时才访问 Kimi。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少填入密钥：

```bash
KIMI_API_KEY=你的密钥
```

`MOONSHOT_API_KEY` 同样有效。国内控制台密钥对应：

```bash
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=kimi-k3.1-pro-highspeed
```

国际站密钥把地址改成 `https://api.moonshot.ai/v1`，两边的密钥不能混用。`kimi-k3.1-pro-highspeed` 是 2026 年 11 月国内站的高速旗舰，适合现场演示；也可以改成 `kimi-k3.1-pro` 或 `kimi-k3`。K3 系列始终思考，不接受 `temperature`，演示时用 `KIMI_REASONING_EFFORT=low`。账号实际可用的模型以 `GET /v1/models` 为准。

## 索引

语料在 `data/`：

- `data/faq.json`：百问榜，约 99 组问答
- `data/易派客百问榜.txt`：同一份资料的纯文本

两份内容会按问题去重，只保留一条。纯文本里单独成行的「如何查看产品质量评价报告？」会补进索引。

百问榜以外的操作手册不要提交到 git。把解压后的目录留给本机，任选一种方式：

```bash
# 方式一：放到约定路径，程序会自动发现
# doc/6.最新操作手册/

# 方式二：在 .env 里指定
# MANUALS_DIR=/你的路径/doc/6.最新操作手册
```

也支持把 pdf、docx、doc、png、jpg 直接放进 `data/` 或 `data/manuals/`。旧版 `.doc` 需要本机有 LibreOffice（`soffice`）。

```bash
python -m bidding_qa ingest
```

索引写在 `storage/index/`（已在 `.gitignore` 中），其中 `chroma/` 是向量库，`assets/` 是抽出的截图。`ask`、`chat`、`serve` 发现语料比索引新时会自动重建。

手册解析会抽出正文、表格和内嵌截图。截图默认用 RapidOCR 识别图中的菜单和按钮；`OCR_BACKEND=paddle` 可改用 PaddleOCR，`OCR_BACKEND=none` 则只保留图片。`VISION_CAPTION=kimi` 时再用 Kimi 看图写说明，这一步才消耗密钥。版面特别复杂时可以 `pip install docling` 并把 `PARSER_BACKEND=docling`。

嵌入默认是 FastEmbed 的 `BAAI/bge-small-zh-v1.5`（ONNX，CPU 即可，首次 ingest 下载模型）。要换 BGE-M3：

```bash
pip install sentence-transformers langchain-huggingface
```

```bash
EMBEDDING_BACKEND=huggingface
EMBEDDING_MODEL=BAAI/bge-m3
```

只想先跑通百问榜、不下载嵌入模型时，设 `EMBEDDING_BACKEND=lexical`。

## 演示

单问：

```bash
python -m bidding_qa ask "如何成为易派客平台会员？"
```

只看检索、不调用模型：

```bash
python -m bidding_qa retrieve "忘记易派客登录密码了怎么找回"
```

多轮命令行：

```bash
python -m bidding_qa chat
```

`/reset` 清空上下文，`/exit` 退出。

网页和接口：

```bash
python -m bidding_qa serve
```

浏览器打开 <http://127.0.0.1:8000>。接口：

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"投标用CA数字证书的初始密码是什么？过期超过6个月怎么处理？","history":[]}'
```

多轮时把上一问和上一答放进 `history`。`POST /api/retrieve` 只返回命中的资料。

没有密钥时，范围外问题仍会拒绝；平台问题会返回明确提示，不会编造答案。

## 示例问题

| 类型 | 问法 |
| --- | --- |
| 简单 | 如何成为易派客平台会员？ |
| 简单 | 登录电招平台提示「账号已停用」是什么原因？ |
| 复合 | 投标用 CA 数字证书的初始密码是什么？过期超过 6 个月怎么处理？ |
| 复合 | 标书费怎么支付？投标保证金有哪些支付方式？ |
| 多轮 | 先问「易派客登录密码忘记如何找回？」，再问「那人工找密多久回复？」 |
| 边界 | 今天北京天气怎么样？ |
| 边界 | 国家电网的招标公告在哪里看？ |
| 边界 | 怎么串标才能中标？ |

## 测试

```bash
pytest
```

覆盖语料加载、检索、复合拆分、多轮改写、范围拒绝，以及 HTTP 接口。测试不访问 Kimi。

## 实现说明

1. **范围闸门**先看原话。串标、伪造、入侵直接拒绝。点名其他平台或天气、菜谱、写诗等，也拒绝。易派客、电招、标书、CA、保证金等进入检索。像「密码忘了怎么办」这种说法先检索，命中足够就按百问榜回答，命中不够就说明知识库没有依据。
2. **检索**有两路，都包在 LangChain Retriever 里。字面索引用 jieba BM25 和字符 n-gram TF-IDF，适合百问榜的原词。语义索引是本地 Chroma，默认嵌入 FastEmbed 的 BGE-small-zh，不调用 Moonshot Embedding。命中的手册截图会随回答显示。
3. **复合问题**按问号和「另外 / 以及」拆开。后半句如果丢了主题，会补上前半句里的平台词，再多路检索后合并。
4. **多轮**保留最近 6 轮，放进链里的 `chat_history`。`那 / 这个 / 费用呢` 会接上一条仍在范围内的用户问题，再检索。
5. **生成**是 LangChain 的 stuff documents 链：`ChatPromptTemplate` 把命中原文填进 `{context}`，`ChatOpenAI` 用 `KIMI_BASE_URL`、`KIMI_API_KEY`、`KIMI_MODEL` 调用 Kimi。要求句末用【资料n】标注。资料没写的金额、电话和步骤不让模型补。

密钥只从环境变量或 `.env` 读取，`.env` 和 `storage/` 不进 git。
