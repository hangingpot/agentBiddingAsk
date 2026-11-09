# 招标百问

面向中国石化物资招投标的问答助手，覆盖易派客、电子招标投标交易平台（电招）和物资采购电子商务平台（EC）的业务知识与平台操作。

实现是普通 Python：本地混合检索，再用 Kimi 的 OpenAI 兼容接口组织答案。不使用 Dify、FastGPT 或其他低代码平台。

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

以后把操作手册放进 `data/` 即可，支持 `.txt`、`.md`、`.json`、`.pdf`、`.docx`，以及里面包含这些文件的 `.zip`。不需要改代码。

```bash
python -m bidding_qa ingest
```

索引写在 `storage/index/`（已在 `.gitignore` 中）。`ask`、`chat`、`serve` 发现语料比索引新时会自动重建。

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
2. **检索**是本地混合索引。jieba 分词做 BM25，字符 2～4 gram 的 TF-IDF 做余弦相似度，再用问题标题的字重合加权。余弦分是绝对分，所以无关问题不会因为归一化变成高分。默认 `EMBEDDING_BACKEND=local`，不调用 Moonshot Embedding，也不下载句向量模型。
3. **复合问题**按问号和「另外 / 以及」拆开。后半句如果丢了主题，会补上前半句里的平台词，再多路检索后合并。
4. **多轮**保留最近 6 轮。`那 / 这个 / 费用呢` 会接上一条仍在范围内的用户问题，再检索。
5. **生成**把命中的原文放进提示词，要求句末用【资料n】标注。资料没写的金额、电话和步骤不让模型补。网页同时列出命中的条目。

密钥只从环境变量或 `.env` 读取，`.env` 和 `storage/` 不进 git。
