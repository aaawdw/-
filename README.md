# 掌柜慧仓

基于 LangGraph 流水线 + FastAPI + SSE 的文档导入与语义检索系统。

## 功能

- **文档导入**：上传 PDF，流水线自动完成 PDF→MD→图片处理→切分→向量化→Milvus 入库，SSE 实时推送进度
- **语义检索**：Dense + Sparse 混合检索 + RRF 融合 + Rerank 重排序
- **RAG 对话**：流式检索 + LLM 生成回答

## 技术栈

- LangGraph（流水线编排）
- FastAPI + SSE（实时通信）
- BGE-M3（稠密+稀疏双向量）
- Milvus 2.5（向量数据库）
- PyMuPDF / pdfplumber（PDF 解析）

## 启动

```bash
# 安装依赖
uv sync

# 配置环境变量（复制 .env.example 为 .env 并填写）
cp .env.example .env

# 启动服务
PYTHONPATH=. .venv\Scripts\python.exe run.py
```

打开 http://localhost:8000

## 流水线节点

```
PDF → MD → 图片处理 → 文档切分 → 商品名识别 → BGE 向量化 → Milvus 入库
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/import` | POST | 上传 PDF，返回 task_id |
| `/api/import/{id}/progress` | GET | SSE 导入进度 |
| `/api/search?q=xxx` | GET | SSE 混合检索 |
| `/api/chat?q=xxx` | GET | SSE RAG 对话 |
