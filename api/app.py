"""
FastAPI 主应用入口
提供文档导入（SSE 实时进度）和语义检索（SSE 流式结果）接口
"""
import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from processor.import_process.base import setup_logging
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_process.nodes.node_md_img import NodeMDImg
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from api.query_engine import QueryEngine

setup_logging()
logger = logging.getLogger("api")

# 全局任务状态存储（生产环境应使用 Redis）
_task_store: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("FastAPI 服务启动")
    yield
    logger.info("FastAPI 服务关闭")


app = FastAPI(
    title="知识库导入与检索服务",
    description="基于 FastAPI + SSE 的文档导入（实时进度）和语义检索（流式结果）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_query_engine: QueryEngine | None = None


def get_query_engine() -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine()
    return _query_engine


def _sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_import_pipeline(task_id: str, file_path: str, file_title: str) -> None:
    """
    后台执行导入流水线，串联各个节点并通过 _task_store 推送进度
    """
    store = _task_store[task_id]

    # 输出目录
    output_dir = os.path.join("D:/output", file_title)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化状态 - 匹配 ImportGraphState 字段名
    state = {
        "import_file_path": file_path,
        "file_title": file_title,
        "file_dir": output_dir,
    }

    # 流水线节点定义：(name, instance, pre_hook)
    pipeline = [
        ("入口校验", NodeEntry(), lambda s: s),
        ("PDF转MD", NodePDFToMD(), lambda s: {**s, "pdf_path": s["import_file_path"]}),
        ("图片处理", NodeMDImg(), lambda s: s),
        ("文档切分", NodeDocumentSplit(), lambda s: s),
        ("商品名识别", NodeItemNameRecognition(), lambda s: s),
        ("向量化", NodeBGEEmbedding(), lambda s: s),
        ("入库Milvus", NodeImportMilvus(), lambda s: s),
    ]

    try:
        for node_name, node, hook in pipeline:
            store["current_node"] = node_name
            store["logs"].append(f"[{node_name}] 开始...")
            state = hook(state)
            state = node(state)
            store["logs"].append(f"[{node_name}] 完成")
            store["progress"] += 1

        store["status"] = "completed"
        store["result"] = {
            "chunks_count": len(state.get("chunks", [])),
            "item_name": state.get("item_name", ""),
        }
        store["logs"].append("✅ 导入完成")

    except Exception as e:
        store["status"] = "failed"
        store["error"] = str(e)
        store["logs"].append(f"❌ 错误: {str(e)}")
        logger.exception("导入流水线失败")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/import")
async def import_document(
    file: UploadFile = File(...),
    file_title: str = Form(""),
):
    """
    接收文件上传，返回 task_id，前端通过 SSE 监听进度
    """
    task_id = str(uuid.uuid4())

    # 保存上传文件
    import tempfile
    import os

    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{task_id}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    title = file_title or (file.filename or "unknown")

    # 初始化任务状态
    _task_store[task_id] = {
        "task_id": task_id,
        "status": "running",
        "progress": 0,
        "total": 6,
        "current_node": "",
        "logs": [f"开始导入: {title}"],
        "error": None,
        "result": None,
    }

    # 后台启动流水线
    asyncio.create_task(_run_import_pipeline(task_id, tmp_path, title))

    return {"task_id": task_id, "status": "running"}


@app.get("/api/import/{task_id}/progress")
async def import_progress(task_id: str) -> StreamingResponse:
    """
    SSE 实时推送导入进度
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        if task_id not in _task_store:
            yield _sse_event("error", {"message": "任务不存在"})
            return

        store = _task_store[task_id]
        last_log_index = 0

        while True:
            # 推送进度
            yield _sse_event("progress", {
                "task_id": task_id,
                "status": store["status"],
                "progress": store["progress"],
                "total": store["total"],
                "current_node": store["current_node"],
            })

            # 推送新增日志
            while last_log_index < len(store["logs"]):
                yield _sse_event("log", {"message": store["logs"][last_log_index]})
                last_log_index += 1

            # 完成或失败则结束
            if store["status"] in ("completed", "failed"):
                if store["status"] == "completed":
                    yield _sse_event("done", store.get("result", {}))
                else:
                    yield _sse_event("error", {"message": store.get("error", "未知错误")})
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/search")
async def search(
    q: str,
    top_k: int = 5,
    rerank: bool = True,
) -> StreamingResponse:
    """
    SSE 流式检索：逐条返回检索结果
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        engine = get_query_engine()
        try:
            yield _sse_event("start", {"query": q, "top_k": top_k})

            results = engine.search(q, top_k=top_k, rerank=rerank)

            for idx, result in enumerate(results):
                yield _sse_event("result", {
                    "rank": idx + 1,
                    "title": result.get("title", ""),
                    "content": result.get("content", "")[:500],
                    "item_name": result.get("item_name", ""),
                    "file_title": result.get("file_title", ""),
                    "score": result.get("score", 0),
                })
                await asyncio.sleep(0.05)  # 轻微延迟，让前端有逐条出现的效果

            yield _sse_event("done", {"total": len(results)})

        except Exception as e:
            logger.exception("检索失败")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/chat")
async def chat(
    q: str,
    top_k: int = 4,
    rerank: bool = True,
) -> StreamingResponse:
    """
    SSE RAG 对话：先检索知识库，再用 LLM 生成回答
    事件：source(×N) → answer_token(×N) → done
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from utils.llm_utils import get_llm_client

    CHAT_SYSTEM_PROMPT = "你是一个知识库问答助手。根据提供的参考文档片段回答问题。要求：1. 仅基于参考内容回答，不编造 2. 参考内容不足则明确说明 3. 简洁准确，使用中文"

    async def event_generator() -> AsyncGenerator[str, None]:
        engine = get_query_engine()
        try:
            yield _sse_event("status", {"message": "正在检索知识库..."})
            results = engine.search(q, top_k=top_k, rerank=rerank)

            # 发送来源
            for idx, r in enumerate(results):
                yield _sse_event("source", {
                    "rank": idx + 1,
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:300],
                    "file_title": r.get("file_title", ""),
                    "score": r.get("score", 0),
                })

            if not results:
                yield _sse_event("answer_token", {"token": "未找到相关知识，请尝试换个问题。"})
                yield _sse_event("done", {})
                return

            # 构建上下文并流式调用 LLM
            yield _sse_event("status", {"message": "正在生成回答..."})
            context = "\n\n".join(f"[来源{i+1}] {r['content']}" for i, r in enumerate(results))

            llm = get_llm_client()
            messages = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=f"参考文档：\n{context}\n\n问题：{q}"),
            ]
            async for chunk in llm.astream(messages):
                token = (chunk.content or "").strip()
                if token:
                    yield _sse_event("answer_token", {"token": token})

            yield _sse_event("done", {"source_count": len(results)})

        except Exception as e:
            logger.exception("对话失败")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
