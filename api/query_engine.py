"""
查询引擎：混合检索（Dense + Sparse）+ Rerank
"""
import logging

from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client
from utils.reranker_http_utils import rerank_documents
from conf.milvus_config import milvus_config

logger = logging.getLogger("api.query")


class QueryEngine:
    """
    语义检索引擎
    流程：
        1. 查询向量化（BGE-M3 生成 dense + sparse）
        2. Milvus 混合检索（ANN dense search + sparse search）
        3. 结果融合（RRF）
        4. 重排序（可选，调用外部 reranker）
    """

    def __init__(self):
        self.client = get_milvus_client()
        self.collection = milvus_config.chunks_collection

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
        """
        执行一次完整的语义检索

        :param query: 用户查询文本
        :param top_k: 返回结果数量
        :param rerank: 是否启用重排序
        :return: 检索结果列表，每个结果包含 title/content/score 等字段
        """
        if not query.strip():
            return []

        # 1. 查询向量化
        vectors = generate_embeddings([query])
        dense_query = vectors["dense"][0]
        sparse_query = vectors["sparse"][0]

        # 2. Milvus 混合检索（先独立检索，再 RRF 融合）
        search_k = top_k * 3  # 为 RRF 融合多取一些候选
        output_fields = ["chunk_id", "title", "content", "file_title", "item_name", "parent_title"]

        # Dense 检索
        dense_results = self.client.search(
            collection_name=self.collection,
            data=[dense_query],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 256}},
            limit=search_k,
            output_fields=output_fields,
        )[0]

        # Sparse 检索
        sparse_results = self.client.search(
            collection_name=self.collection,
            data=[sparse_query],
            anns_field="sparse_vector",
            search_params={"metric_type": "IP"},
            limit=search_k,
            output_fields=output_fields,
        )[0]

        # 3. RRF 融合 (Reciprocal Rank Fusion)
        fused = self._rrf_fuse(dense_results, sparse_results, k=60)

        # 4. 可选重排序
        candidates = fused[:top_k * 2]
        if rerank and candidates:
            candidates = self._rerank(query, candidates)
            candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        return candidates

    @staticmethod
    def _rrf_fuse(dense_results, sparse_results, k: int = 60) -> list[dict]:
        """
        RRF 融合两种检索结果
        score = sum(1 / (k + rank))
        """
        scores: dict[int, float] = {}
        details: dict[int, dict] = {}

        for rank, hit in enumerate(dense_results, start=1):
            pk = hit.id  # Hit object has .id attribute
            scores[pk] = scores.get(pk, 0) + 1.0 / (k + rank)
            details[pk] = hit

        for rank, hit in enumerate(sparse_results, start=1):
            pk = hit.id
            scores[pk] = scores.get(pk, 0) + 1.0 / (k + rank)
            if pk not in details:
                details[pk] = hit

        sorted_pks = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for pk in sorted_pks:
            hit = details[pk]
            results.append({
                "chunk_id": hit.id,
                "title": hit.entity.get("title", ""),
                "content": hit.entity.get("content", ""),
                "file_title": hit.entity.get("file_title", ""),
                "item_name": hit.entity.get("item_name", ""),
                "parent_title": hit.entity.get("parent_title", ""),
                "score": round(scores[pk], 4),
            })
        return results

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """调用 reranker 对候选结果重排序"""
        try:
            passages = [c["content"] for c in candidates]
            rerank_scores = rerank_documents(query, passages)
            for idx, score in enumerate(rerank_scores):
                candidates[idx]["rerank_score"] = score
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            logger.info("Rerank 完成")
            return candidates
        except Exception as e:
            logger.warning(f"Rerank 失败，跳过: {e}")
            return candidates
