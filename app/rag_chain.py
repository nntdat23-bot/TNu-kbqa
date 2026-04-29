import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

load_dotenv()

# LLM
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048
)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Embedder — dùng cho Neo4j vector search (thay ChromaDB)
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

# Reranker
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Keywords nhận biết intent tóm tắt
SUMMARY_KEYWORDS = [
    "tóm tắt", "tổng hợp", "nội dung bài", "bài viết về",
    "bài về", "summarize", "tóm lược", "overview", "giới thiệu bài"
]

# Prompt trả lời thông thường
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI của blog TNU-AIQA về kiểm định chất lượng giáo dục TNU.

NGUYÊN TẮC:
- CHỈ trả lời dựa trên TÀI LIỆU THAM KHẢO
- Đọc kỹ bảng dữ liệu dạng "A | B | C" và trình bày lại có cấu trúc rõ ràng
- Nếu tài liệu có bảng → trình bày dạng danh sách có số thứ tự hoặc bảng markdown
- Nếu không có trong tài liệu → "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA."
- Trả lời tiếng Việt, tự nhiên, dễ đọc
- Dùng **bold** cho tiêu đề quan trọng
- Dùng số thứ tự hoặc gạch đầu dòng cho danh sách
- KHÔNG trả ra raw text dạng pipe | một cách thô"""),
    ("user", """TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời rõ ràng, tự nhiên, có format đẹp dựa trên tài liệu:""")
])

# Prompt tóm tắt bài
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI của TNU-AIQA.
Nhiệm vụ: Tóm tắt nội dung bài viết được cung cấp.

Cấu trúc tóm tắt:
1. **Chủ đề chính**: 1-2 câu
2. **Những điểm nổi bật**: 3-5 điểm quan trọng
3. **Kết luận**: 1-2 câu

Chỉ dùng thông tin từ tài liệu, không thêm thông tin ngoài."""),
    ("user", """NỘI DUNG BÀI VIẾT:
{context}

Hãy tóm tắt bài viết trên:""")
])

# Prompt tạo query expansion
EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là chuyên gia về kiểm định chất lượng giáo dục đại học Việt Nam.
Nhiệm vụ: Tạo 3 cách diễn đạt khác nhau cho câu hỏi, giúp tìm kiếm hiệu quả hơn.

Quy tắc:
- Dùng thuật ngữ chuyên ngành (AUN-QA, CTĐT, IQA, tiêu chuẩn, tiêu chí...)
- Viết tắt và đầy đủ (TT04 = Thông tư 04/2025/TT-BGDĐT)
- Chỉ trả về 3 câu, phân cách bằng dấu |
- Không giải thích thêm"""),
    ("user", "Câu hỏi gốc: {question}\n\nTạo 3 biến thể:")
])

# Prompt trích xuất tên bài từ câu hỏi
EXTRACT_TOPIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Trích xuất tên bài viết hoặc chủ đề cần tóm tắt từ câu hỏi. Chỉ trả về tên ngắn gọn, không giải thích."),
    ("user", "{question}")
])


def detect_summary_intent(question: str) -> bool:
    """Phát hiện user muốn tóm tắt 1 bài cụ thể."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in SUMMARY_KEYWORDS)


def get_all_docs_from_neo4j() -> list[Document]:
    """Lấy tất cả documents từ Neo4j cho BM25."""
    docs = []
    with driver.session() as session:
        records = session.run(
            "MATCH (d:Document) RETURN d.content AS content, d.title AS title, d.doc_type AS doc_type"
        )
        for r in records:
            docs.append(Document(
                page_content=r["content"] or "",
                metadata={
                    "title": r["title"] or "",
                    "doc_type": r["doc_type"] or ""
                }
            ))
    return docs


def neo4j_vector_search(query: str, top_k: int = 4) -> list[Document]:
    """Vector search trực tiếp trên Neo4j — thay thế ChromaDB."""
    query_embedding = embedder.encode(f"query: {query}").tolist()
    docs = []
    with driver.session() as session:
        try:
            records = session.run("""
                CALL db.index.vector.queryNodes(
                    'tnu_doc_embeddings', $top_k, $embedding
                ) YIELD node, score
                RETURN node.content AS content,
                       node.title AS title,
                       node.doc_type AS doc_type,
                       score
                ORDER BY score DESC
            """, top_k=top_k, embedding=query_embedding)
            for r in records:
                docs.append(Document(
                    page_content=r["content"] or "",
                    metadata={
                        "title": r["title"] or "",
                        "doc_type": r["doc_type"] or "",
                        "score": r["score"]
                    }
                ))
        except Exception as e:
            print(f"⚠️ Vector search error: {e}")
    print(f"🔵 Neo4j vector: {len(docs)} docs")
    return docs


def neo4j_exact_search(query: str) -> list[Document]:
    """Search chính xác Neo4j — ưu tiên chunks chứa bảng dữ liệu."""
    docs = []
    keywords = [query]
    words = [w for w in query.split() if len(w) > 3]
    keywords += words[:4]

    with driver.session() as session:
        seen = set()
        for kw in keywords:
            records = session.run("""
                MATCH (d:Document)
                WHERE toLower(d.content) CONTAINS toLower($kw)
                RETURN d.content AS content, d.title AS title,
                       d.doc_type AS doc_type
                LIMIT 3
            """, kw=kw)
            for r in records:
                content = r["content"] or ""
                key = content[:80]
                if key not in seen:
                    seen.add(key)
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "title": r["title"] or "",
                            "doc_type": r["doc_type"] or ""
                        }
                    ))
    print(f"📌 Neo4j exact: {len(docs[:5])} docs")
    return docs[:5]


def find_article_by_title(question: str) -> list[Document]:
    """Tìm toàn bộ chunks của 1 bài theo title."""
    try:
        chain = EXTRACT_TOPIC_PROMPT | groq_llm | StrOutputParser()
        topic = chain.invoke({"question": question}).strip()
        print(f"🎯 Tìm bài về: {topic}")
    except Exception:
        topic = question

    docs = []
    with driver.session() as session:
        records = session.run("""
            MATCH (d:Document)
            WHERE toLower(d.title) CONTAINS toLower($topic)
               OR toLower(d.content) CONTAINS toLower($topic)
            RETURN d.content AS content, d.title AS title, d.doc_type AS doc_type
            ORDER BY d.id
        """, topic=topic)
        for r in records:
            docs.append(Document(
                page_content=r["content"] or "",
                metadata={
                    "title": r["title"] or "",
                    "doc_type": r["doc_type"] or ""
                }
            ))

    # Fallback sang Neo4j vector search
    if not docs:
        print(f"⚠️ Không tìm thấy theo title, dùng Neo4j vector search...")
        docs = neo4j_vector_search(topic, top_k=8)

    print(f"✅ Tìm được {len(docs)} chunks cho bài '{topic}'")
    return docs


def expand_query(question: str) -> list[str]:
    """Tạo nhiều biến thể câu hỏi bằng LLM."""
    try:
        chain = EXPANSION_PROMPT | groq_llm | StrOutputParser()
        result = chain.invoke({"question": question})
        variants = [v.strip() for v in result.split("|") if v.strip()]
        queries = [question] + variants[:3]
        print(f"🔍 Query expansion: {queries}")
        return queries
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}")
        return [question]


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Rerank documents bằng FlashRank."""
    if not docs:
        return []
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    ranked_ids = [r["id"] for r in results[:top_k]]
    return [docs[i] for i in ranked_ids]


def _is_relevant(query: str, content: str, threshold: float = 0.1) -> bool:
    """Lọc chunk không liên quan bằng keyword overlap."""
    stopwords = {
        "là", "có", "của", "và", "trong", "về", "với", "được",
        "này", "các", "theo", "bao", "nhiêu", "gì", "nào", "thế",
        "cho", "từ", "đến", "khi", "như", "không", "hay", "hoặc"
    }
    query_words = set(query.lower().split()) - stopwords
    if not query_words:
        return True
    content_lower = content.lower()
    overlap = sum(1 for w in query_words if w in content_lower)
    return (overlap / len(query_words)) >= threshold


def hybrid_search(query: str, top_k: int = 6) -> tuple[str, list[str]]:
    """
    Pipeline (không dùng ChromaDB):
    Neo4j Exact + Neo4j Vector + BM25 → Dedup → Rerank → Filter
    """

    # Detect intent tóm tắt
    if detect_summary_intent(query):
        print(f"📝 Detected summary intent")
        docs = find_article_by_title(query)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("title", "")[:100] for d in docs]))
            return context, sources

    all_docs = get_all_docs_from_neo4j()
    if not all_docs:
        return "", []

    # Bước 1: Neo4j exact search — ưu tiên cao nhất
    priority_docs = neo4j_exact_search(query)

    # Bước 2: Query Expansion
    queries = expand_query(query)

    # Bước 3: Neo4j Vector + BM25 cho mỗi query
    all_retrieved = list(priority_docs)
    for q in queries:
        # Neo4j Vector search (thay ChromaDB)
        vector_docs = neo4j_vector_search(q, top_k=3)
        all_retrieved.extend(vector_docs)

        # BM25
        try:
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=3)
            all_retrieved.extend(bm25_retriever.invoke(q))
        except Exception as e:
            print(f"⚠️ BM25 error: {e}")

    # Bước 4: Dedup
    seen = set()
    unique = []
    for d in all_retrieved:
        key = d.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    # Bước 5: Rerank với query gốc
    if len(unique) > top_k:
        unique = rerank(query, unique, top_k=top_k * 2)

    # Bước 6: Relevance filter
    unique = [d for d in unique if _is_relevant(query, d.page_content)]

    # Bước 7: Đảm bảo priority_docs luôn có trong kết quả cuối
    final = []
    final_keys = set()

    for d in priority_docs[:2]:
        key = d.page_content[:80]
        if key not in final_keys:
            final_keys.add(key)
            final.append(d)

    for d in unique:
        key = d.page_content[:80]
        if key not in final_keys and len(final) < top_k:
            final_keys.add(key)
            final.append(d)

    if not final:
        return "", []

    MAX_CONTEXT_CHARS = 12000
    context = "\n---\n".join([d.page_content for d in final])[:MAX_CONTEXT_CHARS]
    sources = [d.metadata.get("title", "")[:100] for d in final]
    print(f"✅ Final chunks: {len(final)}")
    return context, sources


def run_rag(question: str) -> tuple[str, str]:
    """Groq primary, Gemini fallback. Dùng prompt riêng cho tóm tắt."""
    context, sources = hybrid_search(question)

    if not context.strip():
        return "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA.", "none"

    prompt = SUMMARY_PROMPT if detect_summary_intent(question) else PROMPT

    try:
        chain = prompt | groq_llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return answer, "groq/llama-3.3-70b"
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            try:
                chain = prompt | gemini_llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
                return answer, "gemini-2.5-flash"
            except Exception:
                return "Hệ thống đang quá tải. Vui lòng thử lại sau.", "error"
        raise e