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

# Lazy load ranker
_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        print("Loading ranker...")
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker

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
- Nếu tài liệu có bảng → trình bày dạng danh sách có số thứ tự
- Nếu không có trong tài liệu → "Tôi không tìm thấy thông tin này trong tài liệu TNU-AIQA."
- Trả lời tiếng Việt, tự nhiên, dễ đọc
- Dùng **bold** cho tiêu đề quan trọng
- Dùng số thứ tự hoặc gạch đầu dòng cho danh sách
- KHÔNG trả ra raw text dạng pipe | một cách thô"""),
    ("user", """TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời rõ ràng, tự nhiên, có format đẹp:""")
])

# Prompt tóm tắt bài
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI của TNU-AIQA.
Tóm tắt nội dung bài viết theo cấu trúc:
1. **Chủ đề chính**: 1-2 câu
2. **Những điểm nổi bật**: 3-5 điểm quan trọng
3. **Kết luận**: 1-2 câu
Chỉ dùng thông tin từ tài liệu."""),
    ("user", "NỘI DUNG:\n{context}\n\nTóm tắt:")
])

# Prompt query expansion
EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Chuyên gia kiểm định chất lượng giáo dục Việt Nam.
Tạo 3 biến thể câu hỏi, phân cách bằng |, không giải thích.
Dùng thuật ngữ: AUN-QA, CTĐT, IQA, TT04, TT20..."""),
    ("user", "Câu hỏi: {question}\n\n3 biến thể:")
])

# Prompt trích xuất topic
EXTRACT_TOPIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Trích xuất chủ đề cần tóm tắt. Chỉ trả về tên ngắn gọn."),
    ("user", "{question}")
])


def detect_summary_intent(question: str) -> bool:
    return any(kw in question.lower() for kw in SUMMARY_KEYWORDS)


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


def neo4j_exact_search(query: str) -> list[Document]:
    """Exact keyword search trên Neo4j."""
    docs = []
    keywords = [query] + [w for w in query.split() if len(w) > 3][:4]
    with driver.session() as session:
        seen = set()
        for kw in keywords:
            records = session.run("""
                MATCH (d:Document)
                WHERE toLower(d.content) CONTAINS toLower($kw)
                RETURN d.content AS content, d.title AS title, d.doc_type AS doc_type
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
    """Tìm toàn bộ chunks của 1 bài."""
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
                metadata={"title": r["title"] or "", "doc_type": r["doc_type"] or ""}
            ))

    if not docs:
        # Fallback: BM25 search
        all_docs = get_all_docs_from_neo4j()
        if all_docs:
            try:
                bm25 = BM25Retriever.from_documents(all_docs, k=8)
                docs = bm25.invoke(topic)
            except Exception as e:
                print(f"⚠️ BM25 fallback error: {e}")

    print(f"✅ Tìm được {len(docs)} chunks")
    return docs


def expand_query(question: str) -> list[str]:
    try:
        chain = EXPANSION_PROMPT | groq_llm | StrOutputParser()
        result = chain.invoke({"question": question})
        variants = [v.strip() for v in result.split("|") if v.strip()]
        queries = [question] + variants[:3]
        print(f"🔍 Query expansion: {queries}")
        return queries
    except Exception as e:
        print(f"⚠️ Expansion failed: {e}")
        return [question]


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    if not docs:
        return []
    try:
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = get_ranker().rerank(request)
        ranked_ids = [r["id"] for r in results[:top_k]]
        return [docs[i] for i in ranked_ids]
    except Exception as e:
        print(f"⚠️ Rerank error: {e}")
        return docs[:top_k]


def _is_relevant(query: str, content: str, threshold: float = 0.1) -> bool:
    stopwords = {
        "là", "có", "của", "và", "trong", "về", "với", "được",
        "này", "các", "theo", "bao", "nhiêu", "gì", "nào", "thế",
        "cho", "từ", "đến", "khi", "như", "không", "hay", "hoặc"
    }
    query_words = set(query.lower().split()) - stopwords
    if not query_words:
        return True
    overlap = sum(1 for w in query_words if w in content.lower())
    return (overlap / len(query_words)) >= threshold


def hybrid_search(query: str, top_k: int = 6) -> tuple[str, list[str]]:
    """
    Pipeline tiết kiệm RAM:
    Neo4j Exact + BM25 + Query Expansion → Dedup → Rerank → Filter
    KHÔNG dùng Vector Search (tiết kiệm ~400MB RAM)
    """

    # Detect summary intent
    if detect_summary_intent(query):
        print(f"📝 Summary intent")
        docs = find_article_by_title(query)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([d.metadata.get("title", "")[:100] for d in docs]))
            return context, sources

    all_docs = get_all_docs_from_neo4j()
    if not all_docs:
        return "", []

    # Bước 1: Neo4j exact — priority
    priority_docs = neo4j_exact_search(query)

    # Bước 2: Query expansion
    queries = expand_query(query)

    # Bước 3: BM25 cho mỗi query
    all_retrieved = list(priority_docs)
    for q in queries:
        try:
            bm25 = BM25Retriever.from_documents(all_docs, k=3)
            all_retrieved.extend(bm25.invoke(q))
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

    # Bước 5: Rerank
    if len(unique) > top_k:
        unique = rerank(query, unique, top_k=top_k * 2)

    # Bước 6: Relevance filter
    unique = [d for d in unique if _is_relevant(query, d.page_content)]

    # Bước 7: Priority docs luôn có mặt
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

    MAX_CONTEXT = 12000
    context = "\n---\n".join([d.page_content for d in final])[:MAX_CONTEXT]
    sources = [d.metadata.get("title", "")[:100] for d in final]
    print(f"✅ Final: {len(final)} chunks")
    return context, sources


def run_rag(question: str) -> tuple[str, str]:
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
                return "Hệ thống đang quá tải Vui lòng thử lại sau."
        raise e
