import os
import uuid
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pipeline.loader import load_all, chunk_documents
from langchain_core.documents import Document

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
EMBEDDING_DIM = 384  # multilingual-e5-small dimension


def deduplicate(chunks: list[Document], threshold: float = 0.85) -> list[Document]:
    if not chunks:
        return chunks
    texts = [c.page_content for c in chunks]
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    keep = []
    removed = 0
    for i in range(len(chunks)):
        is_dup = any(
            cosine_similarity(matrix[i], matrix[j])[0][0] >= threshold
            for j in keep
        )
        if not is_dup:
            keep.append(i)
        else:
            removed += 1
    print(f"🔍 Dedup: {len(chunks)} → {len(keep)} chunks (bỏ {removed})")
    return [chunks[i] for i in keep]


def create_vector_index():
    """Tạo Neo4j Vector Index nếu chưa có."""
    with driver.session() as session:
        try:
            session.run("""
                CREATE VECTOR INDEX tnu_doc_embeddings IF NOT EXISTS
                FOR (d:Document)
                ON d.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            print("✅ Neo4j Vector Index created")
        except Exception as e:
            print(f"⚠️ Vector index: {e}")


def build_neo4j(chunks: list[Document]):
    """Build Neo4j với embedding vectors."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    print("🧠 Encoding embeddings...")
    texts = [f"passage: {c.page_content}" for c in chunks]
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True
    ).tolist()

    print("📤 Uploading to Neo4j...")
    with driver.session() as session:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = chunk.metadata
            title = meta.get("title", "")
            doc_type = "blog"
            if any(kw in title for kw in [
                "Thông tư", "Quyết định", "Nghị quyết",
                "Luật ", "Nghị định"
            ]):
                doc_type = "van_ban_phap_quy"

            session.run("""
                CREATE (d:Document {
                    id: $id,
                    title: $title,
                    content: $content,
                    doc_type: $doc_type,
                    source: $source,
                    embedding: $embedding
                })
            """,
                id=str(uuid.uuid4()),
                title=title,
                content=chunk.page_content,
                doc_type=doc_type,
                source=meta.get("source", "blog"),
                embedding=embedding
            )

            if (i + 1) % 10 == 0:
                print(f"  Uploaded {i+1}/{len(chunks)}")

    print(f"✅ Neo4j: {len(chunks)} nodes với embeddings")


def run_pipeline():
    print("📥 Loading all sources...")
    docs = load_all()

    print("✂️  Chunking...")
    chunks = chunk_documents(docs)

    print("🔍 Deduplicating...")
    chunks = deduplicate(chunks)

    print("🔗 Creating Vector Index...")
    create_vector_index()

    print("🔗 Building Neo4j với embeddings...")
    build_neo4j(chunks)

    print("🎉 Pipeline hoàn tất!")


if __name__ == "__main__":
    run_pipeline()