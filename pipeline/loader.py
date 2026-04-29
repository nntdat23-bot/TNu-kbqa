import os
import time
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

BLOG_URL = "https://tnu-aiqa.blogspot.com"


def parse_html_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with("\n" + "\n".join(rows) + "\n")
    for li in soup.find_all("li"):
        li.insert_before("• ")
    return soup.get_text(separator="\n").strip()


def load_blog(max_posts: int = 200) -> list[Document]:
    """Crawl blog → LangChain Documents."""
    docs = []
    feed_url = f"{BLOG_URL}/feeds/posts/default?alt=json&max-results={max_posts}"
    seen_titles = set()

    try:
        r = requests.get(feed_url, timeout=15)
        entries = r.json().get("feed", {}).get("entry", [])

        for entry in entries:
            title = entry.get("title", {}).get("$t", "").strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)

            content_html = entry.get("content", {}).get("$t", "")
            content = parse_html_content(content_html)

            if len(content) < 100:
                print(f"⚠️  Skip (quá ngắn): {title[:50]}")
                continue

            # Xác định doc_type
            doc_type = "van_ban_phap_quy" if any(
                kw in title for kw in ["Thông tư", "Quyết định", "Nghị quyết", "Luật ", "Nghị định"]
            ) else "blog"

            docs.append(Document(
                page_content=content,
                metadata={
                    "title": title,
                    "source": "blog",
                    "doc_type": doc_type,
                    "url": BLOG_URL
                }
            ))
            print(f"✅ Blog: {title[:60]} ({len(content)} chars)")
            time.sleep(0.3)

        print(f"✅ Tổng blog: {len(docs)} posts")
    except Exception as e:
        print(f"❌ Blog error: {e}")

    return docs


def load_pdfs(folder: str = "data/pdfs") -> list[Document]:
    """Load PDFs → LangChain Documents."""
    docs = []
    if not os.path.exists(folder):
        return docs

    for filename in os.listdir(folder):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(folder, filename)
        try:
            loader = PyMuPDFLoader(path)
            pages = loader.load()
            text = "\n".join([p.page_content for p in pages])

            if len(text.strip()) < 50:
                print(f"⚠️  Skip PDF: {filename}")
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "title": filename.replace(".pdf", ""),
                    "source": "pdf",
                    "doc_type": "van_ban_phap_quy"
                }
            ))
            print(f"✅ PDF: {filename} ({len(text)} chars)")
        except Exception as e:
            print(f"❌ PDF error {filename}: {e}")

    return docs


def load_latex(folder: str = "data/latex") -> list[Document]:
    """Load LaTeX files → LangChain Documents."""
    import re
    docs = []
    if not os.path.exists(folder):
        return docs

    for filename in os.listdir(folder):
        if not filename.endswith(".tex"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Clean LaTeX
        text = re.sub(r"%.*?\n", "\n", raw)
        text = re.sub(r"\\(section|subsection|chapter)\*?\{(.+?)\}", r"\n\2\n", text)
        text = re.sub(r"\\textbf\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\textit\{(.+?)\}", r"\1", text)
        text = re.sub(r"\\item\s*", "• ", text)
        text = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+\*?\{.*?\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            print(f"⚠️  Skip LaTeX: {filename}")
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "title": filename.replace(".tex", ""),
                "source": "latex",
                "doc_type": "van_ban_phap_quy"
            }
        ))
        print(f"✅ LaTeX: {filename} ({len(text)} chars)")

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,   # tăng từ 1500 lên 3000
        chunk_overlap=500, # tăng overlap để không bị cắt đứt bảng
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    filtered = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) < 100:
            continue
        if "http" in text and len(text) < 200:
            continue
        filtered.append(c)

    print(f"✅ Total chunks sau lọc: {len(filtered)}")
    return filtered


def load_all() -> list[Document]:
    """Load tất cả nguồn data."""
    print("📥 Loading blog...")
    blog_docs = load_blog()

    print("📥 Loading PDFs...")
    pdf_docs = load_pdfs()

    print("📥 Loading LaTeX...")
    latex_docs = load_latex()

    all_docs = blog_docs + pdf_docs + latex_docs
    print(f"📄 Total docs: {len(all_docs)}")
    return all_docs