from app.services.pdf_parser import PDFParser
from app.core.chunker import TextChunker
from app.core.embedder import EmbeddingGenerator
from app.core.retriever import RAGRetriever
from app.core.llm import LLMGenerator

# --- Step 1: Parse + Chunk + Embed ---
parser = PDFParser("data/uploads/sample.pdf")
pages = parser.extract_text_by_page()
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_pages(pages, source="sample.pdf")
embedder = EmbeddingGenerator()
embeddings = embedder.embed_chunks(chunks)
print(f"✅ Chunks embedded: {embeddings.shape}")

# --- Step 2: Retrieve context ---
retriever = RAGRetriever()
retriever.add_documents(embeddings, chunks)

query = "What is the main topic of this document?"
retrieved = retriever.retrieve(query, top_k=3)
context = retriever.build_context(retrieved, max_context_length=1500)

print(f"✅ Context built: {len(context)} chars")
print(f"\n📝 Context Preview:\n{context[:400]}...")

# --- Step 3: Load LLM and generate answer ---
print("\n🧠 Loading LLM (may take a minute on first run)...")
llm = LLMGenerator()

answer = llm.answer(
    question=query,
    context=context,
    max_new_tokens=200
)

print(f"\n{'='*60}")
print(f"❓ Question: {query}")
print(f"{'='*60}")
print(f"💡 Answer:   {answer}")
print(f"{'='*60}")

# --- Step 4: Test with different questions ---
questions = [
    "Summarize the key points from the document.",
    "What conclusions are drawn in the document?",
]

print("\n🔁 Testing multiple questions:\n")
for q in questions:
    retrieved = retriever.retrieve(q, top_k=3)
    context = retriever.build_context(retrieved)
    ans = llm.answer(question=q, context=context)
    print(f"❓ {q}")
    print(f"💡 {ans}\n")