# ADR-003: Vector Storage Strategy — LanceDB

| Field | Value |
|-------|-------|
| **Title** | LanceDB as the default vector store with schema, embedding, and index strategy |
| **Status** | accepted |
| **Deciders** | Architecture team, data engineering lead |
| **Date** | 2025-04-05 |
| **Supersedes** | None |
| **Tags** | vectordb, lancedb, storage, embedding, index |

## Context

The Smithy framework requires a vector database to store document chunk embeddings and perform similarity search during retrieval. The current implementation uses LanceDB, an embedded vector database built on the Apache Arrow-based Lance format.

Key requirements and constraints:
- Zero-infrastructure deployment (LanceDB is serverless — no separate database process)
- Support for adding documents in batches with embedding generation
- Similarity search returning top-K results
- Persistence to disk for knowledge base durability
- Schema flexibility for metadata storage
- Compatibility with Python-first stack

Current implementation issues identified during code review:
1. **Table creation uses raw dict schema** — `vectordb.py:38` passes a Python dict to `db.create_table()`, which may not work with all LanceDB versions. LanceDB expects PyArrow schemas or Pydantic models.
2. **No index optimization** — LanceDB supports IVF-PQ indexes for faster search, but the current implementation uses flat (brute-force) search.
3. **No connection pooling** — `Config._init_clients()` creates a new `lancedb.connect()` call per `Config` instance.
4. **Embedding generation is synchronous and blocking** — `_get_embeddings()` in `vectordb.py` processes texts in batches of 100 with a `time.sleep(0.5)` between batches.
5. **Mock table fallback silently swallows errors** — `get_table()` returns a `MockTable` on failure, which silently discards all data operations.

## Decision

We will retain LanceDB as the default vector store but improve the implementation:

### 1. Schema Definition via Pydantic Model

Replace the raw dict schema with a LanceDB Pydantic model:

```python
from lancedb.pydantic import LanceModel, Vector

class ChunkModel(LanceModel):
    id: str
    doc_id: str
    source: str
    content: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None
    embedding: Vector(config.embedding_dimensions)
```

### 2. Index Optimization Strategy

- **Below 10,000 chunks**: Use flat (brute-force) search — no index needed
- **Above 10,000 chunks**: Create an IVF-PQ index after bulk document addition
  ```python
  table.create_index(
      vector_column_name="embedding",
      metric="cosine",
      num_partitions=256,
      num_sub_vectors=96,
  )
  ```

### 3. Connection Management

- Use a singleton pattern or lazy initialization for the LanceDB connection
- Reuse the connection across all `VectorDBManager` operations within a process

### 4. Error Handling

- Remove `MockTable` fallback — fail explicitly with clear error messages
- Add retry logic for embedding API failures
- Log document count and table statistics on each add/search operation

### 5. Migration Strategy

- On first use with the new schema, detect existing tables with old schemas
- Migrate by reading old data, re-creating the table with the new schema, and re-inserting

### Why Not Alternatives?

| Alternative | Why Rejected |
|-------------|-------------|
| **FAISS** | Requires manual serialization, no metadata storage, no built-in persistence |
| **Chroma** | Server or embedded mode adds complexity, less mature than LanceDB for production |
| **Pinecone/Weaviate** | Cloud-hosted, adds network dependency, not suitable for local/offline deployment |
| **Milvus** | Requires separate server process (Zookeeper + etcd + MinIO), overkill for this scale |
| **SQLite + sentence-transformers** | No native vector search until SQLite 3.40+, limited ANN support |

LanceDB offers the best trade-off: zero infrastructure, native Python, Arrow-based columnar storage, and growing ecosystem support.

## Consequences

### Positive
- Zero operational overhead — no database to manage
- Schema evolution through Pydantic models is type-safe
- IVF-PQ indexing will enable sub-second search at scale
- Arrow-native format enables interoperability with pandas, DuckDB, etc.

### Negative
- LanceDB is younger than alternatives — API stability risk
- IVF-PQ index tuning requires empirical testing per dataset
- No multi-process write support — concurrent writes to the same DB file are not safe

### Risks
- **Data corruption**: Concurrent writes from multiple processes could corrupt the Lance files
- **Index staleness**: If documents are added incrementally, the index may become suboptimal and need rebuilding
- **Memory pressure**: Large embedding batches could exhaust memory on resource-constrained systems

## Alternatives Considered

### Chroma
- **Pros:** Simple API, good Python support, active community
- **Cons:** Adds a server component for production, embedding function coupling is opinionated
- **Why rejected:** The zero-infrastructure requirement rules out server-based options. Embedded Chroma is comparable but LanceDB's Arrow foundation provides better interoperability.

### FAISS
- **Pros:** Fastest ANN search, battle-tested at Meta scale
- **Cons:** No persistence layer, no metadata storage, manual index management
- **Why rejected:** Would require building a persistence and metadata layer from scratch.

### Qdrant (embedded mode)
- **Pros:** Rust-based, fast, good API
- **Cons:** Embedded mode is relatively new, less Python-native feel
- **Why rejected:** LanceDB's integration with the Python data ecosystem (pandas, PyArrow) is superior for this use case.

## Notes

- The `with_embeddings` helper from `lancedb.embeddings` is used conditionally — fallback to manual embedding generation if the import fails.
- Future work: Add support for hybrid search (BM25 + vector) using LanceDB's full-text search capabilities.
- Future work: Explore LanceDB's built-in embedding function support to decouple embedding generation from the application layer.

## References

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Lance Format Specification](https://lancedb.github.io/lance/)
- [IVF-PQ Index Guide](https://lancedb.github.io/lancedb/concepts/vector_search/)
