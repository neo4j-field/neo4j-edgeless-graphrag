"""
Confidential GraphRAG Document Ingestion Pipeline
==================================================
Ingests documents from a JSON file into Neo4j with embeddings generated
via Edgeless Privatemode (confidential computing) and security metadata
for Neo4j RBAC/ABAC enforcement.

Usage:
    python ingest_pipeline.py                      # default: documents.json
    python ingest_pipeline.py --file my_docs.json  # custom file

Security model (Neo4j property-level ABAC):
  - finance_reader   -> documents WHERE department = 'finance'
  - hr_reader        -> documents WHERE department = 'hr'
  - executive_reader -> documents WHERE executiveAccess = true
  - analyst          -> documents WHERE classification IN ['public','internal']
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from openai import OpenAI
import httpx
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration -- secrets from .env, everything else configurable with defaults
# ---------------------------------------------------------------------------

# -- Neo4j Connection --
NEO4J_URI = os.getenv("NEO4J_URI", "bolt+ssc://localhost:7688")
NEO4J_USER = os.getenv("NEO4J_ADMIN_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_ADMIN_PASSWORD")

if not NEO4J_PASSWORD:
    raise SystemExit("ERROR: NEO4J_ADMIN_PASSWORD not set. Check your .env file.")

# -- Privatemode Proxy (OpenAI-compatible) --
PRIVATEMODE_BASE_URL = os.getenv("PRIVATEMODE_BASE_URL", "https://localhost:8089/v1")
PRIVATEMODE_API_KEY = os.getenv("PRIVATEMODE_API_KEY")
PRIVATEMODE_CA_CERT = os.getenv("PRIVATEMODE_CA_CERT", "./privatemode/certs/localhost-cert.pem")
PRIVATEMODE_TIMEOUT = int(os.getenv("PRIVATEMODE_TIMEOUT", "120"))

if not PRIVATEMODE_API_KEY:
    raise SystemExit("ERROR: PRIVATEMODE_API_KEY not set. Check your .env file.")

# -- Models --
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

# -- Processing Settings --
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
BATCH_DELAY_SECONDS = float(os.getenv("BATCH_DELAY_SECONDS", "0.25"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client -- pointed at the local Privatemode proxy
# ---------------------------------------------------------------------------
openai_client = OpenAI(
    base_url=PRIVATEMODE_BASE_URL,
    api_key=PRIVATEMODE_API_KEY,
    http_client=None,
)

openai_client._client = httpx.Client(
    base_url=PRIVATEMODE_BASE_URL,
    verify=PRIVATEMODE_CA_CERT,
    timeout=float(PRIVATEMODE_TIMEOUT),
    headers={"Authorization": f"Bearer {PRIVATEMODE_API_KEY}"},
)

# ---------------------------------------------------------------------------
# Embedding helper -- batched
# ---------------------------------------------------------------------------

def generate_confidential_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a batch of texts in a single API call
    through the Privatemode proxy. The proxy encrypts all inputs end-to-end
    before they leave this host; decryption only happens inside an attested
    Confidential VM with AMD SEV-SNP + NVIDIA H100 CC.
    """
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Sort by index to ensure correct ordering
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def setup_schema(driver):
    """Create constraints and vector index once (idempotent)."""
    with driver.session() as s:
        s.run("""
            CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
            FOR (d:Document) REQUIRE d.id IS UNIQUE
        """)
        s.run("""
            CREATE VECTOR INDEX doc_embeddings IF NOT EXISTS
            FOR (d:Document) ON (d.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
        s.run("""
            CREATE FULLTEXT INDEX doc_content_ft IF NOT EXISTS
            FOR (d:Document) ON EACH [d.content, d.title]
        """)
    log.info("Schema constraints and indexes ensured.")


def ingest_document_batch(driver, docs: list[dict], embeddings: list[list[float]]):
    """
    Ingest a batch of documents into Neo4j with pre-generated embeddings
    and all security metadata needed for ABAC grants.
    """
    with driver.session() as s:
        for doc, embedding in zip(docs, embeddings):
            s.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title            = $title,
                    d.content          = $content,
                    d.embedding        = $embedding,
                    d.classification   = $classification,
                    d.department       = $department,
                    d.executiveAccess  = $executive_access,
                    d.accessGroups     = $access_groups,
                    d.docType          = $doc_type,
                    d.ingestedAt       = datetime()
                """,
                doc_id=doc["doc_id"],
                title=doc["title"],
                content=doc["content"],
                embedding=embedding,
                classification=doc["classification"],
                department=doc["department"],
                executive_access=doc["executive_access"],
                access_groups=doc["access_groups"],
                doc_type=doc.get("doc_type", "report"),
            )
            log.info(
                "Ingested %s  [%s / %s / exec=%s]",
                doc["doc_id"], doc["department"],
                doc["classification"], doc["executive_access"],
            )


# ---------------------------------------------------------------------------
# Document loader
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "doc_id", "title", "content", "classification",
    "department", "executive_access", "access_groups",
}

def load_documents(file_path: str) -> list[dict]:
    """Load and validate documents from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        log.error("Document file not found: %s", path)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        log.error("JSON root must be an array of document objects.")
        sys.exit(1)

    for i, doc in enumerate(docs):
        missing = REQUIRED_FIELDS - set(doc.keys())
        if missing:
            log.error(
                "Document at index %d (%s) missing fields: %s",
                i, doc.get("doc_id", "UNKNOWN"), missing,
            )
            sys.exit(1)

    log.info("Loaded %d documents from %s", len(docs), path)
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Neo4j with Privatemode embeddings."
    )
    parser.add_argument(
        "--file", "-f",
        default="documents.json",
        help="Path to the JSON document file (default: documents.json)",
    )
    args = parser.parse_args()

    # Load documents
    documents = load_documents(args.file)

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    log.info("Connected to Neo4j at %s", NEO4J_URI)

    # Create schema (idempotent)
    # setup_schema(driver)

    # Ingest documents in batches
    total = len(documents)
    success = 0

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = documents[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        batch_num = (batch_start // EMBEDDING_BATCH_SIZE) + 1
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total)

        # Step 1: Batch-generate embeddings for all docs in this batch
        texts = [doc["content"] for doc in batch]
        try:
            embeddings = generate_confidential_embeddings_batch(texts)
            log.info(
                "Batch %d: embedded %d documents (%d-%d)",
                batch_num, len(batch), batch_start + 1, batch_end,
            )
        except Exception as e:
            log.error("Batch %d embedding failed: %s", batch_num, e)
            continue

        # Step 2: Ingest into Neo4j
        try:
            ingest_document_batch(driver, batch, embeddings)
            success += len(batch)
        except Exception as e:
            log.error("Batch %d Neo4j ingest failed: %s", batch_num, e)

        log.info("Progress: %d / %d  (success: %d)", batch_end, total, success)

        # Rate limit between batches
        if batch_end < total:
            time.sleep(BATCH_DELAY_SECONDS)

    # Summary stats
    with driver.session() as s:
        result = s.run("""
            MATCH (d:Document)
            RETURN d.department AS department,
                   d.classification AS classification,
                   d.executiveAccess AS execAccess,
                   count(*) AS docCount
            ORDER BY department, classification
        """)
        log.info("=== Ingestion Summary ===")
        for rec in result:
            log.info(
                "  dept=%-12s  class=%-14s  exec=%-5s  count=%d",
                rec["department"], rec["classification"],
                rec["execAccess"], rec["docCount"],
            )

    driver.close()
    log.info("Done. %d / %d documents ingested successfully.", success, total)


if __name__ == "__main__":
    main()