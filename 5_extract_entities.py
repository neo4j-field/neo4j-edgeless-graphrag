"""
Confidential Entity Extraction Pipeline
========================================
Reads Document nodes from Neo4j, extracts entities (topics, concepts,
names, organizations, technologies) via Privatemode's confidential LLM,
generates embeddings for each entity, and connects them to their source
documents with MENTIONS relationships.

Usage:
    python extract_entities.py

All LLM inference and embedding generation routes through the local
Privatemode proxy (OpenAI-compatible API) -- document content never
leaves the confidential compute boundary in plaintext.
"""

import json
import os
import sys
import time
import logging
import httpcore
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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss-120b")

# -- Processing Settings --
BATCH_DELAY_SECONDS = float(os.getenv("BATCH_DELAY_SECONDS", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client -- pointed at the local Privatemode proxy
# ---------------------------------------------------------------------------
# The proxy handles remote attestation, E2E encryption (AES-GCM),
# and secure key exchange with attested Confidential VMs.
# We pass the proxy's self-signed CA cert for TLS verification.

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
# Entity extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction assistant. Given a document's title and content,
extract the key entities as SINGLE NOUNS or proper nouns only.

Entity types to extract:
- Concepts (e.g. "SaaS", "encryption", "attestation", "compliance")
- Technologies (e.g. "Neo4j", "Cypher", "vLLM", "Kubernetes")
- Organizations (e.g. "Okta", "Scaleway", "NIST")
- People (first name and last name as SEPARATE entries, e.g. "Stig", "Engstrom")
- Standards/acronyms (e.g. "GDPR", "FIPS", "RBAC", "ABAC", "COBRA")
- Business terms (e.g. "revenue", "budget", "severance", "onboarding")

CRITICAL RULES:
- Every entity MUST be a SINGLE word. Never combine words.
- Break multi-word phrases into individual nouns:
  "Enterprise SaaS" -> "Enterprise", "SaaS"
  "confidential computing" -> "confidential", "computing"
  "remote attestation" -> "attestation"
  "role-based access control" -> "RBAC"
  "vector embeddings" -> "embeddings"
  "cash flow" -> "cashflow"
  "graph traversal" -> "traversal"
- Proper nouns that are a single compound name stay as one: "Neo4j", "GraphRAG", "OpenAI"
- Acronyms stay as one: "GDPR", "ABAC", "TLS", "HDHP", "OIDC"
- Do NOT extract dollar amounts, percentages, dates, or purely numeric values.
- Do NOT extract generic filler words like "report", "document", "company", "team",
  "section", "overview", "details", "summary", "note", "plan".
- Do NOT extract adjectives unless they function as standalone nouns (e.g. "hybrid").
- Deduplicate: include each word only once.
- Return ONLY a JSON array of unique strings, nothing else.
- Aim for 5-15 entities per document depending on content richness.

Example input: "Quarterly financial projections show 15% growth driven by enterprise
SaaS renewals and expansion into APAC markets."
Example output: ["projections", "growth", "enterprise", "SaaS", "renewals", "expansion", "APAC"]
"""

# ---------------------------------------------------------------------------
# Privatemode API helpers (via OpenAI SDK)
# ---------------------------------------------------------------------------

def extract_entities_from_text(title: str, content: str) -> list[str]:
    """
    Extract entities from document text using the confidential LLM
    routed through the Privatemode proxy.
    """
    user_prompt = f"Title: {title}\n\nContent: {content}"

    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    raw_content = completion.choices[0].message.content
    if raw_content is None:
        log.warning("LLM returned None content for: %s", title[:80])
        return []
    raw = raw_content.strip()

    # Handle cases where the LLM wraps the JSON in markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        entities = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Failed to parse LLM response as JSON: %s", raw[:200])
        return []

    if not isinstance(entities, list):
        log.warning("LLM returned non-list: %s", type(entities))
        return []

    # Normalize: strip whitespace, deduplicate by lowercase key,
    # and split any multi-word entities the LLM may still return
    seen = set()
    normalized = []

    # Known compound proper nouns / acronyms that must NOT be split
    keep_as_is = {
        "neo4j", "graphrag", "openai", "privatemode", "hbase",
        "snowflake", "mongodb", "postgresql", "chatgpt", "langchain",
        "llamaindex", "huggingface", "bitlocker", "javascript",
        "typescript", "kubernetes", "tensorflow", "pytorch",
    }

    for e in entities:
        if not isinstance(e, str):
            continue
        clean = e.strip()
        if not clean:
            continue

        # If it's a known compound word or pure acronym, keep it whole
        if clean.lower() in keep_as_is or clean.isupper():
            key = clean.lower()
            if key not in seen:
                seen.add(key)
                normalized.append(clean)
            continue

        # Split multi-word entries into individual tokens
        tokens = clean.replace("-", " ").split()
        for token in tokens:
            token = token.strip(".,;:!?()[]{}\"'")
            if not token or len(token) < 2:
                continue
            key = token.lower()
            if key not in seen:
                seen.add(key)
                normalized.append(token)

    return normalized


def generate_confidential_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a batch of texts in a single API call
    through the Privatemode proxy.
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

def setup_entity_schema(driver):
    """Create Entity constraints and vector index (idempotent)."""
    with driver.session() as s:
        s.run("""
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        s.run("""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 2560,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
    log.info("Entity schema constraints and indexes ensured.")


def fetch_documents(driver) -> list[dict]:
    """Read all Document nodes with their id, title, and content."""
    with driver.session() as s:
        result = s.run("""
            MATCH (d:Document)
            WHERE d.content IS NOT NULL
            RETURN d.id AS doc_id, d.title AS title, d.content AS content
            ORDER BY d.id
        """)
        return [dict(rec) for rec in result]


def upsert_entity_and_link(driver, doc_id: str, entity_name: str, embedding: list[float]):
    """
    Create or merge an Entity node with its embedding, and create a
    MENTIONS relationship from the Document to the Entity.
    """
    with driver.session() as s:
        s.run(
            """
            MATCH (d:Document {id: $doc_id})
            MERGE (e:Entity {name: $entity_name})
            ON CREATE SET e.embedding  = $embedding,
                          e.createdAt  = datetime()
            MERGE (d)-[:MENTIONS]->(e)
            """,
            doc_id=doc_id,
            entity_name=entity_name,
            embedding=embedding,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    log.info("Connected to Neo4j at %s", NEO4J_URI)

    # Ensure Entity schema
    setup_entity_schema(driver)

    # Fetch all documents
    documents = fetch_documents(driver)
    log.info("Found %d documents to process.", len(documents))

    if not documents:
        log.info("No documents found. Exiting.")
        driver.close()
        return

    # Track unique entities across all documents to avoid redundant
    # embedding calls -- only generate an embedding on first encounter
    entity_embeddings_cache: dict[str, list[float]] = {}

    total = len(documents)
    total_entities = 0
    total_links = 0

    for i, doc in enumerate(documents, 1):
        doc_id = doc["doc_id"]
        title = doc["title"]
        content = doc["content"]

        # Step 1: Extract entities via confidential LLM
        try:
            entities = extract_entities_from_text(title, content)
        except Exception as e:
            log.error("Entity extraction failed for %s: %s", doc_id, e)
            continue

        if not entities:
            log.warning("No entities extracted for %s", doc_id)
            continue

        log.info(
            "Extracted %d entities from %s: %s",
            len(entities), doc_id, entities,
        )

        # Step 2: Collect entities that need new embeddings, batch-embed them
        new_entities = []
        for entity_name in entities:
            if entity_name.lower() not in entity_embeddings_cache:
                new_entities.append(entity_name)

        if new_entities:
            try:
                embeddings = generate_confidential_embeddings_batch(new_entities)
                for entity_name, embedding in zip(new_entities, embeddings):
                    entity_embeddings_cache[entity_name.lower()] = embedding
                    total_entities += 1
                log.info(
                    "  Batch-embedded %d new entities for %s",
                    len(new_entities), doc_id,
                )
            except Exception as e:
                log.error(
                    "Batch embedding failed for %s: %s", doc_id, e,
                )
                continue

        # Step 3: Upsert Entity nodes and MENTIONS relationships
        for entity_name in entities:
            entity_key = entity_name.lower()
            embedding = entity_embeddings_cache.get(entity_key)
            if embedding is None:
                continue  # skip if embedding failed earlier

            try:
                upsert_entity_and_link(driver, doc_id, entity_name, embedding)
                total_links += 1
            except Exception as e:
                log.error(
                    "Neo4j upsert failed for %s -> '%s': %s",
                    doc_id, entity_name, e,
                )

        if i % 10 == 0 or i == total:
            log.info("Progress: %d / %d documents processed.", i, total)

        # Rate limit between documents (LLM + embedding calls)
        if i < total:
            time.sleep(BATCH_DELAY_SECONDS)

    # Summary
    with driver.session() as s:
        result = s.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
            WITH e.name AS entity, count(d) AS docCount
            ORDER BY docCount DESC
            LIMIT 15
            RETURN entity, docCount
        """)
        log.info("=== Top Entities by Document Count ===")
        for rec in result:
            log.info("  %-35s  mentioned in %d docs", rec["entity"], rec["docCount"])

    log.info(
        "Done. Processed %d documents, created %d unique entities, "
        "established %d MENTIONS links.",
        total, total_entities, total_links,
    )

    driver.close()


if __name__ == "__main__":
    main()