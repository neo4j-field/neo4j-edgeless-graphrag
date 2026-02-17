"""
Secure GraphRAG Query Pipeline
===============================
Performs retrieval-augmented generation over the Neo4j knowledge graph
with RBAC enforcement. Each query runs as a specific Neo4j user whose
role-based grants determine which Documents and Entities are visible.

The pipeline:
  1. Embeds the user question via Privatemode (confidential)
  2. Performs vector similarity search on Documents (RBAC-filtered)
  3. Traverses the graph to find related Entities and connected Documents
  4. Assembles a grounded context from graph results
  5. Sends context + question to the confidential LLM for answer generation

Usage:
    python graphrag_query.py --user finance_user --question "What is our Q4 revenue forecast?"
    python graphrag_query.py --user hr_user --question "What are the engineering compensation bands?"
    python graphrag_query.py --user analyst_user --question "What is our remote work policy?"
    python graphrag_query.py --interactive --user executive_user

Neo4j RBAC roles (configured on the server):
  - finance_reader   -> sees documents WHERE department = 'finance'
  - hr_reader        -> sees documents WHERE department = 'hr'
  - executive_reader -> sees documents WHERE executiveAccess = true
  - analyst          -> sees documents WHERE classification IN ['public','internal']
"""

import argparse
import json
import os
import sys
import logging
from openai import OpenAI
import httpx
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — secrets from .env, everything else configurable with defaults
# ---------------------------------------------------------------------------

# -- Neo4j Connection --
NEO4J_URI = os.getenv("NEO4J_URI", "bolt+ssc://localhost:7688")

# -- Neo4j Users (RBAC roles) --
# Passwords loaded from .env, with fallback defaults for development
NEO4J_ADMIN_PASSWORD = os.getenv("NEO4J_ADMIN_PASSWORD")
NEO4J_ALICE_PASSWORD = os.getenv("NEO4J_ALICE_PASSWORD", "secure-p@ssword-One")
NEO4J_BOB_PASSWORD = os.getenv("NEO4J_BOB_PASSWORD", "secure-p@ssword-Two")

if not NEO4J_ADMIN_PASSWORD:
    raise SystemExit("ERROR: NEO4J_ADMIN_PASSWORD not set. Check your .env file.")

NEO4J_USERS = {
    "alice": {"user": "alice", "password": NEO4J_ALICE_PASSWORD, "roles": ["finance_reader", "analyst"]},
    "bob":   {"user": "bob",   "password": NEO4J_BOB_PASSWORD,   "roles": ["hr_reader", "analyst"]},
    "admin": {"user": "neo4j", "password": NEO4J_ADMIN_PASSWORD, "roles": ["admin"]},
}

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

# -- Query Settings --
TOP_K_DOCUMENTS = int(os.getenv("VECTOR_TOP_K", "5"))
MAX_GRAPH_HOPS = int(os.getenv("MAX_GRAPH_HOPS", "2"))
TOP_K_ENTITIES = int(os.getenv("GRAPH_TOP_K_ENTITIES", "10"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client — pointed at the local Privatemode proxy
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
# Embedding helper
# ---------------------------------------------------------------------------

def embed_query(text: str) -> list[float]:
    """Embed the user query via Privatemode (confidential)."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Neo4j retrieval — all queries run as the RBAC-bound user
# ---------------------------------------------------------------------------

def vector_search_documents(driver, query_embedding: list[float], top_k: int) -> list[dict]:
    """
    Vector similarity search on Document nodes.
    Results are automatically filtered by the connected user's RBAC grants —
    the user only sees documents their role permits.
    """
    with driver.session() as s:
        result = s.run(
            """
            CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
            YIELD node AS doc, score
            RETURN doc.id          AS doc_id,
                   doc.title       AS title,
                   doc.content     AS content,
                   doc.department  AS department,
                   doc.classification AS classification,
                   doc.executiveAccess AS executiveAccess,
                   score
            ORDER BY score DESC
            """,
            top_k=top_k,
            embedding=query_embedding,
        )
        return [dict(rec) for rec in result]


def graph_expand(driver, doc_ids: list[str], max_hops: int, top_k_entities: int) -> dict:
    """
    Traverse from matched documents through MENTIONS relationships to
    find related Entities, then find other Documents that share those entities.
    This is the 'graph' part of GraphRAG — enriching context beyond
    what vector search alone returns.

    All results are RBAC-filtered: the user only sees entities linked
    to documents they have access to.
    """
    with driver.session() as s:
        # Find entities mentioned by the matched documents
        entity_result = s.run(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE d.id IN $doc_ids
            WITH e, count(d) AS relevance
            ORDER BY relevance DESC
            LIMIT $top_k
            RETURN e.name AS entity, relevance
            """,
            doc_ids=doc_ids,
            top_k=top_k_entities,
        )
        entities = [dict(rec) for rec in entity_result]

        # Find additional documents connected via shared entities
        # (RBAC automatically filters — user only sees permitted docs)
        neighbor_result = s.run(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbor:Document)
            WHERE d.id IN $doc_ids
              AND NOT neighbor.id IN $doc_ids
            WITH DISTINCT neighbor, collect(DISTINCT e.name) AS shared_entities
            RETURN neighbor.id             AS doc_id,
                   neighbor.title          AS title,
                   neighbor.content        AS content,
                   neighbor.department     AS department,
                   neighbor.classification AS classification,
                   shared_entities
            ORDER BY size(shared_entities) DESC
            LIMIT $limit
            """,
            doc_ids=doc_ids,
            limit=top_k_entities,
        )
        neighbors = [dict(rec) for rec in neighbor_result]

        # Find shortest paths between top matched documents (if multiple)
        paths = []
        if len(doc_ids) >= 2:
            path_result = s.run(
                """
                MATCH (d1:Document {id: $id1}), (d2:Document {id: $id2})
                MATCH path = shortestPath((d1)-[*..6]-(d2))
                UNWIND nodes(path) AS n
                WITH n WHERE n:Entity
                RETURN DISTINCT n.name AS entity
                """,
                id1=doc_ids[0],
                id2=doc_ids[1],
            )
            paths = [rec["entity"] for rec in path_result if rec["entity"]]

    return {
        "entities": entities,
        "neighbor_documents": neighbors,
        "path_entities": paths,
    }


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    question: str,
    vector_docs: list[dict],
    graph_context: dict,
    user_role: str,
) -> str:
    """
    Build a structured context string from vector search results and
    graph traversal for the LLM to ground its answer on.
    """
    parts = []

    parts.append(f"## Query Context (role: {user_role})\n")

    # Vector search results
    if vector_docs:
        parts.append("### Most Relevant Documents (Vector Search)")
        for i, doc in enumerate(vector_docs, 1):
            parts.append(
                f"\n**[{i}] {doc['title']}** "
                f"(dept: {doc['department']}, class: {doc['classification']}, "
                f"score: {doc['score']:.4f})\n"
                f"{doc['content']}"
            )

    # Related entities
    entities = graph_context.get("entities", [])
    if entities:
        entity_str = ", ".join(
            f"{e['entity']} ({e['relevance']})" for e in entities
        )
        parts.append(f"\n### Key Entities Mentioned\n{entity_str}")

    # Path entities (connecting concepts between top results)
    path_entities = graph_context.get("path_entities", [])
    if path_entities:
        parts.append(
            f"\n### Connecting Concepts (Graph Path)\n{', '.join(path_entities)}"
        )

    # Neighbor documents (discovered via graph traversal)
    neighbors = graph_context.get("neighbor_documents", [])
    if neighbors:
        parts.append("\n### Related Documents (Graph Traversal)")
        for n in neighbors[:3]:  # limit to top 3 to keep context manageable
            parts.append(
                f"\n**{n['title']}** "
                f"(shared entities: {', '.join(n['shared_entities'])})\n"
                f"{n['content']}"
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """\
You are a knowledgeable enterprise assistant. Answer the user's question
based ONLY on the provided context. The context comes from documents the
user has access to based on their role.

Rules:
- Ground every claim in the provided documents. Cite document titles when possible.
- If the context does not contain enough information, say so clearly.
- Do not make up facts or reference documents not in the context.
- Be concise and professional.
- If you notice the context seems limited (few documents returned), mention
  that the user's role may not have access to all relevant documents.
"""


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using the confidential LLM, grounded in the
    retrieved context. Routed through the Privatemode proxy.
    """
    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n---\n\n**Question:** {question}"},
        ],
        temperature=0.2,
        max_tokens=2560,
    )

    content = completion.choices[0].message.content
    if content is None:
        return "The model did not generate a response. Please try again."
    return content.strip()


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------

def run_query(neo4j_user: str, question: str):
    """Execute the full GraphRAG query pipeline as a specific RBAC user."""

    if neo4j_user not in NEO4J_USERS:
        log.error("Unknown user '%s'. Available: %s", neo4j_user, list(NEO4J_USERS.keys()))
        sys.exit(1)

    user_config = NEO4J_USERS[neo4j_user]
    roles = ", ".join(user_config["roles"])

    log.info("=" * 60)
    log.info("User: %s  |  Roles: %s", neo4j_user, roles)
    log.info("Question: %s", question)
    log.info("=" * 60)

    # Connect as the RBAC-bound user — Neo4j enforces property-level grants
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(user_config["user"], user_config["password"]),
    )
    try:
        driver.verify_connectivity()
    except Exception as e:
        log.error("Failed to connect as %s: %s", neo4j_user, e)
        return
    log.info("Connected to Neo4j as '%s'", user_config["user"])

    # Step 1: Embed the question
    log.info("Embedding question via Privatemode...")
    query_embedding = embed_query(question)

    # Step 2: Vector search (RBAC-filtered by Neo4j)
    log.info("Performing vector search (top %d)...", TOP_K_DOCUMENTS)
    vector_docs = vector_search_documents(driver, query_embedding, TOP_K_DOCUMENTS)
    log.info("Vector search returned %d documents.", len(vector_docs))

    if not vector_docs:
        log.warning("No documents found. The user's role may not grant access to relevant content.")
        driver.close()
        print("\nAnswer: No documents matched your query within your access level.")
        return

    for doc in vector_docs:
        log.info(
            "  [%.4f] %s  (dept: %s, class: %s)",
            doc["score"], doc["title"], doc["department"], doc["classification"],
        )

    # Step 3: Graph expansion
    doc_ids = [d["doc_id"] for d in vector_docs]
    log.info("Expanding graph context from %d seed documents...", len(doc_ids))
    graph_context = graph_expand(driver, doc_ids, MAX_GRAPH_HOPS, TOP_K_ENTITIES)
    log.info(
        "Graph expansion: %d entities, %d neighbor docs, %d path entities.",
        len(graph_context["entities"]),
        len(graph_context["neighbor_documents"]),
        len(graph_context["path_entities"]),
    )

    # Step 4: Assemble context
    context = assemble_context(question, vector_docs, graph_context, roles)

    # Step 5: Generate answer via confidential LLM
    log.info("Generating answer via Privatemode LLM...")
    answer = generate_answer(question, context)

    driver.close()

    # Output
    print("\n" + "=" * 60)
    print(f"User: {neo4j_user}  |  Roles: {roles}")
    print(f"Question: {question}")
    print("=" * 60)
    print(f"\n{answer}")
    print("\n" + "-" * 60)
    print(f"Sources: {len(vector_docs)} direct + {len(graph_context['neighbor_documents'])} via graph")
    print(f"Entities: {', '.join(e['entity'] for e in graph_context['entities'][:10])}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(neo4j_user: str):
    """Run queries in a loop for hands-on demo."""
    print(f"\nSecure GraphRAG Query -- Interactive Mode")
    print(f"User: {neo4j_user}  |  Roles: {', '.join(NEO4J_USERS[neo4j_user]['roles'])}")
    print(f"Type 'quit' to exit, 'switch <user>' to change user.\n")

    current_user = neo4j_user
    while True:
        try:
            question = input(f"[{current_user}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            print("Bye!")
            break
        if question.lower().startswith("switch "):
            new_user = question.split(" ", 1)[1].strip()
            if new_user in NEO4J_USERS:
                current_user = new_user
                print(f"Switched to {current_user} (roles: {', '.join(NEO4J_USERS[current_user]['roles'])})")
            else:
                print(f"Unknown user. Available: {', '.join(NEO4J_USERS.keys())}")
            continue

        run_query(current_user, question)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Secure GraphRAG query with Neo4j RBAC enforcement."
    )
    parser.add_argument(
        "--user", "-u",
        default="alice",
        help="Neo4j user to query as (default: alice)",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Question to ask the knowledge graph",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.user)
    elif args.question:
        run_query(args.user, args.question)
    else:
        parser.print_help()
        print("\nExample (with uv or just python):")
        print('  uv run 6_graphrag_query.py -u alice -q "What is our revenue forecast?"')
        print('  uv run 6_graphrag_query.py -u bob -q "What are the engineering compensation bands?"')
        print('  uv run 6_graphrag_query.py --interactive -u alice')
        print('\n')
        print('  python 6_graphrag_query.py -u alice -q "What is our revenue forecast?"')
        print('  python 6_graphrag_query.py -u bob -q "What are the engineering compensation bands?"')
        print('  python 6_graphrag_query.py --interactive -u alice')
        print('\n')


if __name__ == "__main__":
    main()