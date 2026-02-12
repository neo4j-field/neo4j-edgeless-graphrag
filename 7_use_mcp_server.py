#!/usr/bin/env python3
"""
Secure GraphRAG Platform - MCP Server Demo
Demonstrates interaction with Neo4j MCP Server for secure graph queries
with Privatemode embedding integration for vector search.
"""

import json
import os
import requests
import base64
import time
from typing import Dict, List, Any, Optional
from urllib3.exceptions import InsecureRequestWarning
from openai import OpenAI
import httpx
from dotenv import load_dotenv

# Suppress SSL warnings for self-signed certificates
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Load .env file (looks for .env in current directory)
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — adjust these values to match your environment
# Secrets are loaded from .env file, everything else can be changed here.
# ═══════════════════════════════════════════════════════════════════════════

# -- MCP Server --
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "https://localhost:443")
MCP_CERT_PATH = os.getenv("MCP_CERT_PATH", "./mcp/certs/localhost-cert.pem")
MCP_VERIFY_SSL = os.getenv("MCP_VERIFY_SSL", "false").lower() == "true"
MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "120"))

# -- Neo4j Admin User (for schema retrieval) --
NEO4J_ADMIN_USERNAME = os.getenv("NEO4J_ADMIN_USERNAME", "neo4j")
NEO4J_ADMIN_PASSWORD = os.getenv("NEO4J_ADMIN_PASSWORD")

# -- Neo4j Demo User (for RBAC-filtered queries) --
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "bob")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secure-p@ssword-Two")

# -- Privatemode Proxy (Embedding & LLM) --
PRIVATEMODE_BASE_URL = os.getenv("PRIVATEMODE_BASE_URL", "https://localhost:8089/v1")
PRIVATEMODE_API_KEY = os.getenv("PRIVATEMODE_API_KEY")
PRIVATEMODE_CA_CERT = os.getenv("PRIVATEMODE_CA_CERT", "./privatemode/certs/localhost-cert.pem")
PRIVATEMODE_TIMEOUT = int(os.getenv("PRIVATEMODE_TIMEOUT", "120"))

# -- Models --
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

# -- Query Settings --
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))
GRAPH_TOP_K_ENTITIES = int(os.getenv("GRAPH_TOP_K_ENTITIES", "10"))
GRAPH_NEIGHBOR_LIMIT = int(os.getenv("GRAPH_NEIGHBOR_LIMIT", "5"))
SCHEMA_SAMPLE_SIZE = int(os.getenv("SCHEMA_SAMPLE_SIZE", "100"))

# -- Demo Question --
DEMO_QUESTION = os.getenv("DEMO_QUESTION", "What are the engineering compensation bands?")

# -- Validate required secrets --
if not NEO4J_ADMIN_PASSWORD:
    raise SystemExit("ERROR: NEO4J_ADMIN_PASSWORD not set. Check your .env file.")
if not PRIVATEMODE_API_KEY:
    raise SystemExit("ERROR: PRIVATEMODE_API_KEY not set. Check your .env file.")

# ═══════════════════════════════════════════════════════════════════════════
# Privatemode Embedding Client
# ═══════════════════════════════════════════════════════════════════════════

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


def embed_query(text: str) -> List[float]:
    """Embed the user query via Privatemode (confidential)."""
    print(f"  Generating embedding via Privatemode for: '{text[:60]}...'")
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    print(f"  Embedding generated ({len(embedding)} dimensions)")
    return embedding


# ═══════════════════════════════════════════════════════════════════════════
# MCP Client
# ═══════════════════════════════════════════════════════════════════════════

class MCPClient:
    """Client for interacting with Neo4j MCP Server"""
    
    def __init__(self, 
                 base_url: str = MCP_BASE_URL,
                 username: str = NEO4J_USERNAME,
                 password: str = NEO4J_PASSWORD,
                 cert_path: Optional[str] = MCP_CERT_PATH,
                 verify_ssl: bool = MCP_VERIFY_SSL,
                 timeout: int = MCP_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode('ascii')
        base64_auth = base64.b64encode(auth_bytes).decode('ascii')
        
        self.session.headers.update({
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        if verify_ssl and cert_path:
            self.session.verify = cert_path
        else:
            self.session.verify = False
            
        self.request_id = 0
    
    def _get_next_id(self) -> int:
        self.request_id += 1
        return self.request_id
    
    def _make_jsonrpc_request(self, method: str, params: Optional[Dict] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }
        
        request_timeout = timeout if timeout is not None else self.timeout
        
        try:
            print(f"  Making request: {method} (timeout: {request_timeout}s)")
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=request_timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                error_code = result["error"].get("code", "N/A")
                print(f"  ERROR - JSON-RPC [{error_code}]: {error_msg}")
                if "data" in result["error"]:
                    print(f"  Details: {result['error']['data']}")
                return {}
            
            return result
            
        except requests.exceptions.Timeout:
            print(f"  TIMEOUT - Request timed out after {request_timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            print(f"  ERROR - Request failed: {e}")
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        print("\nStep 1: Querying available tools from MCP server...")
        print("-" * 80)
        
        try:
            response = self._make_jsonrpc_request("tools/list", timeout=30)
            
            if "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                print(f"  Found {len(tools)} tools available:\n")
                
                for i, tool in enumerate(tools, 1):
                    print(f"  {i}. {tool.get('name', 'Unknown')}")
                    if 'description' in tool:
                        print(f"     Description: {tool['description']}")
                    print()
                
                return tools
            else:
                print("  WARNING - Unexpected response format")
                return []
                
        except Exception as e:
            print(f"  ERROR - Listing tools failed: {e}")
            return []
    
    def get_schema(self, sample_size: int = SCHEMA_SAMPLE_SIZE) -> Dict[str, Any]:
        print("\nStep 2: Retrieving Neo4j database schema...")
        print("-" * 80)
        print(f"  Using sample_size={sample_size} for faster inference")
        
        try:
            print(f"  Calling tool: get-schema")
            print("  This may take a moment for large databases...")
            
            response = self._make_jsonrpc_request(
                "tools/call",
                {
                    "name": "get-schema",
                    "arguments": {"sample_size": sample_size}
                },
                timeout=180
            )
            
            if "result" in response:
                schema = response["result"]
                print(f"  Successfully retrieved schema\n")
                
                if "content" in schema:
                    for content in schema["content"]:
                        if content.get("type") == "text":
                            try:
                                schema_data = json.loads(content.get("text", "{}"))
                                
                                if isinstance(schema_data, list):
                                    schema_data = self._normalize_schema(schema_data)
                                
                                self._print_data_model(schema_data)
                            except json.JSONDecodeError:
                                print("  Raw schema data:")
                                print(content.get("text", ""))
                
                return schema
            else:
                print("  WARNING - No result in response")
                return {}
                
        except requests.exceptions.Timeout:
            print("  TIMEOUT - Schema query timed out.")
            return {}
        except Exception as e:
            print(f"  ERROR - Getting schema failed: {e}")
            return {}

    def _normalize_schema(self, schema_list: List) -> Dict:
        """
        Convert list-format schema (list of {key, value} entries) into
        the dict format expected by _print_data_model.
        """
        node_props = {}
        rel_props = {}
        relationships = []

        for entry in schema_list:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key", "")
            value = entry.get("value", {})
            if not isinstance(value, dict):
                continue

            entry_type = value.get("type", "")

            if entry_type == "node":
                props = value.get("properties", {})
                node_props[key] = [
                    {"property": pname, "type": ptype}
                    for pname, ptype in props.items()
                ]
                for rel_type, rel_info in value.get("relationships", {}).items():
                    direction = rel_info.get("direction", "out")
                    targets = rel_info.get("labels", [])
                    rel_properties = rel_info.get("properties", {})

                    for target in targets:
                        if direction == "out":
                            relationships.append({"start": key, "type": rel_type, "end": target})
                        else:
                            relationships.append({"start": target, "type": rel_type, "end": key})

                    if rel_properties and rel_type not in rel_props:
                        rel_props[rel_type] = [
                            {"property": pname, "type": ptype}
                            for pname, ptype in rel_properties.items()
                        ]

            elif entry_type == "relationship":
                props = value.get("properties", {})
                if props and key not in rel_props:
                    rel_props[key] = [
                        {"property": pname, "type": ptype}
                        for pname, ptype in props.items()
                    ]

        # Deduplicate relationships
        seen = set()
        unique_rels = []
        for rel in relationships:
            sig = (rel["start"], rel["type"], rel["end"])
            if sig not in seen:
                seen.add(sig)
                unique_rels.append(rel)

        return {
            "node_props": node_props,
            "rel_props": rel_props,
            "relationships": unique_rels,
        }

    def _print_data_model(self, schema_data: Dict):
        """Pretty-print the full data model from schema response."""
        
        # ── Node labels & properties ─────────────────────────────
        node_props = schema_data.get("node_props", {})
        if node_props:
            total_props = sum(len(p) for p in node_props.values())
            print(f"  Node Labels ({len(node_props)} labels, {total_props} properties total):")
            print("  " + "-" * 58)
            for label, props in sorted(node_props.items()):
                print(f"    :{label}")
                for prop in props:
                    name = prop.get('property', '?')
                    ptype = prop.get('type', '?')
                    print(f"      {name:<30} {ptype}")
            print()

        # ── Relationship types & properties ──────────────────────
        rel_props = schema_data.get("rel_props", {})
        if rel_props:
            print(f"  Relationship Properties ({len(rel_props)} types with properties):")
            print("  " + "-" * 58)
            for rel_type, props in sorted(rel_props.items()):
                print(f"    [:{rel_type}]")
                for prop in props:
                    name = prop.get('property', '?')
                    ptype = prop.get('type', '?')
                    print(f"      {name:<30} {ptype}")
            print()

        # ── Graph patterns (relationships) ───────────────────────
        relationships = schema_data.get("relationships", [])
        if relationships:
            print(f"  Graph Patterns ({len(relationships)} patterns):")
            print("  " + "-" * 58)
            for rel in relationships:
                start = rel.get('start', '?')
                rtype = rel.get('type', '?')
                end = rel.get('end', '?')
                print(f"    (:{start})-[:{rtype}]->(:{end})")
            print()

        # ── Summary ──────────────────────────────────────────────
        rel_types = set(r.get('type', '') for r in relationships)
        rel_types_with_props = set(rel_props.keys())

        print("  Data Model Summary:")
        print("  " + "-" * 58)
        print(f"    Node labels:          {len(node_props)}")
        print(f"    Relationship types:   {len(rel_types)}" +
              (f" ({len(rel_types_with_props)} with properties)" if rel_types_with_props else ""))
        print(f"    Graph patterns:       {len(relationships)}")
        print()
    
    def execute_query(self, query: str, params: Optional[Dict] = None, timeout: int = 60) -> Dict[str, Any]:
        try:
            response = self._make_jsonrpc_request(
                "tools/call",
                {
                    "name": "read-cypher",
                    "arguments": {
                        "query": query,
                        "params": params or {}
                    }
                },
                timeout=timeout
            )
            
            if "result" in response:
                return response["result"]
            return {}
                
        except Exception as e:
            print(f"  ERROR - Query execution failed: {e}")
            return {}

    def parse_result_data(self, result: Dict) -> Optional[List]:
        """Extract parsed JSON data from MCP result content."""
        if not result or "content" not in result:
            return None
        for content in result["content"]:
            if content.get("type") == "text":
                try:
                    data = json.loads(content.get("text", "[]"))
                    if data and len(data) > 0:
                        return data
                except json.JSONDecodeError as e:
                    print(f"  WARNING - Could not parse JSON: {e}")
                    print(f"  Raw response: {content.get('text', '')[:500]}")
        return None

    def conversation_demo(self, question: str = DEMO_QUESTION):
        print("\nStep 3: Running conversation demo...")
        print("-" * 80)
        print(f"  Question: {question}\n")
        
        # Test connectivity
        print("  Testing database connectivity...")
        result = self.execute_query("RETURN 1 as test", timeout=30)
        if result:
            print("  Database connection successful\n")
        else:
            print("  ERROR - Could not connect to database\n")
            return
        
        print(f"  Answering: '{question}'\n")
        
        results_found = False

        # ── Query 1: Vector search with real embedding ───────────
        if not results_found:
            print("  Trying: Vector search with Privatemode embedding...")
            try:
                query_embedding = embed_query(question)
                
                result = self.execute_query(
                    """
                    CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
                    YIELD node AS doc, score
                    RETURN doc.id             AS doc_id,
                           doc.title          AS title,
                           doc.content        AS content,
                           doc.department     AS department,
                           doc.classification AS classification,
                           score
                    ORDER BY score DESC
                    """,
                    params={"top_k": VECTOR_TOP_K, "embedding": query_embedding},
                    timeout=60
                )
                
                data = self.parse_result_data(result)
                if data:
                    results_found = True
                    print(f"  Vector search returned {len(data)} documents\n")
                    
                    print("  Relevant Documents:\n")
                    for i, doc in enumerate(data, 1):
                        score = doc.get('score', 0)
                        title = doc.get('title', 'N/A')
                        dept = doc.get('department', 'N/A')
                        classification = doc.get('classification', 'N/A')
                        content = doc.get('content', '')
                        
                        print(f"    [{i}] {title}")
                        print(f"        Score: {score:.4f} | Dept: {dept} | Class: {classification}")
                        if content:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            print(f"        Content: {preview}")
                        print()
                    
                    # ── Graph expansion: find related entities ────────
                    doc_ids = [d['doc_id'] for d in data if d.get('doc_id')]
                    if doc_ids:
                        print("  Expanding graph context (related entities)...")
                        entity_result = self.execute_query(
                            """
                            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
                            WHERE d.id IN $doc_ids
                            WITH e, count(d) AS relevance
                            ORDER BY relevance DESC
                            LIMIT $top_k
                            RETURN e.name AS entity, relevance
                            """,
                            params={"doc_ids": doc_ids, "top_k": GRAPH_TOP_K_ENTITIES},
                            timeout=30
                        )
                        
                        entities = self.parse_result_data(entity_result)
                        if entities:
                            entity_names = [e['entity'] for e in entities]
                            print(f"  Related entities: {', '.join(entity_names)}\n")
                        
                        # ── Neighbor documents via graph traversal ────
                        print("  Finding related documents via graph traversal...")
                        neighbor_result = self.execute_query(
                            """
                            MATCH (d:Document)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbor:Document)
                            WHERE d.id IN $doc_ids
                              AND NOT neighbor.id IN $doc_ids
                            WITH DISTINCT neighbor, collect(DISTINCT e.name) AS shared_entities
                            RETURN neighbor.title       AS title,
                                   neighbor.department   AS department,
                                   shared_entities
                            ORDER BY size(shared_entities) DESC
                            LIMIT $limit
                            """,
                            params={"doc_ids": doc_ids, "limit": GRAPH_NEIGHBOR_LIMIT},
                            timeout=30
                        )
                        
                        neighbors = self.parse_result_data(neighbor_result)
                        if neighbors:
                            print(f"  Found {len(neighbors)} related documents via graph:\n")
                            for n in neighbors:
                                shared = ', '.join(n.get('shared_entities', []))
                                print(f"    {n.get('title', 'N/A')} (dept: {n.get('department', 'N/A')})")
                                print(f"      Shared entities: {shared}")
                            print()

            except Exception as e:
                print(f"  WARNING - Vector search failed: {e}\n")

        # ── Query 2: Fulltext search fallback ────────────────────
        if not results_found:
            print("  Trying: Fulltext search fallback...")
            result = self.execute_query(
                """
                MATCH (d:Document)
                WHERE d.content CONTAINS 'compensation'
                   OR d.content CONTAINS 'salary'
                   OR d.title CONTAINS 'compensation'
                RETURN d.title AS title,
                       d.content AS content,
                       d.department AS department,
                       d.classification AS classification
                LIMIT 10
                """,
                timeout=30
            )
            
            data = self.parse_result_data(result)
            if data:
                results_found = True
                print(f"  Fulltext search returned {len(data)} documents\n")
                print(json.dumps(data, indent=2))

        # ── Query 3: Database exploration fallback ───────────────
        if not results_found:
            print("  Trying: Database exploration...")
            result = self.execute_query(
                """
                MATCH (n)
                WITH DISTINCT labels(n) AS nodeLabels, count(*) AS count
                RETURN nodeLabels, count
                ORDER BY count DESC
                LIMIT 10
                """,
                timeout=30
            )
            
            data = self.parse_result_data(result)
            if data:
                print("  Database contents:\n")
                print(json.dumps(data, indent=2))
            else:
                print("  WARNING - No data found with any query pattern.")

        print("\n  Demo completed!")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Secure GraphRAG Platform - MCP Server Demo")
    print("=" * 80)
    
    print(f"\n  Connecting as user: {NEO4J_USERNAME}")
    
    client = MCPClient()
    
    try:
        # Step 1: List available tools
        tools = client.list_tools()
        
        # Step 2: Get database schema (as admin for full visibility)
        print(f"\n  Switching to admin user '{NEO4J_ADMIN_USERNAME}' for schema retrieval...")
        admin_client = MCPClient(username=NEO4J_ADMIN_USERNAME, password=NEO4J_ADMIN_PASSWORD)
        schema = admin_client.get_schema()
        
        # Step 3: Run conversation demo with RBAC-filtered user
        print(f"\n  Switching back to demo user '{NEO4J_USERNAME}' (RBAC-filtered)...")
        client.conversation_demo()
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted by user")
    except Exception as e:
        print(f"\n  ERROR - Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()