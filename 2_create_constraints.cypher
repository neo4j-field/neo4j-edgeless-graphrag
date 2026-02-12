// === Schema Setup ===

// Create constraints for our document graph
CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS  
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Create vector index for semantic search
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:Document)
ON d.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 2560,
  `vector.similarity_function`: 'cosine'
}};

// Create Entity constraints and vector index (idempotent)
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 2560,
    `vector.similarity_function`: 'cosine'
 }};