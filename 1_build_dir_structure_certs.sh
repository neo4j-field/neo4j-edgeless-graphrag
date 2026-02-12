# create directory structure in project folder
mkdir -p neo4j/{certificates/{bolt,https},conf,data,import,logs,metrics}
mkdir -p privatemode/certs
mkdir -p mcp/certs

# Uncomment the chmod command if you have no neo4j user created (neo4j user is preferred)
# Change neo4j directory permssions (less secure)
#chmod -R 777 ./neo4j/

# For bolt
openssl req -newkey rsa:4096 -nodes -keyout neo4j/certificates/bolt/private.key \
  -x509 -days 365 -out neo4j/certificates/bolt/public.crt \
  -subj "/CN=localhost/O=edgelessNeo/C=DE"

# For https
openssl req -newkey rsa:4096 -nodes -keyout neo4j/certificates/https/private.key \
  -x509 -days 365 -out neo4j/certificates/https/public.crt \
  -subj "/CN=localhost/O=edgelessNeo/C=DE"

# Generate private key and certificate for Privatemode proxy
openssl req -newkey rsa:4096 -nodes \
  -keyout privatemode/certs/localhost-key.pem \
  -x509 -days 365 \
  -out privatemode/certs/localhost-cert.pem \
  -subj "/CN=localhost/O=edgelessNeo/C=DE"

# Generate private key and certificate for Neo4j MCP Server
openssl req -newkey rsa:4096 -nodes \
  -keyout mcp/certs/server.key \
  -x509 -days 365 \
  -out mcp/certs/server.crt \
  -subj "/CN=localhost/O=edgelessNeo/C=DE"

# Change neo4j permssions
chmod -R 777 ./neo4j/

# Change permissions accordingly
chmod 600 neo4j/certificates/*/private.key
chmod 644 neo4j/certificates/*/public.crt
chmod 600 privatemode/certs/*.pem
chmod 600 mcp/certs/server.key
chmod 644 mcp/certs/server.crt
