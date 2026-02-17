# neo4j-edgeless-graphrag

Description will be added soon

## Troubleshooting Neo4j Browser connection issues

Getting this error?

![Browser Connection Error](graphics/browser-connect-fix.png  =100x300)

1. Accept the cert for the Bolt port in your browser
Navigate directly to https://localhost:7688 in the same browser you're using for Neo4j Browser. You'll get a security warning — click through to accept the certificate. This allows the browser to trust the Bolt WebSocket connection.
2. Connect using the right scheme
In Neo4j Browser at https://localhost:7473 (accept that cert too if prompted), use this connect URL:
bolt+s://localhost:7688
The browser doesn't support bolt+ssc:// directly, so you need to manually accept the certs first via step 1, then bolt+s:// will work.
