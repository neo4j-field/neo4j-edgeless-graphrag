// === RBAC Configuration (run as admin) ===

// Create roles for different access levels
CREATE ROLE finance_reader IF NOT EXISTS;
CREATE ROLE hr_reader IF NOT EXISTS;
CREATE ROLE executive_reader IF NOT EXISTS;
CREATE ROLE analyst IF NOT EXISTS;

// Grant base access to the neo4j database
GRANT ACCESS ON DATABASE neo4j TO finance_reader;
GRANT ACCESS ON DATABASE neo4j TO hr_reader;
GRANT ACCESS ON DATABASE neo4j TO executive_reader;
GRANT ACCESS ON DATABASE neo4j TO analyst;

// Finance role: can read finance-labeled neo4j
GRANT MATCH {*} ON GRAPH neo4j 
  FOR (document:Document) 
  WHERE document.department = 'finance'
  TO finance_reader;

// HR role: can read HR-labeled neo4j  
GRANT MATCH {*} ON GRAPH neo4j
  FOR  (document:Document)
  WHERE document.department = 'hr'
  TO hr_reader;

// Executive role: can read all neo4j marked for executive access
GRANT MATCH {*} ON GRAPH neo4j
  FOR  (document:Document)
  WHERE document.executiveAccess = true
  TO executive_reader;

// Analyst role: can read all public and internal neo4j
GRANT MATCH {*} ON GRAPH neo4j
  FOR  (document:Document)
  WHERE document.classification IN ['public', 'internal']
  TO analyst;

// All roles can traverse to entities (for context)
GRANT MATCH {*} ON GRAPH neo4j NODES Entity TO finance_reader;
GRANT MATCH {*} ON GRAPH neo4j NODES Entity TO hr_reader;
GRANT MATCH {*} ON GRAPH neo4j NODES Entity TO executive_reader;
GRANT MATCH {*} ON GRAPH neo4j NODES Entity TO analyst;

// Grant relationship traversal
GRANT MATCH {*} ON GRAPH neo4j RELATIONSHIPS * TO finance_reader;
GRANT MATCH {*} ON GRAPH neo4j RELATIONSHIPS * TO hr_reader;
GRANT MATCH {*} ON GRAPH neo4j RELATIONSHIPS * TO executive_reader;
GRANT MATCH {*} ON GRAPH neo4j RELATIONSHIPS * TO analyst;

// === Create Users and Assign Roles ===
CREATE USER alice SET PASSWORD 'secure-p@ssword-One' CHANGE NOT REQUIRED;
CREATE USER bob SET PASSWORD 'secure-p@ssword-Two' CHANGE NOT REQUIRED;

GRANT ROLE finance_reader TO alice;
GRANT ROLE analyst TO alice;

GRANT ROLE hr_reader TO bob;
GRANT ROLE analyst TO bob;