// ============================================
// Neo4j Cypher Queries for AML Detection
// ============================================

// 1. Find Smurfing Patterns
// Accounts sending to 20+ unique recipients in short time
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WITH a, COUNT(DISTINCT b) as recipients, SUM(t.amount) as total_amount
WHERE recipients > 20 AND total_amount > 50000
RETURN a.id as account, recipients, total_amount
ORDER BY recipients DESC
LIMIT 20;

// 2. Find Layering Chains
// Paths of 5+ transactions with consistent amounts
MATCH path = (start:Account)-[:TRANSACTED*5..10]->(end:Account)
WHERE ALL(rel IN relationships(path) 
          WHERE rel.amount > 1000 AND rel.amount < 100000)
WITH path, 
     [rel IN relationships(path) | rel.amount] as amounts,
     length(path) as depth
WHERE stdDev(amounts) / avg(amounts) < 0.2  // Consistent amounts
RETURN start.id as source, 
       end.id as destination, 
       depth,
       avg(amounts) as avg_amount
LIMIT 50;

// 3. Find Round-tripping
// Money that returns to source within 10 hops
MATCH path = (a:Account)-[:TRANSACTED*3..10]->(a)
WHERE length(path) >= 3
WITH a, path, 
     [rel IN relationships(path) | rel.amount] as amounts,
     length(path) as cycle_length
WHERE stdDev(amounts) / avg(amounts) < 0.3
RETURN a.id as account, 
       cycle_length,
       avg(amounts) as amount,
       startNode(path).id as start,
       endNode(path).id as end
LIMIT 30;

// 4. High Velocity Detection
// Accounts with >50 transactions in last hour
MATCH (a:Account)-[t:TRANSACTED]->()
WHERE t.timestamp > datetime() - duration('PT1H')
WITH a, COUNT(t) as tx_count
WHERE tx_count > 50
RETURN a.id as account, tx_count
ORDER BY tx_count DESC;

// 5. Round Number Detection
// Transactions with round amounts (likely structured)
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WHERE t.amount % 1000 = 0 AND t.amount < 100000
WITH a, b, COUNT(*) as round_tx_count, SUM(t.amount) as total
WHERE round_tx_count > 10
RETURN a.id as from_account, 
       b.id as to_account, 
       round_tx_count, 
       total
ORDER BY round_tx_count DESC;

// 6. Connected Components (Fraud Rings)
// Find interconnected suspicious accounts
MATCH (a:Account)-[:TRANSACTED*1..3]-(b:Account)
WHERE a.risk_score > 70 OR b.risk_score > 70
WITH COLLECT(DISTINCT a) + COLLECT(DISTINCT b) as nodes
RETURN nodes, size(nodes) as ring_size
ORDER BY ring_size DESC
LIMIT 10;

// 7. Temporal Pattern Detection
// Transactions occurring at unusual hours (2-4 AM)
MATCH (a:Account)-[t:TRANSACTED]->(b:Account)
WHERE t.timestamp.hour >= 2 AND t.timestamp.hour <= 4
WITH a, COUNT(t) as night_tx, SUM(t.amount) as night_amount
WHERE night_tx > 20
RETURN a.id as account, night_tx, night_amount
ORDER BY night_tx DESC;

// 8. Mule Account Detection
// High inbound then high outbound activity
MATCH (a:Account)
OPTIONAL MATCH (a)<-[in:TRANSACTED]-()
OPTIONAL MATCH (a)-[out:TRANSACTED]->()
WITH a, 
     COUNT(DISTINCT in) as inbound_count,
     SUM(in.amount) as inbound_amount,
     COUNT(DISTINCT out) as outbound_count,
     SUM(out.amount) as outbound_amount
WHERE inbound_count > 10 AND outbound_count > 10
  AND inbound_amount > 100000 AND outbound_amount > 100000
RETURN a.id as mule_account,
       inbound_count, inbound_amount,
       outbound_count, outbound_amount
ORDER BY (inbound_amount + outbound_amount) DESC;

// 9. Path to High-Risk Account
// Find all paths leading to known fraudulent accounts
MATCH path = (a:Account)-[:TRANSACTED*1..5]->(b:Account)
WHERE b.risk_score > 85
RETURN a.id as source,
       b.id as fraud_account,
       length(path) as distance,
       [rel IN relationships(path) | rel.amount] as amounts
LIMIT 100;

// 10. Cleanup Old Data
// Remove transactions older than 90 days
MATCH ()-[t:TRANSACTED]->()
WHERE t.timestamp < datetime() - duration('P90D')
DELETE t;