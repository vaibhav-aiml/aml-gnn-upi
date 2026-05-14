"""
Neo4j database client for graph storage and querying
"""
import logging

logger = logging.getLogger(__name__)

# Try to import neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j not installed. Install with: pip install neo4j")

# Try to import settings
try:
    from config.settings import settings
except ImportError:
    settings = None
    logger.warning("Settings not available")

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        if not NEO4J_AVAILABLE:
            print("⚠️ Neo4j not installed. Skipping Neo4j features.")
            self.driver = None
            return
            
        self.uri = uri or (getattr(settings, 'neo4j_uri', 'bolt://localhost:7687') if settings else 'bolt://localhost:7687')
        self.user = user or (getattr(settings, 'neo4j_user', 'neo4j') if settings else 'neo4j')
        self.password = password or (getattr(settings, 'neo4j_password', 'password123') if settings else 'password123')
        self.driver = None
        
    def connect(self):
        """Establish connection to Neo4j"""
        if not NEO4J_AVAILABLE or not self.driver:
            print("⚠️ Neo4j not available")
            return False
            
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_transaction_graph(self, transactions_df):
        """Create graph nodes and relationships from transactions"""
        if not NEO4J_AVAILABLE:
            print("⚠️ Neo4j not available - skipping graph creation")
            return
            
        with self.driver.session() as session:
            # Create constraint for unique accounts
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
            
            # Create accounts (nodes)
            for account in set(transactions_df['from_account'].unique()) | set(transactions_df['to_account'].unique()):
                session.run(
                    "MERGE (a:Account {id: $id}) RETURN a",
                    id=account
                )
            
            # Create transactions (relationships)
            for _, row in transactions_df.iterrows():
                session.run(
                    """
                    MATCH (from:Account {id: $from_id})
                    MATCH (to:Account {id: $to_id})
                    CREATE (from)-[t:TRANSACTED {
                        amount: $amount,
                        timestamp: $timestamp,
                        is_fraud: $is_fraud
                    }]->(to)
                    """,
                    from_id=row['from_account'],
                    to_id=row['to_account'],
                    amount=row['amount'],
                    timestamp=str(row['timestamp']),
                    is_fraud=row.get('is_fraud', 0)
                )
            
            logger.info(f"Created graph with {len(transactions_df)} transactions")
    
    def detect_suspicious_subgraphs(self, min_risk=70):
        """Query for suspicious subgraphs using Cypher"""
        if not NEO4J_AVAILABLE:
            return []
            
        query = """
        MATCH path = (a:Account)-[:TRANSACTED*1..5]->(b:Account)
        WHERE a.risk_score > $min_risk OR b.risk_score > $min_risk
        RETURN path, length(path) as depth
        LIMIT 100
        """
        with self.driver.session() as session:
            result = session.run(query, min_risk=min_risk)
            return list(result)
    
    def get_account_network(self, account_id, depth=2):
        """Get the transaction network around an account"""
        if not NEO4J_AVAILABLE:
            return []
            
        query = """
        MATCH path = (a:Account {id: $account_id})-[:TRANSACTED*1..%d]->(connected)
        RETURN connected.id as account, 
               relationships(path) as transactions
        """ % depth
        with self.driver.session() as session:
            result = session.run(query, account_id=account_id)
            return list(result)

if __name__ == "__main__":
    print(f"Neo4j available: {NEO4J_AVAILABLE}")