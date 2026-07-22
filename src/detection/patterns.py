"""
Pattern detection for money laundering (Optimized version)
"""
import networkx as nx
from collections import defaultdict
import numpy as np

class PatternDetector:
    def __init__(self, graph):
        """
        Args:
            graph: NetworkX graph with transaction data
        """
        self.graph = graph
        
    def detect_smurfing(self, min_splits=10, min_amount=5000):
        """
        Detect smurfing patterns: One source splitting to many accounts
        Optimized: Only checks nodes with high out-degree
        """
        smurfing_patterns = []
        
        # Only check nodes that have enough outgoing edges
        for node in self.graph.nodes():
            out_edges = list(self.graph.out_edges(node, data=True))
            
            if len(out_edges) >= min_splits:
                total_out = sum(edge[2].get('amount', 0) for edge in out_edges)
                
                if total_out >= min_amount:
                    pattern = {
                        'source': node,
                        'num_splits': len(out_edges),
                        'total_amount': total_out,
                        'targets': [edge[1] for edge in out_edges][:10],  # Limit targets
                        'pattern_type': 'smurfing'
                    }
                    smurfing_patterns.append(pattern)
        
        return smurfing_patterns
    
    def detect_layering(self, min_depth=3, max_depth=5):
        """
        Detect layering: Money through chain of accounts
        Optimized: Limited depth and early stopping
        """
        layering_patterns = []
        visited_chains = set()
        
        # Only check nodes with enough transactions
        for node in list(self.graph.nodes())[:100]:  # Limit to first 100 nodes
            # BFS with depth limit
            queue = [(node, [node], 0)]
            
            while queue:
                current, path, depth = queue.pop(0)
                
                if depth >= min_depth and len(path) >= min_depth:
                    chain_key = tuple(path[:min_depth])
                    if chain_key not in visited_chains:
                        pattern = {
                            'source': path[0],
                            'target': path[-1],
                            'chain': path,
                            'length': len(path),
                            'pattern_type': 'layering'
                        }
                        layering_patterns.append(pattern)
                        visited_chains.add(chain_key)
                
                if depth >= max_depth:
                    continue
                
                for neighbor in list(self.graph.successors(current))[:5]:  # Limit neighbors
                    if neighbor not in path:
                        queue.append((neighbor, path + [neighbor], depth + 1))
                
                if len(layering_patterns) >= 50:  # Limit results
                    break
            
            if len(layering_patterns) >= 50:
                break
        
        return layering_patterns
    
    def detect_round_tripping(self, min_hops=3, max_hops=6):
        """
        Detect round-tripping: Money returns to source
        Optimized: Compute global cycles once outside node loop
        """
        round_trips = []
        try:
            cycles = list(nx.simple_cycles(self.graph, length_bound=max_hops))
        except Exception:
            return round_trips

        nodes_to_check = set(list(self.graph.nodes())[:50])
        for cycle in cycles:
            if len(cycle) >= min_hops and cycle[0] in nodes_to_check:
                if self._check_cycle_amounts_fast(cycle):
                    pattern = {
                        'source': cycle[0],
                        'cycle': cycle,
                        'length': len(cycle),
                        'pattern_type': 'round_tripping'
                    }
                    round_trips.append(pattern)
                    if len(round_trips) >= 20:
                        break

        return round_trips
    
    def _check_cycle_amounts_fast(self, cycle):
        """Fast check for cycle amounts"""
        amounts = []
        for i in range(len(cycle)):
            from_node = cycle[i]
            to_node = cycle[(i+1) % len(cycle)]
            edge_data = self.graph.get_edge_data(from_node, to_node)
            if edge_data:
                if 'amount' in edge_data:
                    amount = edge_data['amount']
                elif isinstance(edge_data, dict) and len(edge_data) > 0:
                    first_val = list(edge_data.values())[0]
                    if isinstance(first_val, dict):
                        amount = first_val.get('amount', 0)
                    else:
                        amount = first_val
                else:
                    amount = 0
                amounts.append(amount)
        
        if len(amounts) < 2:
            return False
        
        mean_amount = float(np.mean(amounts))
        if mean_amount == 0:
            return False
            
        for amount in amounts[:5]:  # Check first 5 only
            if abs(amount - mean_amount) / mean_amount > 0.3:
                return False
        
        return True
    
    def get_all_patterns(self):
        """Detect all fraud patterns (optimized)"""
        print("  Detecting smurfing patterns...")
        smurfing = self.detect_smurfing()
        print(f"    Found {len(smurfing)} smurfing patterns")
        
        print("  Detecting layering patterns...")
        layering = self.detect_layering()
        print(f"    Found {len(layering)} layering patterns")
        
        print("  Detecting round-tripping patterns...")
        round_tripping = self.detect_round_tripping()
        print(f"    Found {len(round_tripping)} round-tripping patterns")
        
        patterns = {
            'smurfing': smurfing,
            'layering': layering,
            'round_tripping': round_tripping
        }
        
        total_patterns = sum(len(p) for p in patterns.values())
        print(f"\n✅ Total suspicious patterns detected: {total_patterns}")
        
        return patterns