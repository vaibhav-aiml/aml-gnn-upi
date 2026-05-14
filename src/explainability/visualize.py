"""
Visualization utilities for explanations
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ExplanationVisualizer:
    def __init__(self):
        self.colors = {
            'low_risk': '#2ecc71',     # Green
            'medium_risk': '#f39c12',   # Orange
            'high_risk': '#e74c3c',     # Red
            'critical': '#8e44ad'       # Purple
        }
    
    def plot_risk_distribution(self, risk_scores, title="Risk Score Distribution"):
        """Plot histogram of risk scores"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=20,
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig.add_vline(x=30, line_dash="dash", line_color="green", 
                     annotation_text="Low Risk")
        fig.add_vline(x=60, line_dash="dash", line_color="orange",
                     annotation_text="Medium Risk")
        fig.add_vline(x=85, line_dash="dash", line_color="red",
                     annotation_text="High Risk")
        
        fig.update_layout(
            title=title,
            xaxis_title="Risk Score",
            yaxis_title="Count",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, top_k=10):
        """Plot top K feature importance"""
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[-top_k:]
        top_features = [feature_names[i] for i in sorted_idx]
        top_scores = [importance_scores[i] for i in sorted_idx]
        
        fig = go.Figure(go.Bar(
            x=top_scores,
            y=top_features,
            orientation='h',
            marker_color='#e74c3c',
            text=top_scores,
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def plot_explanation_subgraph(self, edge_index, important_edges, node_labels=None):
        """Visualize the important subgraph for explanation"""
        G = nx.Graph()
        
        # Add edges
        for i, (u, v) in enumerate(edge_index.T):
            weight = important_edges[i] if i < len(important_edges) else 0
            if weight > 0.5:  # Only show important edges
                G.add_edge(int(u), int(v), weight=weight)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=weight*5, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_colors = []
        for node in G.nodes():
            if node_labels and node in node_labels:
                risk = node_labels[node]
                if risk > 85:
                    node_colors.append(self.colors['critical'])
                elif risk > 70:
                    node_colors.append(self.colors['high_risk'])
                elif risk > 40:
                    node_colors.append(self.colors['medium_risk'])
                else:
                    node_colors.append(self.colors['low_risk'])
            else:
                node_colors.append('blue')
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=20, color=node_colors, line=dict(width=2, color='white')),
            text=[str(n) for n in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Important Subgraph for Prediction",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def plot_attention_weights(self, attention_matrix, layer_name="Layer 1"):
        """Plot attention weights heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Attention Weights - {layer_name}",
            xaxis_title="Target Node",
            yaxis_title="Source Node",
            height=500,
            width=600
        )
        
        return fig
    
    def create_dashboard(self, risk_scores, feature_importance, explanations):
        """Create comprehensive explanation dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Risk Distribution", "Feature Importance",
                          "Explanation Subgraph", "Confidence Score"),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Add risk distribution
        fig.add_trace(
            go.Histogram(x=risk_scores, nbinsx=20),
            row=1, col=1
        )
        
        # Add feature importance
        if feature_importance:
            features = list(feature_importance.keys())[:5]
            scores = list(feature_importance.values())[:5]
            fig.add_trace(
                go.Bar(x=features, y=scores),
                row=1, col=2
            )
        
        # Add confidence indicator
        avg_confidence = np.mean([exp.get('confidence', 0.8) for exp in explanations])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_confidence * 100,
                title={"text": "Avg Confidence (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, template="plotly_white")
        return fig

class GraphVisualizer:
    """Simple graph visualization utilities"""
    
    @staticmethod
    def create_network_graph(adj_matrix, labels=None):
        """Create network graph from adjacency matrix"""
        G = nx.from_numpy_array(adj_matrix)
        
        pos = nx.spring_layout(G)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                               node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                               alpha=0.5, ax=ax)
        
        if labels:
            nx.draw_networkx_labels(G, pos, labels, ax=ax)
        
        ax.set_title("Transaction Network")
        ax.axis('off')
        
        return fig

if __name__ == "__main__":
    visualizer = ExplanationVisualizer()
    print("Visualization tools ready")