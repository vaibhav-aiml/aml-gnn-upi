"""
Graph visualization component for Streamlit dashboard
"""
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from streamlit.components.v1 import html

class GraphVisualizationComponent:
    """Interactive graph visualization component"""
    
    @staticmethod
    def create_force_directed_graph(adj_matrix, labels=None, risk_scores=None):
        """Create force-directed graph visualization"""
        
        # Create networkx graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Get positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=1, color='#cccccc'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Color based on risk score
            if risk_scores and node < len(risk_scores):
                risk = risk_scores[node]
                if risk > 85:
                    node_colors.append('#8e44ad')  # Purple - Critical
                elif risk > 70:
                    node_colors.append('#e74c3c')  # Red - High
                elif risk > 40:
                    node_colors.append('#f39c12')  # Orange - Medium
                else:
                    node_colors.append('#2ecc71')  # Green - Low
                node_sizes.append(15 + risk / 10)
                node_texts.append(f"Node {node}<br>Risk: {risk:.1f}")
            else:
                node_colors.append('#3498db')
                node_sizes.append(15)
                node_texts.append(f"Node {node}")
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_texts,
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Transaction Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_hierarchical_graph(data, root_node=None):
        """Create hierarchical tree visualization"""
        
        if root_node is None:
            root_node = data.get('root', 0)
        
        G = nx.DiGraph()
        
        # Build tree
        for edge in data.get('edges', []):
            G.add_edge(edge[0], edge[1], weight=edge.get('weight', 1))
        
        # Get hierarchical layout
        pos = nx.spring_layout(G, k=1, iterations=30)
        
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=1, color='gray'),
                showlegend=False
            ))
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='white')),
            text=[str(node) for node in G.nodes()],
            textposition="bottom center",
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Transaction Tree (Root: {root_node})",
            height=500,
            showlegend=False,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    @staticmethod
    def display_risk_heatmap(adj_matrix, risk_scores):
        """Display risk heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix[:50, :50],  # Show first 50 nodes
            colorscale='Reds',
            zmin=0,
            zmax=1,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Transaction Heatmap (First 50 Nodes)",
            xaxis_title="Target Account",
            yaxis_title="Source Account",
            height=500,
            width=600
        )
        
        return fig
    
    @staticmethod
    def create_animated_graph(adj_matrix, timestamps):
        """Create animated graph over time"""
        
        frames = []
        
        # Create frames for different time points
        for t in range(min(10, len(timestamps))):
            frame_data = []
            
            # Add nodes and edges for this time point
            # (Simplified for demo)
            frame_data.append(go.Scatter(
                x=[0, 1, 2],
                y=[0, 1, 0],
                mode='markers',
                marker=dict(size=10)
            ))
            
            frames.append(go.Frame(data=frame_data, name=f"frame{t}"))
        
        fig = go.Figure(
            data=frames[0].data if frames else [],
            layout=go.Layout(
                title="Temporal Graph Evolution",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                 method="animate",
                                 args=[None])]
                )],
                height=500
            ),
            frames=frames
        )
        
        return fig

class SimpleGraphViewer:
    """Simple graph viewer for basic visualization"""
    
    @staticmethod
    def view_transaction_chain(transactions_df, account_id, depth=2):
        """View transaction chain for a specific account"""
        
        st.subheader(f"Transaction Chain for {account_id}")
        
        # Filter transactions
        related_tx = transactions_df[
            (transactions_df['from_account'] == account_id) |
            (transactions_df['to_account'] == account_id)
        ].head(20)
        
        if related_tx.empty:
            st.info("No transactions found for this account")
            return
        
        # Display as table
        st.dataframe(
            related_tx[['from_account', 'to_account', 'amount', 'is_fraud']],
            use_container_width=True,
            column_config={
                'amount': st.column_config.NumberColumn(format="₹%.2f"),
                'is_fraud': st.column_config.CheckboxColumn("Fraud Flag")
            }
        )
    
    @staticmethod
    def display_network_stats(G):
        """Display network statistics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        with col2:
            st.metric("Edges", G.number_of_edges())
        with col3:
            density = nx.density(G)
            st.metric("Density", f"{density:.3f}")
        with col4:
            components = nx.number_weakly_connected_components(G)
            st.metric("Components", components)
        
        # Degree distribution
        degrees = [d for n, d in G.degree()]
        st.subheader("Degree Distribution")
        st.bar_chart(pd.Series(degrees).value_counts().head(10))

if __name__ == "__main__":
    print("Graph visualization components ready")