import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

class Visulaizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        traversal_graph = nx.DiGraph()

        # Add nodes and edges from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u,v,data in graph.edges(data= True):
            traversal_graph.add_edge(u,v, **data)
        
        fig, ax = plt.subplots(figsize= (16,12))

        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        edges = traversal_graph.edges()
        edge_weighs = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color = edge_weighs,
                               edge_cmap= plt.cm.Blues,
                               width=2,
                               ax=ax)
        
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)
        
        edge_offset = 0.1
        for i in range(len(traversal_path)-1):
            start = traversal_path[i]
            end = traversal_path[i+1]
            start_pos = pos[start]
            end_pos = pos[end]

            mid_point = ((start_pos[0]+end_pos[0])/2, (start_pos[1]+end_pos[1])/2)
            control_point = (mid_point[0]+edge_offset, mid_point[1]+edge_offset)
            # dx = end_pos[0] - start_pos[0]
            # dy = end_pos[1] - start_pos[1]
            arrow = patches.FancyArrow(start_pos, end_pos, 
                                       connectionstyle = f"arc3, rad={0.3}",
                                       color='red',
                                       arrowstyle= "->",
                                       mutation_scale = 20,
                                       linestyle = '--',
                                       linewidth=2,
                                       zorder=4)
            ax.add_patch(arrow)
        
        labels = {}

        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label= f"{i+1}. {concepts[0] if concepts else ''}"
            labels[node] = label
        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

            nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

            start_node = traversal_path[0]
            end_node = traversal_path[-1]

            nx.draw_networkx_nodes(traversal_graph, pos,
                                   nodelist=[start_node],
                                   node_color='lightgreen',
                                   node_size=3000,
                                   ax=ax)
            nx.draw_networkx_nodes(traversal_graph, pos,
                                   nodelist=[end_node],
                                   node_color='lightcoral',
                                   node_size=3000,
                                   ax=ax)
            ax.set_title('Graph Traversal Flow')
            ax.axis("off")

            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weighs),vmax=max(edge_weighs)))
            sm.set_array([])
            cbar=fig.colorbar(sm, ax=ax, orientation= 'vertical', fraction = 0.046, pad=0.04)
            cbar.set_label('Edge Weight', rotation=270, labelpad= 15)

            regular_line = plt.line2D([0],[0], color='blue', linewidth=2, label='Regular Edge')
            traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
            start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Start Node')
            end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='End Node')
            legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
            legend.get_frame().set_alpha(0.8)

            plt.tight_layout()
            st.pyplot(fig)

            