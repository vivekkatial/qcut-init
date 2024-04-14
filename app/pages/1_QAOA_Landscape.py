import streamlit as st
import streamlit.components.v1 as components
import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Custom imports
from src.features.build_features import get_graph_features
from src.features.calculate_optimal_angles import calculate_optimal_angles_for_triangle_free_graph, find_min_gamma_beta

st.set_page_config(
    page_title="QAOA",
    layout="wide",
    page_icon=":sunny:",
    initial_sidebar_state="expanded"
)
st.title("QAOA Landscape")

@st.cache_data
def load_data(json_file_path):
    """Load data from the JSON file."""
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data

def get_number_of_triangles(graph):
    """Calculate the number of triangles in a graph."""
    return sum(nx.triangles(graph).values()) // 3

def adjacency_matrix_to_graph(adj_matrix):
    """Convert an adjacency matrix back to a networkx graph."""
    # Convert the list of lists (adjacency matrix) to a numpy array
    np_matrix = np.array(adj_matrix)
    # Use from_numpy_array since from_numpy_matrix might be deprecated or causing issues
    return nx.from_numpy_array(np_matrix)

def display_graph_info(graph_name, graph):
    """Display a table with information about the graph, including various graph features."""

    features = get_graph_features(graph)
    selected_features = [
        'radius', 
        'minimum_degree', 
        'minimum_dominating_set', 
        'regular', 
        'planar', 
        'average_distance', 
        'laplacian_largest_eigenvalue', 
        'group_size', 
        'number_of_edges', 
        'number_of_orbits'
    ]


    # Calculate number of triangles
    number_of_triangles = get_number_of_triangles(graph)

    # Create DataFrames for the top entries and the selected features.
    top_info_df = pd.DataFrame({
        'Feature': ['Source', 'Number of Triangles'],
        'Value': [graph_name, number_of_triangles]
    })
    
    selected_info_df = pd.DataFrame({
        'Feature': [feature.replace('_', ' ').capitalize() for feature in selected_features],
        'Value': [features[feature] for feature in selected_features]
    })

    # Concatenate the top entries DataFrame with the selected features DataFrame.
    info_df = pd.concat([top_info_df, selected_info_df], ignore_index=True)

    # Display the DataFrame using st.table which automatically bolds the header
    st.table(info_df)

def plot_landscape_plotly(landscape_data, dimension='2d', show_colorbar=True, source=None, point=None, **kwargs):
    # Define custom colors
    colors = [
        "#0000A3",  # Dark blue
        "#7282ee",  # Lighter blue
        "#B0C7F9",  # Very light blue
        "#e2d9d4",  # Light beige
        "#F6BFA6",  # Light red
        "#de4d4d",  # Darker red
    ]

    # Convert to np.array
    beta = np.array(landscape_data['beta']) / np.pi  # Normalize and convert beta
    gamma = np.array(landscape_data['gamma']) / np.pi  # Normalize and convert gamma
    obj_vals = np.array(landscape_data['obj_vals'])
    
    # Create a figure
    fig = go.Figure()

    if dimension == '2d':
        # Add the heatmap for 2D representation
        fig.add_trace(go.Heatmap(
            z=obj_vals.T,
            x=beta,
            y=gamma,
            coloraxis="coloraxis"
        ))
    elif dimension == '3d':
        # Add the surface plot for 3D representation
        fig.add_trace(go.Surface(
            z=obj_vals.T,
            x=beta,
            y=gamma,
            colorscale=colors
        ))

    fig.update_layout(
        title=source,
        xaxis_title=r'$\beta$',
        yaxis_title=r'$\gamma$',
        coloraxis=dict(colorscale=colors) if dimension == '2d' else None,
    )

    # Show color bar if needed and if in 2D mode
    if show_colorbar and dimension == '2d':
        fig.update_layout(coloraxis_colorbar=dict(title="Objective Values"))

    # Adjust the layout for 3D if needed
    if dimension == '3d':
        fig.update_layout(scene=dict(
            xaxis_title='beta',
            yaxis_title='gamma',
            zaxis_title='Objective Value',
        ))

    # Add a point to the plot if provided for 2d
    if point is not None and dimension == '2d':
        # Scale the point to the normalized range
        point = [point[0] / np.pi, point[1] / np.pi]

        # Extract the color from the kwargs or use red as default
        color = kwargs.get('color', 'red')

        # Marker for the optimal angles as stars

        fig.add_trace(go.Scatter
        (
            x=[point[0]],
            y=[point[1]],
            mode='markers',
            marker=dict(color=color, size=10),
            marker_symbol='star',
            name='Optimal Angles',
        ))


    return fig

def plot_graph(graph):
    """Plot a networkx graph using a planar layout if possible, otherwise use a spring layout."""
    plt.figure(figsize=(9, 4.5), facecolor='white')  # Adjusted size to take up less space
    
    try:
        pos = nx.planar_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', font_weight='bold', font_color='black')
    except nx.NetworkXException:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', font_weight='bold', font_color='black')
    
    st.pyplot(plt)

def app():

    # List directories in data/external/qaoa-landscape
    directories = [f for f in os.listdir("data/external/qaoa-landscape") if os.path.isdir(os.path.join("data/external/qaoa-landscape", f))]
    # Number of layers (1 to 15)
    p_values = [f"p={i}" for i in range(1, 16)]

    # Sidebar
    st.sidebar.title("QAOA")
    st.sidebar.write("This app visualizes the landscape of the Quantum Approximate Optimization Algorithm (QAOA).")
    st.sidebar.write("Select a graph type from the dropdown to visualize the landscape and graph information.")

    # Add Instance Selection to sidebar
    st.sidebar.subheader("Instance Selection")
    selected_graph_type = st.sidebar.selectbox("Select Graph Type:", sorted(list(set(directories))))

    # Load the JSON data
    final_layer_data_fp = f"data/external/qaoa-landscape/{selected_graph_type}/optimization_results_15_layers.json"
    final_layer_data = load_data(final_layer_data_fp)
    

    st.subheader("Instance Information")    

    st.write("""
    For $QAOA_{p=1}$ applied to MaxCut on any triangle-free graph, the optimal angles maximizing $C$ (or, equivalently, maximizing $C/m$) satisfy the following:
    """)

    st.latex(r'''
        (\gamma^*, \beta^*) := \left( \arctan \frac{1}{\sqrt{D - 1}}, \frac{\pi}{8} \right)
    ''')

    
    if st.button('Click for more details on how to find optimal angles.'):
        st.write("""
            for $D \geq 2$ (i.e., no other optimal pair $( \gamma, \\beta )$ exists with $0 < \gamma \leq \gamma^*$ or $0 < \\beta \leq \\beta^*$).

            All optimal angles are periodic in $\gamma, \\beta$ with periodicity depending on $D$:

            - If $D$ is even, there is a second independent pair of optimal angles given by $(-\gamma^*, -\\beta^*)$, independent in the sense that all optimal angles are generated from these two pairs as

                $$
                (\gamma^* + a\pi, \\beta^* + b\\frac{\pi}{2}), \quad (-\gamma^* + c\pi, -\\beta^* + d\\frac{\pi}{2}), \quad a, b, c, d \in \mathbb{Z}.
                $$

            - Else if $D$ is odd, there are four independent pairs of optimal angles $(\gamma^*, \\beta^*)$, $(-\gamma^*, -\\beta^*)$, $(\pi - \gamma^*, \\beta^*)$, and $(\pi + \gamma^*, -\\beta^*)$, and all optimal angles are generated from one of these pairs, denoted $(\gamma', \\beta')$, as

                $$
                (\gamma' + a2\pi, \\beta' + b\\frac{\pi}{2}), \quad a, b \in \mathbb{Z}.
                $$

            - For an arbitrary triangle-free graph with maximum vertex degree $D_G$ and minimum vertex degree $D_{\min}$, the smallest positive optimal angles $\gamma^*, \\beta^*$ in $[0, \pi/2]$ satisfy

            $$
            \\arctan \\frac{1}{\sqrt{D_G - 1}} \leq \gamma^* \leq \\arctan \\frac{1}{\sqrt{D_{\min} - 1}}, \quad \\beta^* = \\frac{\pi}{8}.
            $$

            Given such a pair, the angles $(-\gamma^*, -\\beta^*)$ are also optimal, and both pairs are $2\pi$-periodic in the first argument and $\pi/2$-periodic in the second, with respect to optimality.

            The theorem implies the smallest optimal angles are $(\pi/4, \pi/8)$ and $(0.6155, \pi/8)$ for 2-regular and 3-regular triangle-free graphs.

            Cited from: [Zhou, L., Wang, S., Choi, S., Pichler, H., & Lukin, M. D. (2018). Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices. *Physical Review A*, 97(2), 022304.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.022304)
            """)
                    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Instance**")
        # Find and plot the graph for the selected graph type
        st.write(final_layer_data["graph_type"])
    
        if final_layer_data["graph_type"] == selected_graph_type:
            graph = adjacency_matrix_to_graph(final_layer_data["graph"]["adjacency_matrix"])
            plot_graph(graph)

    with col2:
        # Show table
        display_graph_info(selected_graph_type, graph)

    
    # Facet chart for the optimization results
    st.subheader("Optimization Results")
    # Show keys
    st.write("Keys:", final_layer_data.keys())
    d_optimization_results = final_layer_data["optimization_results"]
    # Conver to DataFrame
    d_optimization_results = pd.DataFrame(d_optimization_results)
    # Convert layers to string if they are not already
    d_optimization_results['layer'] = d_optimization_results['layer'].astype(str)

    # Create subplot titles
    subplot_titles = [f'Layer = {i+1}' for i in range(1, 16)]
    # Change last subplot title to be ""
    subplot_titles[-1] = ""

    # Create a subplot figure with 4 rows and 4 columns
    fig = make_subplots(rows=4, cols=4, subplot_titles=subplot_titles, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.1)
    


    # Assuming we have two optimizers for the sake of this example
    optimizers = d_optimization_results['optimizer'].unique()

    # Loop over each layer and each optimizer and add a trace for each combination to the appropriate subplot
    for i, layer in enumerate(sorted(d_optimization_results['layer'].unique(), key=lambda x: int(x)), start=1):
        layer_df = d_optimization_results[d_optimization_results['layer'] == layer]
        row, col = divmod(i - 1, 4)
        row += 1  # Adjust row to 1-indexed
        col += 1  # Adjust col to 1-indexed
        for optimizer in optimizers:
            optimizer_df = layer_df[layer_df['optimizer'] == optimizer]
            fig.add_trace(
                go.Scatter(
                    x=optimizer_df['iteration'],
                    y=optimizer_df['energy'],
                    mode='lines',
                    name=optimizer,
                    showlegend=(row == 1 and col == 1)  # Only show legend for the first subplot
                ),
                row=row,
                col=col
            )

            # Add a trace for the exact ground state energy (dashed line) -- make same color on each plot
            fig.add_trace(
                go.Scatter(
                    x=[0, optimizer_df['iteration'].max()],
                    y=[final_layer_data["exact_ground_state_energy"], final_layer_data["exact_ground_state_energy"]],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Exact Ground State Energy',
                    showlegend=False,
                    line_color='pink'
                ),
                row=row,
                col=col
            )

            # Extend the y-axis range to include the exact ground state energy and a bit more
            fig.update_yaxes(range=[final_layer_data["exact_ground_state_energy"] *1.5, optimizer_df['energy'].max()], row=row, col=col)

    # Adjust the layout for each optimizer to have the same color across all plots
    colors = iter(px.colors.qualitative.Plotly)  # Use Plotly's qualitative color scale
    for optimizer in optimizers:
        fig.update_traces(selector=dict(name=optimizer), line=dict(color=next(colors)))

    # Update layout
    fig.update_layout(
        title_text='Energy by Layer and Optimizer',
        height=1000, 
        width=1200,  
        showlegend=True
    )

    # Remove the empty last subplot in the 4th row
    for i in range(1, 5):
        fig.update_xaxes(row=4, col=i, visible=(i != 4))
        fig.update_yaxes(row=4, col=i, visible=(i != 4))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    
    st.subheader("Landscape Visualization")

    # Create tabs for the landscape plots based on the number of layers (and add latex around the p values)
    tabs = st.tabs([f"$p={i}$" for i in range(1, 16)])

    # Loop through each one
    for i, tab in enumerate(tabs):
        with tab:
            # Print what instance we are looking at and the number of layers
            st.write(f"Instance: {selected_graph_type}")
            st.write(f"Number of Layers: {i + 1}")
            if i + 1 == 1:
                "first layer"
            else:
                # Load the landscape data for the selected graph type and number of layers
                landscape_data_fp = f"data/external/qaoa-landscape/{selected_graph_type}/optimization_results_{i + 1}_layers.json"
                landscape_data = load_data(landscape_data_fp)
                landscape_data = landscape_data["landscape_data"]
                # Add radio button to select dimension (for each layer)
                dimension = st.radio("Select Dimension:", ["2d", "3d"], index=0, key=i)


                # plot the landscape
                fig = plot_landscape_plotly(landscape_data, dimension=dimension, source=selected_graph_type)
                st.plotly_chart(fig)
                



app()