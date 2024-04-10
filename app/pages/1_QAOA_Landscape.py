import streamlit as st
import streamlit.components.v1 as components
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    # Load the JSON data
    json_file_path = "data/interim/landscape_p_2_data.json"
    data = load_data(json_file_path)
    convergence_data_path = "data/interim/convergence_data_p_2.json"
    convergence_data = load_data(convergence_data_path)

    # Sidebar
    st.sidebar.title("QAOA")
    st.sidebar.write("This app visualizes the landscape of the Quantum Approximate Optimization Algorithm (QAOA).")
    st.sidebar.write("Select a graph type from the dropdown to visualize the landscape and graph information.")

    # Add Instance Selection to sidebar
    st.sidebar.subheader("Instance Selection")
    graph_types = list(set(item["graph_type"] for item in data))
    selected_graph_type = st.sidebar.selectbox("Select Graph Type:", sorted(list(set(item["graph_type"] for item in data))))

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
        for item in data:
            if item["graph_type"] == selected_graph_type:
                graph = adjacency_matrix_to_graph(item["graph"]["adjacency_matrix"])
                plot_graph(graph)
                break  # Assuming you only want to plot the first matching graph

    with col2:
        # Show table
        display_graph_info(selected_graph_type, graph)                
    
    st.subheader("Landscape Visualization")

    tabs = st.tabs(["$p=1$", "$p=2$", "$\gamma^{*}_{p=2}, \\beta^{*}_{p=2}$"])

    with tabs[0]:

        dimension = st.radio("Select the dimension for the landscape plot:", ('2d', '3d'))
        # Ensure that the landscape_data is correctly retrieved from the selected item
        landscape_data = None
        for item in data:
            if item["graph_type"] == selected_graph_type:
                landscape_data = item["landscape_data"]
                break  # Assuming you only want to use the first matching item's landscape data
        
        if landscape_data:
            st.write("Landscape for instance.")
            if get_number_of_triangles(graph) == 0:
                st.write("The graph is triangle-free. The optimal angles are calculated analytically.")
                point = calculate_optimal_angles_for_triangle_free_graph(graph)
                # Write Latex for Optimal Angles extracted from point[0] and point[1] divided by pi (rounded to 4 decimal places)
                st.latex(r'''
                    (\gamma^*, \beta^*) := \left( {:.4f}\pi, {:.4f}\pi \right)
                '''.format(point[1]/np.pi, point[0]/np.pi))
                plotly_fig = plot_landscape_plotly(landscape_data, dimension=dimension, point=point)
            else:
                st.write("The graph is **not** triangle-free. The numerical results are used to fix the angles.")
                # If the graph is not triangle-free, we will fix the angles based on numerically computed results
                point = find_min_gamma_beta(landscape_data)
                # Write Latex for Optimal Angles extracted from point[0] and point[1] (rounded to 4 decimal places)
                st.latex(r'''
                    (\gamma^*, \beta^*) := \left( {:.4f}\pi, {:.4f}\pi \right)
                '''.format(point[1]/np.pi, point[0]/np.pi))
                

                
                
                plotly_fig = plot_landscape_plotly(landscape_data, dimension=dimension, point=point, color='lightgreen')
            st.components.v1.html(plotly_fig.to_html(include_mathjax='cdn'),height=500, width=800)
        else:
            st.write("No landscape data available for the selected graph type.")


    with tabs[1]:
        # Read in notes from notes/landscape_p_02.md
        with open("app/notes/landscape_p_02.md", "r") as file:
            landscape_notes_p2 = file.read()

        st.markdown(landscape_notes_p2)
        # Get optimal angles for p=1
        point_p1 = point

        dimension_p_2 = st.radio("Select the dimension for the p=2 landscape plot:", ('2d', '3d'))

        st.write("Optimal angles for $p=1$ are $(\gamma^*, \\beta^*) = ({:.4f}\pi, {:.4f}\pi)$".format(point_p1[1]/np.pi, point_p1[0]/np.pi))

        col1, col2 = st.columns([1, 1])
        with col1:
            # Plot landscape for p=2
            landscape_data_p2 = None
            for item in data:
                if item["graph_type"] == selected_graph_type:
                    landscape_data_p2 = item["landscape_storage_p_2"]
                    plotly_fig = plot_landscape_plotly(landscape_data_p2, dimension=dimension_p_2)
                    st.components.v1.html(plotly_fig.to_html(include_mathjax='cdn'),height=500, width=500)

        with col2:
            # Convert the convergence data to a DataFrame
            for item in convergence_data:
                if item["graph_type"] == selected_graph_type:
                    convergence_df = pd.DataFrame(item["p_2_results"])
                    fig = go.Figure()

                    min_energy = min(convergence_df['energy'].min(), convergence_df['exact_min_energy'].min())
                    # Extend the y-axis a bit lower than the minimum energy value
                    y_axis_min = min_energy - (abs(min_energy) * 0.25)  # Extend by 5% of the absolute minimum value

                    # Add a line for each optimizer_name
                    for optimizer_name in convergence_df['optimizer_name'].unique():
                        df_sub = convergence_df[convergence_df['optimizer_name'] == optimizer_name]
                        fig.add_trace(go.Scatter(x=df_sub['eval_count'], y=df_sub['energy'],
                                                mode='lines', name=optimizer_name))

                    # Add a dashed line for the exact_min_energy across all eval_count values
                    # Assuming exact_min_energy is constant across all rows, if not adjust accordingly
                    fig.add_trace(go.Scatter(x=convergence_df['eval_count'], y=convergence_df['exact_min_energy'],
                                            mode='lines', name='Exact Min Energy', line=dict(dash='dash')))

                    # Update layout to add titles and adjust legend
                    fig.update_layout(title='Energy vs. Eval Count by Optimizer Name',
                                    xaxis_title='Eval Count',
                                    yaxis_title='Energy',
                                    yaxis=dict(range=[y_axis_min, convergence_df['energy'].max()]),
                                    legend_title='Optimizer Name')

                    # Display the figure in Streamlit
                    st.plotly_chart(fig)

    with tabs[2]:
        for item in convergence_data:
            if item["graph_type"] == selected_graph_type:
                convergence_df = pd.DataFrame(item["p_2_results"])
                
                # Define the angle types you want to plot
                angle_types = ['gamma_1', 'gamma_2', 'beta_1', 'beta_2']
                
                for angle in angle_types:
                    fig = go.Figure()
                    
                    # Add a line for each optimizer_name for the current angle type
                    for optimizer_name in convergence_df['optimizer_name'].unique():
                        df_sub = convergence_df[convergence_df['optimizer_name'] == optimizer_name]
                        fig.add_trace(go.Scatter(x=df_sub['eval_count'], y=df_sub[angle],
                                                mode='lines', name=f'{optimizer_name} {angle}'))
                    
                    # Update layout to add titles and adjust legend
                    fig.update_layout(title=f'{angle} vs. Eval Count by Optimizer Name',
                                    xaxis_title='Eval Count',
                                    yaxis_title=angle.replace('_', ' ').title(),
                                    legend_title='Optimizer Name')
                    
                    # Display the figure in Streamlit
                    st.plotly_chart(fig)

                    




app()