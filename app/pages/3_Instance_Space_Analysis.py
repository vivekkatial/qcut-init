import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import time
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Instance Space Analysis", page_icon="📈",     layout="wide",
    initial_sidebar_state="expanded",)

st.markdown("# Instance Space Analysis")
st.sidebar.header("Select ISA Experiment")

def plot_facetted_features(d_coords, d_features):
    """
    Creates a single facetted plot with multiple facets where each facet is a scatter plot of z_1 vs z_2,
    colored by the values of a different feature from d_features, excluding 'Source' and 'Evolution' columns.

    :param d_coords: A DataFrame containing the z_1 and z_2 coordinates.
    :param d_features: A DataFrame containing the features.
    """
    # Merge the coordinates and features dataframes on the 'Row' column
    merged_df = pd.merge(d_coords, d_features, on='Row')

    # Remove 'Source' and 'Evolution' columns if they exist
    if 'Source' in merged_df.columns:
        merged_df.drop(columns='Source', inplace=True)
    if 'Evolution' in merged_df.columns:
        merged_df.drop(columns='Evolution', inplace=True)

    # Prepare the long-form DataFrame for facetting
    melted_df = merged_df.melt(id_vars=['Row', 'z_1', 'z_2'], var_name='Feature', value_name='Value')

    # Create the facetted plot using Plotly Express
    fig = px.scatter(
        melted_df,
        x='z_1',
        y='z_2',
        color='Value',
        facet_col='Feature',
        facet_col_wrap=3,  # Adjust the number of columns per row
        color_continuous_scale='Magma',
        title='Facetted Plot of z_1 vs z_2 Colored by Feature Values',
        labels={'Value': 'Feature Value'},
        height=3000,  # Adjust the height as necessary
    )

    # Update the layout if needed
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='closest'
    )
    
    # Update the axis titles for each facet
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].title.text = ''
    
    # Update the facet titles to be more readable
    for annotation in fig.layout.annotations:
        feature_name = annotation.text.split('=')[1].replace('_', ' ')
        annotation.text = feature_name

    return fig


def plot_evolution_chart(d_coords, d_bounds):
    # Find the unique values in the 'Evolution' column
    evolutions = d_coords['Source'].unique()

    # Use Plotly Express to get a discrete color sequence
    color_sequence = px.colors.qualitative.Plotly

    # Create a color map for each evolution step to a color
    color_map = {evolution: color_sequence[i % len(color_sequence)] for i, evolution in enumerate(evolutions)}

    # Create a list to hold all the scatter traces
    scatter_traces = []

    # Create a scatter trace for each evolution step
    for evolution in evolutions:
        evolution_df = d_coords[d_coords['Source'] == evolution]
        scatter_traces.append(go.Scatter(
            x=evolution_df['z_1'],
            y=evolution_df['z_2'],
            mode='markers',
            marker=dict(size=5),  # Smaller circle size
            name=evolution,  # Use the evolution step as the name
            marker_color=color_map[evolution]  # Use the color from the color map
        ))

    # Polygon for the bounds
    bounds_trace = go.Scatter(
        x=d_bounds['z_1'],
        y=d_bounds['z_2'],
        mode='lines',
        fill='toself',
        fillcolor='rgba(0,0,0,0)',  # Transparent fill
        line=dict(color='red'),  # Red outline
        name='Bounds'
    )

    # Combine the plots
    data = scatter_traces + [bounds_trace]

    # Define layout options
    layout = go.Layout(
        xaxis=dict(title='z_1', showline = False), #removes X-axis line),
        yaxis=dict(title='z_2', showline = False), #removes Y-axis line),
        hovermode='closest',
        legend=dict(title='Source'),
        margin=dict(l=0, r=0, t=0, b=0)  # Minimize margin to use space efficiently
    )

    # Create the figure with data and layout
    fig = go.Figure(data=data, layout=layout)

    # Make chart square if desired
    # fig.update_xaxes(scaleanchor="y", scaleratio=1)

    # Return the figure
    return fig

# Get the data for experiments based on directories in `data/external/`

data_dir = "data/external/ISA"
experiments = os.listdir(data_dir)
experiment = st.sidebar.selectbox("Select ISA Experiment", experiments)

# List out the files in the selected experiment
experiment_dir = os.path.join(data_dir, experiment)

# Read in the coordinates.csv file for the experiment 
d_coords = pd.read_csv(os.path.join(experiment_dir, "coordinates.csv"))
# Extract the source based on whats after the hash `742a761d5ca8400d86cd72e76811737d_three_regular_graph`
d_coords['Source'] = d_coords['Row'].str.extract(r'_(\w+)$')
d_coords['Source'] = d_coords['Source'].str.title().str.replace('_', ' ')

# If the source contains Evolution keep the str else fill with Original
d_coords['Evolution'] = np.where(d_coords['Source'].str.contains('Evolution'), d_coords['Source'], 'Original')

# Read the SVM table
d_svm = pd.read_csv(os.path.join(experiment_dir, "svm_table.csv"))

# Read the bounds file
d_bounds = pd.read_csv(os.path.join(experiment_dir, "bounds_prunned.csv"))
d_features = pd.read_csv(os.path.join(experiment_dir, "feature_process.csv"))

# Display the SVM table
st.subheader("SVM")
st.dataframe(d_svm[["Row","Probability_of_good", "CV_model_accuracy", "CV_model_precision"]])



col1, col2 = st.columns(2)

with col1:
    st.subheader("Evolution")
    fig = plot_evolution_chart(d_coords, d_bounds)
    st.plotly_chart(fig, use_container_width=True)


with col2:
    st.subheader("Best Algorithm")


# Features
st.subheader("Features")
fig = plot_facetted_features(d_coords, d_features)
st.plotly_chart(fig, use_container_width=True)