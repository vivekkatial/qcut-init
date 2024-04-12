import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import time
import numpy as np
import pandas as pd
import os
import scipy.io

st.set_page_config(page_title="Instance Space Analysis", page_icon="📈",     layout="wide",
    initial_sidebar_state="expanded",)

st.markdown("# Instance Space Analysis")
st.sidebar.header("Select ISA Experiment")

def csv_to_latex(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert the dataframe to LaTeX format
    latex_str = df.to_latex(index=False, escape=False)
    
    return latex_str


def plot_facetted_data(d_coords, d_data, exclude_columns=None, binary_scale=False):
    """
    Creates a single facetted plot with multiple facets where each facet is a scatter plot of z_1 vs z_2,
    colored by the values of different data columns from d_data, with an option to exclude specific columns.

    :param d_coords: DataFrame containing the z_1 and z_2 coordinates.
    :param d_data: DataFrame containing the data to be plotted alongside the coordinates.
    :param exclude_columns: List of column names to be excluded from the plotting.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Merge the coordinates and data dataframes on the 'Row' column
    merged_df = pd.merge(d_coords, d_data, on='Row')

    if 'Source' in merged_df.columns:
        merged_df.drop(columns='Source', inplace=True)
    if 'Evolution' in merged_df.columns:
        merged_df.drop(columns='Evolution', inplace=True)

    # Remove specified columns
    for column in exclude_columns:
        if column in merged_df.columns:
            merged_df.drop(columns=column, inplace=True)

    # Prepare the long-form DataFrame for facetting
    melted_df = merged_df.melt(id_vars=['Row', 'z_1', 'z_2'], var_name='DataColumn', value_name='Value')

    # Calculate how many columns to wrap the facets
    num_columns = len(melted_df['DataColumn'].unique())

    # Update the color scale to be PER chart
    colorscale = [(0, "#F8F71A"), (0.3, "#A1C840"), (0.6, "#1CAADF"), (0.9, "#4742E6"), (1, "#3F29B2")]
    
        

    # Create the facetted plot using Plotly Express
    fig = px.scatter(
        melted_df,
        x='z_1',
        y='z_2',
        color='Value',
        facet_col='DataColumn',
        facet_col_wrap=3,  # Adjust the number of columns per row as necessary
        color_continuous_scale=[(0, "#F8F71A"), (0.3, "#A1C840"), (0.6, "#1CAADF"), (0.9, "#4742E6"), (1, "#3F29B2")],
        title='',
        labels={'Value': 'Data Value'},
        # if more than 6 columns use smaller height
        height=1600 if num_columns > 6 else 800,
    )

    # Update the layout if needed
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='closest'
    )
    
    # Update the facet titles to be more readable
    for annotation in fig.layout.annotations:
        feature_name = annotation.text.split('=')[1].replace('_', ' ')
        # Add <b> to split the title into two lines
        feature_name = f'<b>{feature_name}</b>'
        annotation.text = feature_name
        # Make color of title black
        annotation.font.color = 'black'
    
    # Update each facet to have its own colorscale
    for i, trace in enumerate(fig.data):
        trace.marker.coloraxis = f'coloraxis{i+1}'
    
    for fr in fig.frames:
    # update each of the traces in each of the animation frames
        for i, t in enumerate(fr.data):
            t.update(coloraxis=f"coloraxis{i+1}")
    
    
    coloraxis_config = {}
    for i in range(1, num_columns + 1):
        axis_name = f'coloraxis{i}'

        # If only 2 values in the column, use a different color scale
        if len(melted_df[melted_df['DataColumn'] == melted_df['DataColumn'].unique()[i - 1]]['Value'].unique()) == 2:
            colorscale = [(0, "#F8F71A"), (1, "#3F29B2")]
            if binary_scale:
                colorscale = [(0, "orange"), (1, "blue")]
        else:
            colorscale = [(0, "#F8F71A"), (0.3, "#A1C840"), (0.6, "#1CAADF"), (0.9, "#4742E6"), (1, "#3F29B2")]

        coloraxis_config[axis_name] = {
            "colorbar": {"showticklabels": False},  # Hide colorbar tick labels
            "showscale": False,  # Hide the colorbar itself
            "colorscale": colorscale
        }

    # Apply the dynamic color axis configuration to the figure layout
    fig.update_layout(**coloraxis_config)
    # Apply simple theme
    fig.update_layout(template="simple_white")


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
        margin=dict(l=0, r=0, t=0, b=0),  # Minimize margin to use space efficiently
        height=350,  # Adjust the height as necessary
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
# Read in the metadata file for the experiment
d_metadata = pd.read_csv(os.path.join(experiment_dir, "metadata.csv"))
# select only the columns that are needed
d_metadata = d_metadata[["Instances", "Source"]]
# Merge the metadata with the coordinates ("Row" = "Instances")
d_coords = pd.merge(d_coords, d_metadata, left_on="Row", right_on="Instances")
# Drop the "Instances" column
d_coords.drop(columns="Instances", inplace=True)

# If the source contains Evolution keep the str else fill with Original
d_coords['Evolution'] = np.where(d_coords['Source'].str.contains('Evolution'), d_coords['Source'], 'Original')

# Read the SVM table
d_svm = pd.read_csv(os.path.join(experiment_dir, "svm_table.csv"))

# Read the bounds file
d_bounds = pd.read_csv(os.path.join(experiment_dir, "bounds_prunned.csv"))
if experiment in ["qaoa-param-inform-pub", "qaoa-classical-opts-init"]:
    d_bounds = pd.read_csv(os.path.join(experiment_dir, "bounds.csv"))

# Read the feature table
d_features = pd.read_csv(os.path.join(experiment_dir, "feature_process.csv"))
# Read the algorithm table
d_algorithm_raw = pd.read_csv(os.path.join(experiment_dir, "algorithm_raw.csv"))
d_algorithm_process = pd.read_csv(os.path.join(experiment_dir, "algorithm_process.csv"))

# Read the SVM table and best algorithm
d_svm_preds = pd.read_csv(os.path.join(experiment_dir, "algorithm_svm.csv"))
d_svm_selection = pd.read_csv(os.path.join(experiment_dir, "portfolio_svm.csv"))
d_best_algo = pd.read_csv(os.path.join(experiment_dir, "portfolio.csv"))

# Read the 


mat = scipy.io.loadmat(os.path.join(experiment_dir, "model.mat"))
algos = mat["data"]["algolabels"]
algos = np.array([item for sublist in algos.flat for item in sublist.flat])

# Update d_svm_selection "Best_Algorithm" column with the algorithm names
d_svm_selection["Best_Algorithm"] = algos[d_svm_selection["Best_Algorithm"]-1]
# Update d_best_algo "Best_Algorithm" column with the algorithm names
d_best_algo["Best_Algorithm"] = algos[d_best_algo["Best_Algorithm"]-1]


### DISPLAY THE DATA ###

# Display the SVM table
st.subheader("SVM")

d_svm_display = d_svm[["Row","Probability_of_good", "CV_model_accuracy", "CV_model_precision"]].style.format({
    "Probability_of_good": "{:.2%}",
    # Just 2 d.p. for the accuracy and precision
    "CV_model_accuracy": "{:.2f}",
    "CV_model_precision": "{:.2f}"    
})

st.dataframe(d_svm_display)


col1, col2 = st.columns(2)

with col1:
    # Make tabs for Source and Evolution
    tabs = st.tabs(["Source Distribution", "Evolved Distribution"])
    with tabs[0]:
        st.subheader("Source Distribution")
        fig = plot_evolution_chart(d_coords, d_bounds)
        st.plotly_chart(fig, use_container_width=True)
    


with col2:
    st.subheader("Transformation")
    dataframe = pd.read_csv(f"data/external/ISA/{experiment}/projection_matrix.csv")
    dataframe = dataframe.T

    # Filter first row
    projection_matrix = dataframe.iloc[1:]
    # Round all values to 3 decimal places
    projection_matrix = projection_matrix.round(2).astype(str)
    
    projection_matrix = projection_matrix.to_latex(index=False, escape=False, column_format='', header=False). \
        replace('\\begin{tabular}', '\\begin{bmatrix}').replace('\\end{tabular}', '\\end{bmatrix}'). \
        replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', ''). \
        replace('\\{c\\}', '')

    # Extract index column
    sel_feats = dataframe.reset_index()
    # Remove first reow
    sel_feats = sel_feats.iloc[1:]
    # Select only the first column
    sel_feats = sel_feats.iloc[:, 0]
    # Add `\text{}` around each value
    sel_feats = sel_feats.apply(lambda x: f'\\text{{{x}}}')
    # Replace _ with space 
    sel_feats = sel_feats.str.replace('_', ' ')
    

    # Convert to LaTeX
    sel_feats = sel_feats.to_latex(index=False, escape=False, column_format='', header=False). \
        replace('\\begin{tabular}', '\\begin{bmatrix}').replace('\\end{tabular}', '\\end{bmatrix}'). \
        replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', ''). \
        replace('\\{c\\}', '')


    st.latex(
        r"""

            \begin{{bmatrix}}
            Z_1 \\
            Z_2
            \end{{bmatrix}} = {projection_matrix}^\top
            {sel_feats}
        """.format(projection_matrix=projection_matrix, sel_feats=sel_feats)
    )
        




# Create 4 tabs
tabs = st.tabs(["Feature Distribution", "Performance Distribution", "Best Algorithm", "SVM Model Prediction", "SVM Selection", "MATILDA Information"])

# tabs = ["Feature Distribution", "Performance Distribution", "", "SVM Model Prediction", "SVM Selection"]

with tabs[0]:
    st.subheader("Feature Distribution")
    fig = plot_facetted_data(d_coords, d_features)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Performance Distribution")
    # Add radio button to flick between raw and processed algorithm data
    algorithm_data = st.radio("Algorithm Data", ["Raw", "Processed"])
    if algorithm_data == "Raw":
        d_algorithm = d_algorithm_raw
    else:
        d_algorithm = d_algorithm_process

    fig = plot_facetted_data(d_coords, d_algorithm)
    st.plotly_chart(fig, use_container_width=True)

    # Analyse the binary algorithm data
    st.subheader("Binary Distribution")
    d_algorithm_binary = pd.read_csv(os.path.join(experiment_dir, "algorithm_bin.csv"))
    fig = plot_facetted_data(d_coords, d_algorithm_binary, binary_scale=True)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Best Algorithm")
    fig = plot_facetted_data(d_coords, d_best_algo)
    st.plotly_chart(fig, use_container_width=True)


with tabs[3]:
    st.subheader("SVM Model Prediction")
    fig = plot_facetted_data(d_coords, d_svm_preds, binary_scale=True)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("SVM Selection")
    fig = plot_facetted_data(d_coords, d_svm_selection)
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Model Information")
    # Read options.json
    options = pd.read_json(os.path.join(experiment_dir, "options.json"), typ='series')
    # Fully unnest the options
    st.write(options)

    