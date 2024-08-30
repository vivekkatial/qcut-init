import streamlit as st
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import numpy as np
import pandas as pd
import os
import shutil
import scipy.io
from src.visualization.visualize import *

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

# Get the data for experiments based on directories in `data/external/`

data_dir = "data/external/ISA"
experiments = os.listdir(data_dir)
# Remove .DS_Store file if it exists
if ".DS_Store" in experiments:
    experiments.remove(".DS_Store")

experiment = st.sidebar.selectbox("Select ISA Experiment", experiments)

# List out the files in the selected experiment
experiment_dir = os.path.join(data_dir, experiment)

# Read in the coordinates.csv file for the experiment 
d_coords = pd.read_csv(os.path.join(experiment_dir, "coordinates.csv"))

# if the experiment is INFORMS-Revision-evolved then change source column to be either "Original"


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
d_features_raw = pd.read_csv(os.path.join(experiment_dir, "feature_raw.csv"))
# Read the algorithm table
d_algorithm_raw = pd.read_csv(os.path.join(experiment_dir, "algorithm_raw.csv"))
d_algorithm_process = pd.read_csv(os.path.join(experiment_dir, "algorithm_process.csv"))
d_algorithm_binary = pd.read_csv(os.path.join(experiment_dir, "algorithm_bin.csv"))

# Read the SVM table and best algorithm
d_svm_preds = pd.read_csv(os.path.join(experiment_dir, "algorithm_svm.csv"))
d_svm_selection = pd.read_csv(os.path.join(experiment_dir, "portfolio_svm.csv"))
# Find the best algorithm for each row


# Create a new df for best_algorithm from d_algorithm_raw
# The column with the minimum value is the best algorithm (excluding the first column)
d_best_algo = pd.DataFrame()
d_best_algo['Best_Algorithm'] = d_algorithm_raw.iloc[:, 1:].idxmin(axis=1)
d_best_algo['Row'] = d_algorithm_raw['Row']
# Reshuflle the columns
d_best_algo = d_best_algo[['Row', 'Best_Algorithm']]



mat = scipy.io.loadmat(os.path.join(experiment_dir, "model.mat"))
algos = mat["data"]["algolabels"]
algos = np.array([item for sublist in algos.flat for item in sublist.flat])
# Flatten algos
algos = [item for sublist in algos for item in sublist]




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

# Function to simulate progress for demonstration purposes
def update_progress_bar(progress_bar, progress_text, iteration, total):
    progress_percentage = int((iteration / total) * 100)
    progress_bar.progress(progress_percentage)
    progress_text.text(f"Progress: {progress_percentage}%")

# Download all the plots as a zip file
download_all = st.button("Download All Plots")
if download_all:
    st.write("Downloading all plots...")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Create a temporary directory to store the plots
    os.makedirs("temp", exist_ok=True)

    # Common layout updates
    def update_layout_fonts(fig):
        fig.update_layout(
            font=dict(size=20),  # Increase overall font size
            xaxis=dict(
                title_font_size=24,  # Increase axis label font size
                tickfont=dict(size=18)  # Increase tick font size
            ),
            yaxis=dict(
                title_font_size=24,  # Increase axis label font size
                tickfont=dict(size=18)  # Increase tick font size
            ),
            legend=dict(font=dict(size=18))  # Increase legend font size
        )
        return fig

    # Make the source distribution plot
    source_fig = plot_source_distribution(d_coords, d_bounds)
    source_fig.update_layout(width=600, height=600)
    source_fig.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
    source_fig.update_layout(title_font_size=26)
    source_fig = update_layout_fonts(source_fig)
    source_fig.write_image(os.path.join("temp", "source_distribution.png"), scale=3)
    
    # Update progress bar
    update_progress_bar(progress_bar, progress_text, 1, 14)

    # Create a vector for all features
    features = d_features.columns[1:]
    for i, feature in enumerate(features):
        fig_feat = plot_feature_distribution(d_features, d_coords, feature)
        fig_feat_raw = plot_feature_distribution(d_features_raw, d_coords, feature)
        
        feat_copy = feature
        fig_feat.update_layout(width=600, height=600)
        if feature == "ratio_of_two_largest_laplacian_eigenvaleus":
            feature = "2_largest_eigenvalue_ratio"
        elif feature == "laplacian_second_largest_eigenvalue":
            feature = "laplacian_2nd_largest_eigenvalue"
        feature_title = feature.replace("_", " ").upper()
        fig_feat.update_layout(title=f'{feature_title}')
        fig_feat.update_layout(title_font_size=26)
        fig_feat = update_layout_fonts(fig_feat)
        fig_feat.write_image(os.path.join("temp", f"feature_{feat_copy}_distribution.png"), scale=3)

        update_progress_bar(progress_bar, progress_text, 2 + (i+1)/len(features), 14)
        fig_feat_raw.update_layout(width=600, height=600)
        fig_feat_raw.update_layout(title=f'<b>{feature_title}</b>')
        fig_feat_raw.update_layout(title_font_size=26)
        fig_feat_raw = update_layout_fonts(fig_feat_raw)
        fig_feat_raw.write_image(os.path.join("temp", f"feature_{feat_copy}_raw_distribution.png"), scale=3)

    # Update progress bar
    update_progress_bar(progress_bar, progress_text, 3, 14)

    # Create a vector for all algorithms
    algorithms = d_algorithm_raw.columns[1:]
    for i, algorithm in enumerate(algorithms):
        fig_algo = plot_performance_distribution(d_algorithm_process, d_coords, algorithm)
        fig_algo.update_layout(width=600, height=600)
        fig_algo.update_layout(title_font_size=26)
        fig_algo = update_layout_fonts(fig_algo)
        fig_algo.write_image(os.path.join("temp", f"performance_{algorithm}_distribution.png"), scale=3)

        # Do the raw performance
        fig_algo_raw = plot_performance_distribution(d_algorithm_raw, d_coords, algorithm)
        fig_algo_raw.update_layout(width=600, height=600)
        fig_algo_raw.update_layout(title_font_size=26)
        fig_algo_raw = update_layout_fonts(fig_algo_raw)
        fig_algo_raw.write_image(os.path.join("temp", f"performance_{algorithm}_raw_distribution.png"), scale=3)

        # Do the binary performance
        fig_algo_bin = plot_performance_distribution(d_algorithm_binary, d_coords, algorithm, binary_scale=True)
        fig_algo_bin.update_layout(width=600, height=600)
        fig_algo_bin.update_layout(title_font_size=26)
        fig_algo_bin = update_layout_fonts(fig_algo_bin)
        fig_algo_bin.write_image(os.path.join("temp", f"performance_{algorithm}_binary_distribution.png"), scale=3)

        update_progress_bar(progress_bar, progress_text, 4 + (i+1)/len(algorithms), 14)

    # Create the best algorithm plot
    fig_best_algo = plot_best_algorithm(d_coords, d_best_algo)
    fig_best_algo.update_layout(width=600, height=600)
    fig_best_algo.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
    fig_best_algo.update_layout(title_font_size=26)
    fig_best_algo = update_layout_fonts(fig_best_algo)
    fig_best_algo.write_image(os.path.join("temp", "best_algorithm_distribution.png"), scale=3)
    
    # Update progress bar
    update_progress_bar(progress_bar, progress_text, 12, 14)

    # Create the SVM selection plot
    for algorithm in algos:
        fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm, experiment_dir, show_footprints=True)
        fig_svm.update_layout(width=600, height=600)
        fig_svm.update_layout(title_font_size=26)
        fig_svm = update_layout_fonts(fig_svm)
        fig_svm.write_image(os.path.join("temp", f"svm_selection_{algorithm}.png"), scale=3)
        
        # Create the SVM selector plot without footprints
        fig_svm_no_bounds = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm, experiment_dir, show_footprints=False)
        fig_svm_no_bounds.update_layout(width=600, height=600)
        fig_svm_no_bounds.update_layout(title_font_size=26)
        fig_svm_no_bounds = update_layout_fonts(fig_svm_no_bounds)
        fig_svm_no_bounds.write_image(os.path.join("temp", f"svm_selection_{algorithm}_no_footprints.png"), scale=3)

        # Update progress bar
        update_progress_bar(progress_bar, progress_text, 13, 14)

    # Create the SVM selector plot
    fig_svm_selector = plot_svm_selector(d_coords, d_svm_preds, d_svm, experiment_dir=experiment_dir, show_footprints=True)
    fig_svm_selector.update_layout(width=600, height=600)
    fig_svm_selector.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
    fig_svm_selector.update_layout(title_font_size=26)
    fig_svm_selector = update_layout_fonts(fig_svm_selector)
    fig_svm_selector.write_image(os.path.join("temp", "svm_selection.png"), scale=3)

    # Create the SVM selector plot without footprints
    fig_svm_selector_no_bounds = plot_svm_selector(d_coords, d_svm_preds, d_svm, experiment_dir=experiment_dir, show_footprints=False)
    fig_svm_selector_no_bounds.update_layout(width=600, height=600)
    fig_svm_selector_no_bounds.update_layout(legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2))
    fig_svm_selector_no_bounds.update_layout(title_font_size=26)
    fig_svm_selector_no_bounds = update_layout_fonts(fig_svm_selector_no_bounds)
    fig_svm_selector_no_bounds.write_image(os.path.join("temp", "svm_selection_no_footprints.png"), scale=3)

    # Update progress bar
    update_progress_bar(progress_bar, progress_text, 14, 14)
    def callback():
        st.balloons()

    # Create a zip file with all the plots
    shutil.make_archive("temp", "zip", "temp")
    # Download the zip file

    with open("temp.zip", "rb") as f:            
        st.download_button(
            label="Download as Zip",
            data=f,
            file_name=f"{experiment_dir}.zip",
            mime="application/zip",
            key="callback",
        )

    st.write("Download complete.")
    progress_bar.progress(100)
    progress_text.text("Progress: 100%")
    # Delete the temporary directory
    shutil.rmtree("temp")


col1, col2 = st.columns(2)

with col1:
    # Make tabs for Source and Evolution
    st.subheader("Source Distribution")
    fig = plot_source_distribution(d_coords, d_bounds)
    st.plotly_chart(fig)


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

    # Select the feature to display (except the first column)
    feature = st.selectbox("Feature", d_features.columns[1:])
    # Radio button for raw or processed features
    feature_data = st.radio("Feature Data", ["processed", "raw"])
    if feature_data == "raw":
        d_features = pd.read_csv(os.path.join(experiment_dir, "feature_raw.csv"))
    else:
        d_features = pd.read_csv(os.path.join(experiment_dir, "feature_process.csv"))
        
    # if feature == "ratio_of_two_largest_laplacian_eigenvaleus":
    #         feature = "2_largest_eigenvalue_ratio"

    fig = plot_feature_distribution(d_features, d_coords, feature)
    st.plotly_chart(fig, use_container_width=True)

    feature_columns = d_features.columns[1:]


    # Determine the number of columns to fit all features in a single row
    num_features = len(feature_columns)
    cols = num_features
    rows = 1

    # Creating subplots for each feature column
    fig = make_subplots(rows=rows, cols=cols, subplot_titles='')

    for i, feature in enumerate(feature_columns):
        col = i + 1
        # clean feature title
        feature_title = feature.replace("_", " ").upper()
        # fig.update_xaxes(title_text=feature_title, row=1, col=col)
        fig.add_trace(
            go.Box(y=d_features[feature], name="", marker_color="black", fillcolor="lightgrey", line_color="black"),
            row=1, col=col)
        fig.update_yaxes(tickfont=dict(size=18), row=1, col=col)
        fig.update_xaxes(title_text=feature_title, row=1, col=col, title_font=dict(color="black", size=18))
        
        
    fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>', font=dict(size=24)))



    # Updating the layout for better visibility
    fig.update_layout(
        # title="Box-Whisker Plot of Selected Features",
        showlegend=False,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add a download button for the plot
    download_feature = st.button("Download Plot", key="feature_plot")
    # Download  the feature distribution plot
    if download_feature:
        st.write("Downloading feature distribution plot...")
        fig.write_image("feature_distribution.png", scale=3)
        st.download_button(
            label="Download as PNG",
            data="feature_distribution.png",
            file_name="feature_distribution.png",
            mime="image/png",
            key="feature_plot_download",
        )
        st.write("Download complete.")
    

with tabs[1]:
    st.subheader("Performance Distribution")
    # Select the algorithm to display
    algorithm = st.selectbox("Algorithm", d_algorithm_raw.columns[1:])

    # Add radio button to flick between raw and processed algorithm data
    algorithm_data = st.radio("Algorithm Data", ["raw", "processed"])
    if algorithm_data == "raw":
        d_algorithm = d_algorithm_raw
    else:
        d_algorithm = d_algorithm_process

    fig_algo_perf = plot_performance_distribution(d_algorithm, d_coords, algorithm)
    st.plotly_chart(fig_algo_perf, use_container_width=True)

    # Analyse the binary algorithm data
    st.subheader("Binary Distribution")
    fig_algo_bin = plot_performance_distribution(d_algorithm_binary, d_coords, algorithm, binary_scale=True)
    st.plotly_chart(fig_algo_bin, use_container_width=True)




with tabs[2]:
    st.subheader("Best Algorithm")
    # Display the best algorithm
    fig = plot_best_algorithm(d_coords, d_best_algo)
    st.plotly_chart(fig, use_container_width=True)

    # Add a download button for the plot
    download_best_algo = st.button("Download Plot", key="best_algo_plot")



with tabs[3]:
    st.subheader("SVM Selection")
    # Add selector for algorithm
    algorithm_svm = st.selectbox("Algorithm (SVM)", algos)
    # Add radio button to add polygons to the chart
    show_bounds_selector = st.checkbox("Show Footprints (SVM Selection)")

    if show_bounds_selector:
        fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds, algorithm_svm, experiment_dir, show_footprints=True)
    else:
        # Plot the SVM selection for the selected algorithm
        fig_svm = plot_svm_selection_single_algo(d_coords, d_svm_preds,algorithm_svm, experiment_dir)

    # fig = plot_facetted_data(d_coords, d_svm_selection)
    st.plotly_chart(fig_svm, use_container_width=True)



with tabs[4]:
    st.subheader("SVM Selection")
    show_bounds = st.checkbox("Show Footprints")

    fig_svm_selector = plot_svm_selector(d_coords, d_svm_preds, d_svm, experiment_dir=experiment_dir, show_footprints=show_bounds)
    st.plotly_chart(fig_svm_selector, use_container_width=True)

with tabs[5]:
    st.subheader("Model Information")
    # Read options.json
    with open(os.path.join(experiment_dir, "options.json")) as f:
        options = json.load(f)
    # Fully unnest the options
    st.json(options, expanded=False)

    