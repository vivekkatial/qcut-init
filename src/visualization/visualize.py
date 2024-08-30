import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# Set the default template to Plotly
pio.templates.default = 'simple_white' # Set the default template to Plotly

def plot_feature_distribution(d_features, d_coords, feature):
    """ Plot the distribution of a feature in the instance space.

    Args:
        d_features (pd.DataFrame): The feature data.
        d_coords (pd.DataFrame): The coordinates data.
        feature (str): The feature to plot.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Join the feature data with the coordinates
    d_feature_coords = pd.merge(d_coords, d_features, on='Row')
    # Plot the feature distribution
    fig = px.scatter(d_feature_coords, x='z_1', y='z_2', color=feature, hover_data=['Row'], color_continuous_scale='plasma')
    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # remove the colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title=''))
    # update the title to be the feature (in title case - remove _ and capitalize)
    fig.update_layout(title_text=feature.replace("_", " ").title())

    fig.update_xaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    fig.update_yaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
            )
    
    # Make the title bold and larger
    fig.update_layout(title_font=dict(size=20, family='Courier', color='black'))
    
    return fig

def plot_performance_distribution(d_algorithm, d_coords, algorithm, binary_scale=False):
    """ Plot the distribution of an algorithm in the instance space.

    Args:
        d_algorithm (pd.DataFrame): The algorithm data.
        d_coords (pd.DataFrame): The coordinates data.
        algorithm (str): The algorithm to plot.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Join the algorithm data with the coordinates
    d_algorithm_coords = pd.merge(d_coords, d_algorithm, on='Row')
    # Plot the algorithm distribution (if binary_scale is True, use a binary color scale)
    if binary_scale:
        # convert algorithm to str (if 1 its GOOD and if 0 its BAD)
        d_algorithm_coords[algorithm] = d_algorithm_coords[algorithm].astype(str)
        d_algorithm_coords[algorithm] = np.where(d_algorithm_coords[algorithm] == '1', 'GOOD', 'BAD')
        
        fig = px.scatter(d_algorithm_coords, x='z_1', y='z_2', color=algorithm, hover_data=['Row'], color_discrete_map={'GOOD': 'blue', 'BAD': 'orange'})
    else:
        fig = px.scatter(d_algorithm_coords, x='z_1', y='z_2', color=algorithm, hover_data=['Row'], color_continuous_scale='plasma')
    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # remove the colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title=''))
    # If the algo is 'fixed_angle_constant' then replace with just CONSTANT
    if algorithm == 'fixed_angles_constant':
        algorithm = 'CONSTANT'
        
    # remove the colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title=''))
    # remove legend title
    fig.update_layout(legend_title_text='')
    # Adjust the legend markers size
    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            font=dict(size=12),
        )
    )

    # update the title to be the feature (in title case - remove _ and capitalize)
    fig.update_layout(title_text=algorithm.replace("_", " ").upper())

    fig.update_xaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    fig.update_yaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)
    
    return fig

def plot_best_algorithm(d_portfolio, d_coords):
    """ Plot the best algorithm in the instance space.

    Args:
        d_portfolio (pd.DataFrame): The portfolio data.
        d_coords (pd.DataFrame): The coordinates data.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Join the portfolio data with the coordinates
    d_portfolio_coords = pd.merge(d_coords, d_portfolio, on='Row')
    # Convert algos to str
    d_portfolio_coords['Best_Algorithm'] = d_portfolio_coords['Best_Algorithm'].astype(str)
    # Make the algorithms title case
    d_portfolio_coords['Best_Algorithm'] = d_portfolio_coords['Best_Algorithm'].str.replace('_', ' ').str.upper()
    # Plot the best algorithm distribution
    fig = px.scatter(d_portfolio_coords, x='z_1', y='z_2', color='Best_Algorithm', hover_data=['Row'])
    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # remove the legend title
    fig.update_layout(legend_title_text='')

    # Adjust the legend markers size
    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            font=dict(size=12),
        )
    )


    # update the title to be the feature (in title case - remove _ and capitalize)
    fig.update_layout(title_text='Best Algorithm')

    fig.update_xaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    fig.update_yaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)
    
    return fig

def plot_svm_selection_single_algo(d_coords, d_svm_selection, algorithm, experiment_dir, show_footprints=False):
    """ Plot the SVM selection for a single algorithm in the instance space.

    Args:
        d_coords (pd.DataFrame): The coordinates data.
        d_svm_selection (pd.DataFrame): The SVM selection data.
        algorithm (str): The algorithm to plot.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Join the SVM selection data with the coordinates
    d_svm_selection_coords = pd.merge(d_coords, d_svm_selection, on='Row')
    # Select the column based on the algorithm
    d_svm_selection_coords = d_svm_selection_coords[['Row', algorithm, 'z_1', 'z_2']]
    # Convert the algorithm to str (if 1 its GOOD and if 0 its BAD)
    d_svm_selection_coords[algorithm] = d_svm_selection_coords[algorithm].astype(str)
    d_svm_selection_coords[algorithm] = np.where(d_svm_selection_coords[algorithm] == '1', 'GOOD', 'BAD')
    
    # Plot the SVM selection distribution
    fig = px.scatter(d_svm_selection_coords, x='z_1', y='z_2', color=algorithm, hover_data=['Row'], color_discrete_map={'GOOD': 'blue', 'BAD': 'orange'})

    # Add polygons for the bounds
    if show_footprints:
        try:
            d_footprint_algo = pd.read_csv(os.path.join(experiment_dir, f"footprint_{algorithm}_good.csv"))
            def plot_polygon(data, fig, name):
                if not data.empty:
                    # Close the polygon by appending the first row to the end
                    data = pd.concat([data, data.iloc[[0]]])
                    fig.add_trace(go.Scatter(
                        x=data['z_1'], 
                        y=data['z_2'], 
                        mode='lines', 
                        fill='toself', 
                        fillcolor='rgba(0,0,255,0.1)', 
                        line=dict(color='black', width=0), 
                        name=name,
                        showlegend=False
                    ))
                else:
                    st.warning(f"No footprint data found for {algorithm}.")

            # Split the dataset into separate polygons based on NaN rows
            polygons = []
            current_polygon = []
            for index, row in d_footprint_algo.iterrows():
                if pd.isna(row['z_1']) or pd.isna(row['z_2']):
                    if current_polygon:
                        polygons.append(pd.DataFrame(current_polygon))
                        current_polygon = []
                else:
                    current_polygon.append(row)

            # Add the last polygon if any
            if current_polygon:
                polygons.append(pd.DataFrame(current_polygon))

            # Plot each polygon
            for i, polygon in enumerate(polygons):
                plot_polygon(polygon, fig, f'Footprint {i + 1}')
        except FileNotFoundError:
            st.warning(f"Footprint file for {algorithm} not found.")

    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # Remove the legend title   
    fig.update_layout(legend_title_text='')

    # Adjust the legend markers size
    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            font=dict(size=12),
        )
    )
    
    # update the title to be the feature (in title case - remove _ and capitalize)
    fig.update_layout(title_text=f'SVM Selection for {algorithm.replace("_", " ").upper()}')

    fig.update_xaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    fig.update_yaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)
    
    return fig

def plot_svm_selector(d_coords, d_svm_preds, svm_table, experiment_dir, show_footprints=False):
    """ Plot the SVM selection for all algorithms in the instance space.

    Args:
        d_coords (pd.DataFrame): The coordinates data.
        d_svm_preds (pd.DataFrame): The SVM selection data.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Create a dictionary for quick precision lookup
    precision_dict = svm_table.set_index('Row')['CV_model_precision'].to_dict()
    
    algorithms = d_svm_preds.columns[1:].tolist()
    def select_algorithm(row):
        selected = [alg for alg in algorithms if row[alg] == 1]
        
        if len(selected) == 0:
            return np.nan  # No algorithm selected
        if len(selected) == 1:
            return selected[0]
        
        # More than one selected, pick the one with the highest precision
        selected_precisions = {alg: precision_dict.get(alg, -1) for alg in selected}
        return max(selected_precisions, key=selected_precisions.get)

    # Apply the function to create the new column
    d_svm_preds['selected_algorithm'] = d_svm_preds.apply(select_algorithm, axis=1)
    # Join on coords
    d_svm_selection_coords = pd.merge(d_coords, d_svm_preds, on='Row')
    # Convert algorithms to CAPITAL and remove _
    d_svm_selection_coords['selected_algorithm'] = d_svm_selection_coords['selected_algorithm'].str.replace('_', ' ').str.upper()
    # For NaN values, set to 'NO ALGORITHM'
    d_svm_selection_coords['selected_algorithm'] = d_svm_selection_coords['selected_algorithm'].fillna('NO ALGORITHM')

    color_sequence = px.colors.qualitative.G10
    

    # Map algorithms to colors with opacity for polygons
    unique_algorithms = d_svm_selection_coords['selected_algorithm'].unique()
    algorithm_colors = {alg: color_sequence[i % len(color_sequence)] for i, alg in enumerate(unique_algorithms)}
    algorithm_fill_colors = {alg: f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.3)' for alg, c in algorithm_colors.items()}

    # Plot the SVM selection distribution
    fig = px.scatter(d_svm_selection_coords, x='z_1', y='z_2', color='selected_algorithm', hover_data=['Row'],
                     color_discrete_map=algorithm_colors)
    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    # Check if the footprints should be shown
    if show_footprints:
        # Read footprint file for each algorithm
        for algorithm in algorithms:
            # check if the file exists
            if os.path.exists(os.path.join(experiment_dir, f"footprint_{algorithm}_best.csv")):
                    
                d_footprint_algo = pd.read_csv(os.path.join(experiment_dir, f"footprint_{algorithm}_best.csv"))
                def plot_polygon(data, fig, name, fill_color):
                    if not data.empty:
                        # Close the polygon by appending the first row to the end
                        data = pd.concat([data, data.iloc[[0]]])
                        fig.add_trace(go.Scatter(
                            x=data['z_1'], 
                            y=data['z_2'], 
                            mode='lines', 
                            fill='toself',
                            fillcolor=fill_color,
                            line=dict(color='black', width=0),
                            name=name,
                            showlegend=False
                        ))
                    else:
                        st.warning(f"No footprint data found for {algorithm}.")

                # Split the dataset into separate polygons based on NaN rows
                polygons = []
                current_polygon = []
                for index, row in d_footprint_algo.iterrows():
                    if pd.isna(row['z_1']) or pd.isna(row['z_2']):
                        if current_polygon:
                            polygons.append(pd.DataFrame(current_polygon))
                            current_polygon = []
                    else:
                        current_polygon.append(row)

                # Add the last polygon if any
                if current_polygon:
                    polygons.append(pd.DataFrame(current_polygon))

                # Plot each polygon
                for i, polygon in enumerate(polygons):
                    fill_color = algorithm_fill_colors.get(algorithm.upper().replace('_', ' '), 'rgba(0, 0, 0, 0.1)')
                    plot_polygon(polygon, fig, f'Footprint {i + 1}', fill_color)
            else:
                st.warning(f"Footprint file for {algorithm} not found.")
                continue
    else:
        pass

    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # Remove the legend title   
    fig.update_layout(legend_title_text='')

    # Adjust the legend markers size
    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            font=dict(size=12),
        )
    )

    fig.update_xaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    fig.update_yaxes(showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True)

    return fig

def plot_source_distribution(d_coords, d_bounds, bounds=False):
    """ Plot the distribution of the source in the instance space.

    Args:
        d_coords (pd.DataFrame): The coordinates data.
        bounds (bool): Whether to plot the bounds.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure.
    """

    # Join the source data with the coordinates
    d_source_coords = d_coords

    color_discrete_map = px.colors.qualitative.G10
    source_colors = {source: color_discrete_map[i % len(color_discrete_map)] for i, source in enumerate(d_source_coords['Source'].unique())}

    # Plot the source distribution
    fig = px.scatter(d_source_coords, x='z_1', y='z_2', color='Source', hover_data=['Row'], color_discrete_map=source_colors)
    
    # Make the chart square
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Make the marker size smaller
    fig.update_traces(marker=dict(size=3))
    # Make the axis latex style (z_1 and z_2)
    fig.update_layout(
        xaxis_title='z<sub>1</sub>',
        yaxis_title='z<sub>2</sub>'
    )
    # Remove the colorbar title
    fig.update_layout(coloraxis_colorbar=dict(title=''))
    # Remove the legend title
    fig.update_layout(legend_title_text='')

    # Adjust the legend markers size
    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            font=dict(size=12),
        )
    )

    # update the title to be the feature (in title case - remove _ and capitalize)
    fig.update_layout(title_text='Source')

    # Ensure all four sides of the plot have black lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig