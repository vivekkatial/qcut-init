import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(page_title="QAOA Parameter Evolution", page_icon="📈", layout="wide")

@st.cache_data
def load_data(csv_file_path):
    """Load data from the CSV file."""
    data = pd.read_csv(csv_file_path)
    return data
    
# Title
st.title("QAOA Parameter Evolution")

## Sidebar: Configuration
st.sidebar.header("Configuration")
num_layers = st.sidebar.slider("Number of Layers", 1, 20, 5, help="Adjust the number of QAOA layers.")
# Number of qubits should be based on the distinct number of vertices in the dataset
num_qubits_list = [6, 8, 10, 12]
# Default select all qubits are selected (use multiselect)
num_qubits = st.sidebar.multiselect("Number of Qubits", num_qubits_list, num_qubits_list, help="Select the number of qubits.")


# select based on the number of layers
data_file_path = f"data/processed/QAOA_{num_layers}_layers.csv"
data = load_data(data_file_path)


# Column layout for dataset overview and distribution information
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(f"Dataset Overview")
    st.markdown(f"**Layers:** {num_layers}")
    st.markdown(f"**Qubits:** {num_qubits}")
    st.markdown(f"**Rows:** {data.shape[0]}")
    st.markdown(f"**Columns:** {data.shape[1]}")

with col2:
    st.subheader("Instance Class Distribution")
    class_dist = data["static_feature_instance_class"].value_counts().to_frame()
    st.dataframe(class_dist)

with col3:
    st.subheader("Qubits Distribution")
    qubits_dist = data["static_feature_number_of_vertices"].value_counts().to_frame()
    st.dataframe(qubits_dist)


st.subheader("Data Preview")
st.dataframe(data.head())

# Assuming beta and gamma values are prefixed with 'beta' and 'gamma' in your columns
# Generate lists of columns for betas and gammas
beta_columns = [col for col in data.columns if 'beta' in col]
gamma_columns = [col for col in data.columns if 'gamma' in col]

# Sidebar: Layer selection for plotting
st.sidebar.header("Plot Settings")
selected_layer = st.sidebar.selectbox("Select Layer", options=range(1, num_layers+1), index=num_layers-1)

# Create a regex pattern for the beta and gamma columns of the selected layer
beta_pattern = f"dynamic_QAOA_L{selected_layer}_beta_.*"
gamma_pattern = f"dynamic_QAOA_L{selected_layer}_gamma_.*"

# Filter columns based on the pattern
beta_columns = data.filter(regex=beta_pattern).columns
gamma_columns = data.filter(regex=gamma_pattern).columns

# Plotting
st.header(f"Parameter Distributions for Layer {selected_layer}")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Beta Distributions")
    for col in beta_columns:
        st.write(f"Distribution for {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax, kde=True)
        st.pyplot(fig)

with col2:
    st.subheader("Gamma Distributions")
    for col in gamma_columns:
        st.write(f"Distribution for {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax, kde=True)
        st.pyplot(fig)