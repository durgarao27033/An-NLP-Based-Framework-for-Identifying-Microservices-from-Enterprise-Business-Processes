# An NLP-Based Framework for Identifying Microservices from Enterprise Business Processes

This repository contains the implementation, datasets, experiments, clustering algorithms, and evaluation system developed for the project:

**"An NLP-Based Framework for Identifying Microservices from Enterprise Business Processes"**

---

## 🧠 Overview

Microservices architecture is widely used in modern software development because it improves scalability, flexibility, and maintainability. However, identifying suitable microservice boundaries from existing business processes is a challenging task.

Traditional approaches often depend on manual analysis, predefined service boundaries, or structured business process models. These approaches may require significant human effort and may not efficiently identify relationships between different business activities.

This project proposes an NLP-based framework that automatically analyzes business process data and identifies groups of related activities that can serve as candidate microservices.

The framework integrates:

- Natural Language Processing (NLP) for text preprocessing
- TF-IDF for textual feature extraction
- Truncated SVD for dimensionality reduction
- K-Means clustering for grouping related activities
- DBSCAN clustering for density-based grouping and noise detection
- Silhouette Score and Davies–Bouldin Index for cluster evaluation
- Automatic cluster naming based on important keywords
- Interactive visualization using a Streamlit dashboard

The system aims to simplify microservice identification by grouping semantically related business activities and providing an understandable visualization of the clustering results.

---

## 📁 Repository Structure

```text
📦 Microservice-Identification/
│
├── data/
│   ├── cargo_usecases.json
│   ├── trading_usecases.json
│   └── retail_dataset.csv
│
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   └── evaluation.py
│
├── results/
│   ├── clustering_results.csv
│   └── evaluation_results.json
│
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

*Note: This is a suggested README structure. Adjust filenames and folders to match your actual project directory.*

---

## 🔧 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>

cd Microservice-Identification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Technologies Used

- Python 3.13
- Natural Language Processing (NLP)
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Truncated SVD
- K-Means Clustering
- DBSCAN Clustering
- Matplotlib
- Plotly
- Streamlit

---

## ⚙️ Methodology

### 1️⃣ Dataset Collection and Loading

The system accepts business process use cases and activity-related datasets for identifying potential microservice boundaries.

The datasets contain descriptions of business activities that are analyzed to discover relationships between related operations.

Examples of business activities include:

- View Tracking
- Book Cargo
- Change Cargo Destination
- Route Cargo
- Handle Cargo Events

Pandas is used to load and organize the datasets for further processing.

### 2️⃣ NLP-Based Text Preprocessing

The collected textual data is cleaned and standardized using Natural Language Processing techniques.

Preprocessing operations include:

- Lowercase conversion
- Tokenization
- Stop-word removal
- Punctuation removal
- Text normalization
- Lemmatization (if enabled in the implementation)

This process reduces unnecessary textual information and prepares the data for feature extraction.

### 3️⃣ Feature Extraction Using TF-IDF

After preprocessing, the textual descriptions are converted into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF).

TF-IDF assigns weights to words based on their importance within a particular document compared with the entire dataset.

Features include:

- Term frequency calculation
- Inverse document frequency calculation
- Textual feature generation
- Numerical vector representation

These vectors allow the clustering algorithms to identify similarities between business activities.

### 4️⃣ Dimensionality Reduction Using Truncated SVD

TF-IDF can generate high-dimensional sparse feature matrices.

To reduce the dimensionality of these representations, the system applies Truncated Singular Value Decomposition (Truncated SVD).

Benefits include:

- Reduces feature dimensionality
- Creates a compact representation
- Helps manage high-dimensional textual data
- Retains important latent information

The reduced feature representations are passed to the clustering algorithms.

### 5️⃣ K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to group similar business activities into a predefined number of clusters.

The algorithm works by:

- Selecting the number of clusters (K)
- Initializing cluster centroids
- Assigning activities to the nearest centroid
- Updating centroids based on assigned activities
- Repeating until convergence

The resulting clusters represent groups of related business activities that may form candidate microservices.

### 6️⃣ DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies groups based on the density of data points.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified in advance.

Features include:

- Density-based grouping
- Identification of closely related activities
- Detection of noise and outliers
- Automatic identification of dense clusters

Activities that do not belong to a sufficiently dense cluster may be classified as noise (cluster label `-1`).

### 7️⃣ Cluster Evaluation

The framework evaluates the quality of generated clusters using internal clustering evaluation metrics.

**Silhouette Score**

Measures how closely an activity belongs to its assigned cluster compared with other clusters.

- Range: -1 to +1
- Higher values generally indicate better-separated and more cohesive clusters.

**Davies–Bouldin Index**

Measures the similarity between clusters based on their internal spread and separation.

- Lower values indicate better clustering.
- A value closer to zero is generally preferred.

These metrics help compare the clustering performance of K-Means and DBSCAN.

### 8️⃣ Cluster Naming and Microservice Identification

After clustering, the system analyzes important terms within each cluster to generate meaningful cluster names.

Related business activities are grouped together and presented as candidate microservice boundaries.

For example:

- Cargo Booking
- Cargo Tracking
- Cargo Routing

The generated groups help developers understand which business activities may belong to the same microservice.

### 9️⃣ Interactive Streamlit Dashboard

The system provides an interactive dashboard developed using Streamlit.

Dashboard features include:

- Dataset upload and preview
- Text preprocessing
- Feature extraction
- K-Means clustering results
- DBSCAN clustering results
- Cluster assignments
- Silhouette Score
- Davies–Bouldin Index
- PCA-based 2D cluster visualization
- Cluster comparison and analysis

The dashboard makes the results easier to understand without requiring users to examine the underlying code.

---

## 📊 Evaluation Metrics

The framework uses the following evaluation measures:

| Metric | Purpose |
|---|---|
| Silhouette Score | Measures cluster cohesion and separation |
| Davies–Bouldin Index | Measures similarity between clusters |
| Number of Clusters | Identifies the number of generated groups |
| Noise Points | Measures unassigned points detected by DBSCAN |

Since the framework performs unsupervised clustering, internal clustering metrics are used rather than conventional classification accuracy, unless ground-truth labels are available.

---

## 🧪 Experimental Results

The system compares the performance of K-Means and DBSCAN clustering algorithms.

**K-Means Clustering**

- Groups business activities into a predefined number of clusters.
- Assigns every data point to a cluster.
- Produces candidate microservice groups.

**DBSCAN Clustering**

- Identifies dense groups of related activities.
- Does not require a predefined number of clusters.
- Can identify noise and outlier activities.

**Evaluation**

The clustering algorithms are compared using Silhouette Score and Davies–Bouldin Index.

Actual experimental metric values should be added here after running the final version of the project.

---

## 🚀 Running the System

Run the main processing pipeline:

```bash
python main.py
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

*Use the actual dashboard filename from your project if it is different.*

---

## 📈 Outputs

The system generates:

- Preprocessed business activity descriptions
- TF-IDF feature representations
- Reduced feature representations using Truncated SVD
- K-Means cluster assignments
- DBSCAN cluster assignments
- Automatically generated cluster names
- Silhouette Score
- Davies–Bouldin Index
- PCA-based cluster visualizations
- Candidate microservice identification results

---

## 🎯 Contributions

- NLP-based business activity preprocessing.
- TF-IDF-based textual feature extraction.
- Dimensionality reduction using Truncated SVD.
- Implementation of K-Means and DBSCAN clustering algorithms.
- Comparative analysis of clustering techniques.
- Evaluation using Silhouette Score and Davies–Bouldin Index.
- Automatic cluster naming based on important keywords.
- Interactive Streamlit dashboard for visualizing microservice identification results.

---

## 🔮 Future Enhancements

- Integration of transformer-based sentence embeddings.
- Semantic similarity-based activity matching.
- Knowledge graph-based business relationship analysis.
- Automatic recommendation of optimal cluster parameters.
- Integration with BPMN business process models.
- Validation using additional real-world business process datasets.
- Further improvement of microservice boundary identification.
