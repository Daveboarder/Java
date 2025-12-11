# server_app.py (runs on server)
import math
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import h5py
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from LIBSmethods import peak_intensity, voigt_fit, simple_sum, simple_voigt_fit, snv, movingMinimum
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from SpectraGenerator import create_spectra
import pandas as pd
import sqlite3
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed

@app.route('/')
def index():
    """Serve the index.html file"""
    return send_file(os.path.join(os.path.dirname(__file__),'index.html'))

@app.route('/api/load_file', methods=['POST'])
def load_file():
    """Load HDF5 file and return metadata"""
    # Validate request has JSON data
    if not request.json:
        return jsonify({'success': False, 'error': 'No JSON data received'}), 400
    
    # Validate file_path exists and is not empty
    file_path = request.json.get('file_path')
    if not file_path or not file_path.strip():
        return jsonify({'success': False, 'error': 'File path is required and cannot be empty'}), 400
    
    # Remove any whitespace
    file_path = file_path.strip()
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'File not found: {file_path}'}), 404
    
    # Check if it's actually a file (not a directory)
    if not os.path.isfile(file_path):
        return jsonify({'success': False, 'error': f'Path is not a file: {file_path}'}), 400
    
    try:
        with h5py.File(file_path, "r") as file:
            wavelength = file['measurements/Measurement_1/libs/calibration'][:]
            data = file['/measurements/Measurement_1/libs/data'][:]
            x_pos = file['measurements/Measurement_1/libs/metadata/X_pos'][:]
            y_pos = file['measurements/Measurement_1/libs/metadata/Y_pos'][:]
            x_len = file['measurements/Measurement_1/libs/metadata/x'][:]
            y_len = file['measurements/Measurement_1/libs/metadata/y'][:]
            x_step = file['measurements/Measurement_1/global_metadata/Width Spacing'][:]
            y_step = file['measurements/Measurement_1/global_metadata/Height Spacing'][:]
            max_data = np.max(data, axis=0).tolist()
            max_data_subtracted = movingMinimum(max_data, m=50, n=50).tolist()
            
            return jsonify({
                'success': True,
                'wavelength': wavelength.tolist(),
                'max_data': max_data,
                'max_data_subtracted': max_data_subtracted,
                'x_pos': x_pos.tolist(),
                'y_pos': y_pos.tolist(),
                'data_shape': data.shape,
                'x_len': x_len.tolist(),
                'y_len': y_len.tolist(),
                'x_step': x_step.tolist(),
                'y_step': y_step.tolist()
            })
    except KeyError as e:
        return jsonify({'success': False, 'error': f'Invalid HDF5 structure: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/calculate_pca', methods=['POST'])
def calculate_pca():
    """Calculate PCA for given parameters"""
    try:
        # Get data source from request (default to 'complete_spectra' for backward compatibility)
        data_source = request.json.get('data_source', 'complete_spectra')
        
        # If data source is selected_lines, get lines_data from request
        if data_source == 'selected_lines':
            lines_data = request.json.get('lines_data', {})
            selected_lines = request.json.get('selected_lines', [])
            print(f"DEBUG: data_source={data_source}, selected_lines={selected_lines}, lines_data keys={list(lines_data.keys()) if lines_data else []}")
            if len(selected_lines) < 3:
                return jsonify({'success': False, 'error': 'At least 3 spectral lines are required for PCA'}), 400
            
            if not lines_data or len(lines_data) == 0:
                return jsonify({'success': False, 'error': 'No line data provided'}), 400
            
            # Extract position data from first line (all should have same positions)
            first_line_name = selected_lines[0]
            if first_line_name not in lines_data:
                return jsonify({'success': False, 'error': f'Line data not found for: {first_line_name}'}), 400
            
            x_pos = np.array(lines_data[first_line_name]['x_pos'])
            y_pos = np.array(lines_data[first_line_name]['y_pos'])
            x_len = lines_data[first_line_name]['x_len']
            y_len = lines_data[first_line_name]['y_len']
            x_step = lines_data[first_line_name]['x_step']
            y_step = lines_data[first_line_name]['y_step']
            
            # Extract intensity arrays for each selected line
            intensity_arrays = []
            line_names = []
            
            for line_name in selected_lines:
                if line_name not in lines_data:
                    return jsonify({'success': False, 'error': f'Line data not found for: {line_name}'}), 400
                
                line_intensity = np.array(lines_data[line_name]['intensity'])
                intensity_arrays.append(line_intensity)
                line_names.append(line_name)
            
            # Stack intensities to create feature matrix: rows = samples, columns = lines
            if len(set(len(arr) for arr in intensity_arrays)) != 1:
                return jsonify({'success': False, 'error': 'All selected lines must have the same number of data points'}), 400
            
            data_for_pca = np.column_stack(intensity_arrays)
            print(f"PCA from selected lines: {len(selected_lines)} lines, {data_for_pca.shape[0]} samples")
            
            # Normalize the data (SNV normalization)
            data_snv = snv(data_for_pca)
            
            # Calculate PCA model
            n_components = min(5, len(selected_lines))
            pcaModel = PCA(n_components=n_components).fit(data_snv)
            pca_scores = pcaModel.transform(data_snv)
            loadings_wavelength = np.arange(len(selected_lines))
            
        else:
            # Default: use complete spectra - need file_path
            file_path = request.json.get('file_path')
            if not file_path:
                return jsonify({'success': False, 'error': 'file_path is required for complete_spectra data source'}), 400
            
            if not os.path.exists(file_path):
                return jsonify({'success': False, 'error': f'File not found: {file_path}'}), 404
            if not os.path.isfile(file_path):
                return jsonify({'success': False, 'error': f'Path is not a file: {file_path}'}), 400
            
            with h5py.File(file_path, "r") as file:
                wavelength = file['measurements/Measurement_1/libs/calibration'][:]
                data = file['/measurements/Measurement_1/libs/data'][:]
                x_pos = file['measurements/Measurement_1/libs/metadata/X_pos'][:]
                y_pos = file['measurements/Measurement_1/libs/metadata/Y_pos'][:]
                x_len = file['measurements/Measurement_1/libs/metadata/x'][:]
                y_len = file['measurements/Measurement_1/libs/metadata/y'][:]
                x_step = file['measurements/Measurement_1/global_metadata/Width Spacing'][:]
                y_step = file['measurements/Measurement_1/global_metadata/Height Spacing'][:]
                
                #normalize the data to standard normal distribution
                data_snv = snv(data)
                #calculate pca model
                pcaModel = PCA(n_components=5).fit(data_snv)
                # Transform data to get scores (PC values for each sample)
                pca_scores = pcaModel.transform(data_snv)
                loadings_wavelength = wavelength
        
        # Get clustering method from request (default to 'kmeans' for backward compatibility)
        clustering_method = request.json.get('clustering_method', 'kmeans')
        
        # Apply clustering based on selected method
        if clustering_method == 'meanshift':
            # Mean Shift clustering - prefers density
            # Estimate bandwidth automatically
            bandwidth = estimate_bandwidth(pca_scores[:,[0,1]], quantile=0.2, n_samples=min(500, len(pca_scores)))
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            cluster = meanshift.fit_predict(pca_scores[:,[0,1]])
            print(f"Mean Shift clustering: {cluster}, bandwidth: {bandwidth}, n_clusters: {len(np.unique(cluster))}")
        else:
            # Default to k-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42) #random state for reproducibility
            cluster = kmeans.fit_predict(pca_scores[:,[0,1]])
            print(f"K-means clustering: {cluster}")
        
        # Calculate k-means scree plot data (inertia for 1-10 clusters)
        # Use PCA scores for k-means clustering
        n_clusters_range = list(range(1, 11))
        inertias = []
        max_pc_for_scree = min(3, pca_scores.shape[1])
        for n_clusters in n_clusters_range:
            kmeans_scree = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
            kmeans_scree.fit(pca_scores[:, :max_pc_for_scree])  # Use available PC scores
            inertias.append(float(kmeans_scree.inertia_))
        
        # Get explained variance ratios
        explained_variance = pcaModel.explained_variance_ratio_
        
        # Prepare response data
        response_data = {
            'success': True,
            'wavelength': loadings_wavelength.tolist() if isinstance(loadings_wavelength, np.ndarray) else loadings_wavelength,
            'x_pos': x_pos.tolist() if isinstance(x_pos, np.ndarray) else x_pos,
            'y_pos': y_pos.tolist() if isinstance(y_pos, np.ndarray) else y_pos,
            'x_len': x_len.tolist() if isinstance(x_len, np.ndarray) else x_len,
            'y_len': y_len.tolist() if isinstance(y_len, np.ndarray) else y_len,
            'x_step': x_step.tolist() if isinstance(x_step, np.ndarray) else x_step,
            'y_step': y_step.tolist() if isinstance(y_step, np.ndarray) else y_step,
            'PC1_scores': pca_scores[:, 0].tolist(),  # Scores for plotting
            'explained_variance_PC1': float(explained_variance[0]),
            'kmeans_labels': cluster.tolist(),
            'kmeans_n_clusters': n_clusters_range,  # [1, 2, 3, ..., 10]
            'kmeans_inertias': inertias,  # Inertia values for each n_clusters
            'clustering_method': clustering_method,  # Return the method used
            'data_source': data_source  # Return the data source used
        }
        
        # Add PC2 and PC3 if available
        if pca_scores.shape[1] >= 2:
            response_data['PC2'] = pcaModel.components_[1].tolist()
            response_data['PC2_scores'] = pca_scores[:, 1].tolist()
            response_data['explained_variance_PC2'] = float(explained_variance[1])
        
        if pca_scores.shape[1] >= 3:
            response_data['PC3'] = pcaModel.components_[2].tolist()
            response_data['PC3_scores'] = pca_scores[:, 2].tolist()
            response_data['explained_variance_PC3'] = float(explained_variance[2])
        
        # Always include PC1 loadings
        response_data['PC1'] = pcaModel.components_[0].tolist()
        
        # If using selected lines, include line names for reference
        if data_source == 'selected_lines':
            response_data['selected_line_names'] = line_names if 'line_names' in locals() else []
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plot_kmeans_map', methods=['POST'])
def plot_kmeans_map():
    """
    Plot the k-means clustering result as a 2D map using x_pos and y_pos.
    """
    try:
        x_pos = np.array(request.json['x_pos'])
        y_pos = np.array(request.json['y_pos'])
        kmeans_labels = np.array(request.json['kmeans_labels'])
        x_len = request.json.get('x_len', [])
        y_len = request.json.get('y_len', [])
        x_step = request.json.get('x_step', [])
        y_step = request.json.get('y_step', [])

        # Validate arrays
        if len(x_pos) == 0 or len(y_pos) == 0 or len(kmeans_labels) == 0:
            return jsonify({'success': False, 'error': 'Empty arrays provided'}), 400
        
        # Make sure we have the correct number of points
        if not (len(x_pos) == len(y_pos) == len(kmeans_labels)):
            return jsonify({
                'success': False, 
                'error': f'Array lengths do not match: x_pos={len(x_pos)}, y_pos={len(y_pos)}, labels={len(kmeans_labels)}'
            }), 400

        # Convert step and length values safely
        try:
            x_step = int(x_step[0] if isinstance(x_step, (list, np.ndarray)) and len(x_step) > 0 else x_step) if x_step else 1
            y_step = int(y_step[0] if isinstance(y_step, (list, np.ndarray)) and len(y_step) > 0 else y_step) if y_step else 1
            x_len = int(np.max(x_len)) if isinstance(x_len, (list, np.ndarray)) and len(x_len) > 0 else int(x_len) if x_len else len(np.unique(x_pos))
            y_len = int(np.max(y_len)) if isinstance(y_len, (list, np.ndarray)) and len(y_len) > 0 else int(y_len) if y_len else len(np.unique(y_pos))
        except Exception as e:
            # Use defaults if conversion fails
            x_step = 1
            y_step = 1
            x_len = len(np.unique(x_pos))
            y_len = len(np.unique(y_pos))

        # Swap axes if needed (same logic as plot_intensity_map)
        if x_len < y_len:
            x_len, y_len = y_len, x_len
            x_step, y_step = y_step, x_step
            x_pos, y_pos = y_pos, x_pos

        # Calculate dimensions with reasonable defaults
        x_span = x_len*x_step
        y_span = y_len*y_step

        print(f"x_span: {x_span}, y_span: {y_span}")    

        if x_span > 0 and y_span > 0:
            real_aspect_ratio = y_span/x_span
        else:
            real_aspect_ratio = 1

        base_size = 800

        if real_aspect_ratio > 1.0:
            plot_height = base_size
            plot_width = int(plot_height/real_aspect_ratio)
        else:
            plot_width = base_size
            plot_height = int(plot_width*real_aspect_ratio)

        legend_width = 200
        margin_space = 150
        total_width = plot_width + legend_width + margin_space
        total_height = plot_height + margin_space

        # Debug print
        print(f"Plotting k-means map: {len(x_pos)} points, width={total_width}, height={total_height}")
        print(f"x_pos range: [{np.min(x_pos):.2f}, {np.max(x_pos):.2f}]")
        print(f"y_pos range: [{np.min(y_pos):.2f}, {np.max(y_pos):.2f}]")
        print(f"kmeans_labels range: [{np.min(kmeans_labels)}, {np.max(kmeans_labels)}]")

        # Get unique clusters and number of clusters
        unique_clusters = np.unique(kmeans_labels)
        n_clusters = len(unique_clusters)
        
        # Define discrete color palette
        discrete_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
            '#e5c494', '#b3b3b3', '#8dd3c7', '#ffffb3', '#bebada'
        ]
        
        # Create interactive Plotly scatter plot with separate trace for each cluster
        fig = go.Figure()
        
        # Convert to lists for easier manipulation
        x_pos_list = x_pos.tolist()
        y_pos_list = y_pos.tolist()
        labels_list = kmeans_labels.tolist()
        
        # Create a separate trace for each cluster
        for i, cluster_id in enumerate(sorted(unique_clusters)):
            # Find indices where this cluster appears
            cluster_mask = [label == cluster_id for label in labels_list]
            cluster_x = [x_pos_list[j] for j in range(len(x_pos_list)) if cluster_mask[j]]
            cluster_y = [y_pos_list[j] for j in range(len(y_pos_list)) if cluster_mask[j]]
            
            # Add trace for this cluster
            fig.add_trace(go.Scatter(
                x=cluster_x,
                y=cluster_y,
                mode='markers',
                name=f'Cluster {int(cluster_id)+1}',
                marker=dict(
                    size=3,
                    color=discrete_colors[i % len(discrete_colors)]
                    #line=dict(width=0.5, color='white')  # Optional: white border for clarity
                ),
                hovertemplate=f'Cluster {int(cluster_id)+1}<br>X: %{{x:.2f}} mm<br>Y: %{{y:.2f}} mm<extra></extra>',
                legendgroup='clusters',
                showlegend=True
            ))
        
        # Determine title based on clustering method if provided
        clustering_method = request.json.get('clustering_method', 'kmeans')
        title = 'Mean Shift Cluster Map' if clustering_method == 'meanshift' else 'K-Means Cluster Map'
        
        fig.update_layout(
            title=title,
            xaxis_title='X Position (mm)',
            yaxis_title='Y Position (mm)',
            template='plotly_white',       
            width=int(total_width),
            height=int(total_height),
            autosize=False,
            xaxis=dict(
                scaleanchor='y',
                scaleratio=1#real_aspect_ratio
            ),
            margin=dict(
                l=80,
                r=legend_width + 20,
                t=60,
                b=60
            ),
            hovermode='closest',
            legend=dict(
                title='Clusters',
                itemsizing='constant',
                itemwidth=30
            )
        )
        
        return jsonify({
            'success': True,
            'plot': fig.to_dict()
        })
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in plot_kmeans_map: {error_msg}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/calculate_intensity', methods=['POST'])
def calculate_intensity():
    """Calculate peak intensity for given parameters"""
    file_path = request.json['file_path']
    b1_w = request.json['b1_w']
    b2_w = request.json['b2_w']

    if b1_w > b2_w:
        b1_w, b2_w = b2_w, b1_w
    
    try:
        with h5py.File(file_path, "r") as file:
            wavelength = file['measurements/Measurement_1/libs/calibration'][:]
            data = file['/measurements/Measurement_1/libs/data'][:]
            x_pos = file['measurements/Measurement_1/libs/metadata/X_pos'][:]
            y_pos = file['measurements/Measurement_1/libs/metadata/Y_pos'][:]
            x_len = file['measurements/Measurement_1/libs/metadata/x'][:]
            y_len = file['measurements/Measurement_1/libs/metadata/y'][:]
            x_step = file['measurements/Measurement_1/global_metadata/Width Spacing'][:]
            y_step = file['measurements/Measurement_1/global_metadata/Height Spacing'][:]
            #Select the method of intensity calculation
            method = request.json['method']
            if method == 'voigt_fit':
                intensity = voigt_fit(data, wavelength, b1_w, b2_w)
            elif method == 'peak_intensity':
                intensity = peak_intensity(data, wavelength, b1_w, b2_w)
            elif method == 'simple_sum':
                intensity = simple_sum(data, wavelength, b1_w, b2_w)
            elif method == 'simple_voigt_fit':
                intensity = simple_voigt_fit(data, wavelength, b1_w, b2_w)
            else:   #default to voigt_fit
                intensity = simple_voigt_fit(data, wavelength, b1_w, b2_w)
                return jsonify({'success': False, 'error': 'Invalid method'}), 400
            
            # Calculate central wavelength (a1_w) - peak position in the wavelength range
            b1_idx = np.argmin(np.abs(wavelength - b1_w))
            b2_idx = np.argmin(np.abs(wavelength - b2_w))
            # Use average spectrum to find peak
            avg_spectrum = np.mean(data[:, b1_idx:b2_idx], axis=0)
            peak_idx = np.argmax(avg_spectrum)
            a1_idx = b1_idx + peak_idx
            a1_w = float(wavelength[a1_idx])
            
            return jsonify({
                'success': True,
                'intensity': intensity.tolist(),
                'x_pos': x_pos.tolist(),
                'y_pos': y_pos.tolist(),
                'x_len': x_len.tolist(),
                'y_len': y_len.tolist(),
                'x_step': x_step.tolist(),
                'y_step': y_step.tolist(),
                'a1_w': a1_w
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/plot_spectrum', methods=['POST'])
def plot_spectrum():
    """Generate and return interactive spectrum plot"""
    try:
        wavelength = request.json['wavelength']
        max_data = request.json['max_data']
        # Use background-subtracted data if available, otherwise fall back to max_data
        max_data_subtracted = request.json.get('max_data_subtracted', max_data)
        title = request.json.get('title', 'Maximum Intensity Spectrum')
        
        # Create interactive Plotly figure and add trace for syntetic spectra
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavelength,
            y=max_data_subtracted,
            mode='lines',
            line=dict(color='blue', width=1.5),
            name='Max Intensity'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Wavelength (nm)',
            yaxis_title='Intensity (a.u.)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return jsonify({
            'success': True,
            'plot': fig.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/plot_intensity_map', methods=['POST'])
def plot_intensity_map():
    """Generate and return interactive intensity map"""
    try:
        intensity = request.json['intensity']
        x_pos = request.json['x_pos']
        y_pos = request.json['y_pos']
        title = request.json.get('title', 'Intensity Map')
        x_len = request.json['x_len']
        y_len = request.json['y_len']
        x_step = request.json['x_step']
        y_step = request.json['y_step']

        # Convert x_step and y_step to integers
        # Extract scalar value if it's a list/array
        x_step = int(x_step[0] if isinstance(x_step, (list, np.ndarray)) else x_step)
        y_step = int(y_step[0] if isinstance(y_step, (list, np.ndarray)) else y_step)
        x_len = int(np.max(x_len))
        y_len = int(np.max(y_len))

        if x_len < y_len:
            x_len, y_len = y_len, x_len
            x_step, y_step = y_step, x_step
            x_pos, y_pos = y_pos, x_pos

        x_span = x_len*x_step
        y_span = y_len*y_step

        if x_span > 0 and y_span > 0:
            real_aspect_ratio = y_span/x_span
        else:
            real_aspect_ratio = 1

        base_size = 1000

        if real_aspect_ratio > 1.0:
            plot_height = base_size
            plot_width = int(plot_height/real_aspect_ratio)
        else:
            plot_width = base_size
            plot_height = int(plot_width*real_aspect_ratio)

        legend_width = 200
        margin_space = 100
        total_width = plot_width + legend_width + margin_space
        total_height = plot_height + margin_space
        print(f"total_width: {total_width}, total_height: {total_height}")
        print(f"aspect ratio: {real_aspect_ratio}")

     

        print(f"x_step: {x_step}, y_step: {y_step}")
        print(f"x_len: {int(np.max(x_len))}, y_len: {int(np.max(y_len))}")
        
        # Create interactive Plotly scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers',
            marker=dict(
                size=8,
                color=intensity,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Peak Intensity')
            ),
            text=[f'Intensity: {i:.2f}' for i in intensity],
            hovertemplate='X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='X Position (mm)',
            yaxis_title='Y Position (mm)',
            template='plotly_white',       
            width=int(total_width),
            height=int(total_height),
            hovermode='closest',
            autosize=False,
            xaxis=dict(
                scaleanchor='y',
                scaleratio=1#real_aspect_ratio
            ),
            margin=dict(
                l=80,
                r=legend_width + 20,
                t=60,
                b=60
            ),
            legend=dict(
                title='Intensity',
                itemsizing='constant',
                itemwidth=30,
                x=1.02,
                xanchor='left',
                y=1.0,
                yanchor='top'
            )
        )
        
        return jsonify({
            'success': True,
            'plot': fig.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/browse_files', methods=['POST'])
def browse_files():
    """List files and directories in a given path"""
    try:
        path = request.json.get('path', '/')
        
        # Validate path exists
        if not os.path.exists(path):
            return jsonify({
                'success': False,
                'error': f'Path does not exist: {path}'
            }), 404
        
        # Security: prevent directory traversal beyond allowed paths
        # You can set a base directory to restrict browsing
        # base_dir = '/home/LIBS/prochazka/data'  # Example restriction
        
        items = []
        
        try:
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                is_dir = os.path.isdir(item_path)
                items.append({
                    'name': item,
                    'path': item_path,
                    'is_directory': is_dir,
                    'size': os.path.getsize(item_path) if not is_dir else None
                })
        except PermissionError:
            return jsonify({
                'success': False,
                'error': f'Permission denied: {path}'
            }), 403
        
        return jsonify({
            'success': True,
            'current_path': path,
            'items': items,
            'parent_path': os.path.dirname(path) if path != '/' else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plot_syntetic_spectra', methods=['POST'])
def plot_syntetic_spectra():
    """Plot syntetic spectra for a given element"""
    try:
        element = request.json['element']
        wavelength = request.json['wavelength']
        max_data = request.json.get('max_data')  # Optional - for scaling
        
        # Generate synthetic spectra
        syntetic_spectra = create_spectra(element, wavelength, Te=12705, Ne=1.79e+18, N=1e-4, C=1, l=1.4e-04)
        
        # Scale synthetic spectra to match max_data if provided
        if max_data is not None and len(max_data) > 0:
            max_data_array = np.array(max_data, dtype=np.float64)
            syntetic_spectra_array = np.array(syntetic_spectra, dtype=np.float64)
            
            syntetic_max = np.max(syntetic_spectra_array)
            max_coeff = np.argmax(syntetic_spectra_array)
            max_data_max = np.max(max_data_array[(max_coeff-3):(max_coeff+3)])
            
            print(f"Scaling info - max_data max: {max_data_max}, syntetic max: {syntetic_max}")
            print(f"max_data type: {type(max_data)}, length: {len(max_data) if hasattr(max_data, '__len__') else 'N/A'}")
            print(f"syntetic_spectra shape: {syntetic_spectra_array.shape}, dtype: {syntetic_spectra_array.dtype}")
            
            # Check for valid values
            if np.isfinite(syntetic_max) and np.isfinite(max_data_max) and syntetic_max > 0 and max_data_max > 0:
                scaler = max_data_max / syntetic_max
                syntetic_spectra_scaled = syntetic_spectra_array * scaler
                syntetic_spectra = syntetic_spectra_scaled.tolist()
                print(f"Scaler applied: {scaler}, new syntetic max: {np.max(syntetic_spectra_scaled)}")
            else:
                print(f"Warning: Cannot scale - syntetic_max={syntetic_max} (finite: {np.isfinite(syntetic_max)}), max_data_max={max_data_max} (finite: {np.isfinite(max_data_max)})")
                # Convert to list if not already
                if not isinstance(syntetic_spectra, list):
                    syntetic_spectra = syntetic_spectra.tolist()
        else:
            print(f"No max_data provided (max_data is None: {max_data is None}, length: {len(max_data) if max_data is not None else 'N/A'}), returning unscaled synthetic spectra")
            # Convert to list if not already
            if not isinstance(syntetic_spectra, list):
                syntetic_spectra = syntetic_spectra.tolist()
        
        return jsonify({
            'success': True,
            'syntetic_spectra': syntetic_spectra
        })
    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing required parameter: {str(e)}'}), 400
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in plot_syntetic_spectra: {error_msg}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calculate_kmeans', methods=['POST'])
def calculate_kmeans():
    """Calculate clustering (k-means or Mean Shift) with specified parameters"""
    try:
        # Check if request has JSON data
        if not request.json:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        # Get PC scores with error handling
        PC1_scores = request.json.get('PC1_scores')
        PC2_scores = request.json.get('PC2_scores')
        PC3_scores = request.json.get('PC3_scores')
        n_clusters = int(request.json.get('n_clusters', 3))
        clustering_method = request.json.get('clustering_method', 'kmeans')
        
        # Validate that PC scores are provided
        if PC1_scores is None:
            return jsonify({'success': False, 'error': 'PC1_scores is required'}), 400
        if PC2_scores is None:
            return jsonify({'success': False, 'error': 'PC2_scores is required'}), 400
        if PC3_scores is None:
            return jsonify({'success': False, 'error': 'PC3_scores is required'}), 400
        
        # Convert to numpy arrays
        PC1_scores = np.array(PC1_scores)
        PC2_scores = np.array(PC2_scores)
        PC3_scores = np.array(PC3_scores)
        
        # Validate array lengths match
        if not (len(PC1_scores) == len(PC2_scores) == len(PC3_scores)):
            return jsonify({'success': False, 'error': 'PC scores arrays must have the same length'}), 400
        
        # Apply clustering based on selected method
        if clustering_method == 'meanshift':
            # Mean Shift clustering - prefers density
            # Use all 3 PC scores for Mean Shift
            pca_data = np.column_stack([PC1_scores, PC2_scores, PC3_scores])
            # Estimate bandwidth automatically
            bandwidth = estimate_bandwidth(pca_data, quantile=0.2, n_samples=min(500, len(pca_data)))
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            cluster = meanshift.fit_predict(pca_data)
            n_clusters_found = len(np.unique(cluster))
            print(f"Mean Shift clustering: {cluster}, bandwidth: {bandwidth}, n_clusters: {n_clusters_found}")
            
            return jsonify({
                'success': True,
                'kmeans_labels': cluster.tolist(),
                'clustering_method': 'meanshift',
                'n_clusters_found': n_clusters_found,
                'bandwidth': float(bandwidth)
            })
        else:
            # Default to k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
            cluster = kmeans.fit_predict(np.column_stack([PC1_scores, PC2_scores, PC3_scores]))
            print(f"K-means clustering: {cluster}")
            
            return jsonify({
                'success': True,
                'kmeans_labels': cluster.tolist(),
                'clustering_method': 'kmeans'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)