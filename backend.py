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
from LIBSmethods import peak_intensity, voigt_fit, simple_sum, simple_voigt_fit, calculate_snr
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
            
            return jsonify({
                'success': True,
                'wavelength': wavelength.tolist(),
                'max_data': max_data,
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
    file_path = request.json['file_path']
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': f'File not found: {file_path}'}), 404
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
            
            #normalize the data to standard normal distribution
            data_normalized = np.apply_along_axis(calculate_snr, 1, data)
            #calculate pca components to explain 95% of the variance
            pca = PCA(n_components=0.95)
            pca.fit(data_normalized)
            pca_components = pca.n_components_
            pca_explained_variance_ratio = pca.explained_variance_ratio_
            pca_explained_variance = pca.explained_variance_
            #calculate k-means clustering for the pca components
            n_clusters = request.json['n_clusters'] #number of clusters to use for k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42) #random state for reproducibility
            kmeans.fit(pca.components_)
            kmeans_labels = kmeans.labels_
            return jsonify({
                'success': True, 
                'pca_components': pca_components,
                'pca_explained_variance_ratio': pca_explained_variance_ratio.tolist(),
                'pca_explained_variance': pca_explained_variance.tolist(),
                'x_pos': x_pos.tolist(),
                'y_pos': y_pos.tolist(),
                'x_len': x_len.tolist(),
                'y_len': y_len.tolist(),
                'x_step': x_step.tolist(),
                'y_step': y_step.tolist(),
                'PC1': pca.components_[0].tolist(),
                'PC2': pca.components_[1].tolist(),
                'PC3': pca.components_[2].tolist()
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plot_pca_map', methods=['POST'])
def plot_pca_map():
    """Plot PCA for given parameters"""
    pca_components = request.json['pca_components']
    pca_explained_variance_ratio = request.json['pca_explained_variance_ratio']
    pca_explained_variance = request.json['pca_explained_variance']
    return jsonify({'success': True, 'pca_components': pca_components, 'pca_explained_variance_ratio': pca_explained_variance_ratio, 'pca_explained_variance': pca_explained_variance})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
            
            return jsonify({
                'success': True,
                'intensity': intensity.tolist(),
                'x_pos': x_pos.tolist(),
                'y_pos': y_pos.tolist(),
                'x_len': x_len.tolist(),
                'y_len': y_len.tolist(),
                'x_step': x_step.tolist(),
                'y_step': y_step.tolist()
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/plot_spectrum', methods=['POST'])
def plot_spectrum():
    """Generate and return interactive spectrum plot"""
    try:
        wavelength = request.json['wavelength']
        max_data = request.json['max_data']
        title = request.json.get('title', 'Maximum Intensity Spectrum')
        
        # Create interactive Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavelength,
            y=max_data,
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
            width=x_len*(x_step/10),
            height=y_len*(y_step/10),
            hovermode='closest'
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)