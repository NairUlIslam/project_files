import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import zipfile

zip_file_path = 'FINAL_ANALYSIS.zip'
extract_path = '.'

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted '{zip_file_path}' to '{extract_path}'")
else:
    print(f"Warning: '{zip_file_path}' not found. Assuming files are already extracted.")

all_files = os.listdir(extract_path)
mat_files = [f for f in all_files if f.endswith('.mat')]
print("Found .mat files:", mat_files)

try:
    roi_data = scipy.io.loadmat(os.path.join(extract_path, 'resultsROI_Condition001.mat'))
    roi_names_raw = roi_data['names']
    roi_names = [name[0] for name in roi_names_raw[0]]
except FileNotFoundError:
    print("Error: 'resultsROI_Condition001.mat' not found. Please ensure it's extracted.")
    exit()
except KeyError:
    print("Error: 'names' key not found in 'resultsROI_Condition001.mat'.")
    exit()

def clean_roi_name(name):
    name = name.replace('network ', '')
    name = name.replace('ROI ', '')
    name = name.replace('(R)', 'R')
    name = name.replace('(L)', 'L')
    name = name.replace(' ', '')
    return name

cleaned_roi_names = [clean_roi_name(name) for name in roi_names]

def abbreviate_roi_name(name):
    parts = name.split('.')
    if len(parts) > 1:
        network = parts[0]
        region = parts[-1]
        
        network_abbreviations = {
            'DefaultMode': 'DMN',
            'SensoriMotor': 'SMN',
            'Visual': 'VIS',
            'Salience': 'SAL',
            'DorsalAttention': 'DAN',
            'FrontoParietal': 'FPN',
            'Language': 'LAN',
            'Cerebellar': 'CER'
        }
        
        network_abbr = network_abbreviations.get(network, network)
        
        region_abbr = region
        if len(region_abbr) > 4 and region_abbr != 'ACC':
            if '_' in region_abbr:
                region_abbr = ''.join([p[0] for p in region_abbr.split('_')]).upper()
            elif len(region_abbr) > 5:
                region_abbr = region_abbr[:3].upper()
        return f"{network_abbr}.{region_abbr}"
    return name

abbreviated_roi_names = [abbreviate_roi_name(name) for name in cleaned_roi_names]

connectivity_data_control = []
connectivity_data_disease = []

num_subjects_total = 40
num_controls = 20

for i in range(1, num_subjects_total + 1):
    subject_file = os.path.join(extract_path, f'resultsROI_Subject{str(i).zfill(3)}_Condition001.mat')
    try:
        data = scipy.io.loadmat(subject_file)
        connectivity_matrix = data['Z'] 
        
        if connectivity_matrix.shape != (len(roi_names), len(roi_names)):
             print(f"Warning: Subject {i} matrix shape {connectivity_matrix.shape} unexpected. Expected ({len(roi_names)}, {len(roi_names)}). Skipping this subject or handling appropriately.")
             continue

        if i <= num_controls:
            connectivity_data_control.append(connectivity_matrix)
        else:
            connectivity_data_disease.append(connectivity_matrix)
    except FileNotFoundError:
        print(f"Warning: File {subject_file} not found. Skipping subject {i}.")
    except KeyError:
        print(f"Warning: Key 'Z' not found in {subject_file}. Skipping subject {i}.")
    except Exception as e:
        print(f"An error occurred processing {subject_file}: {e}. Skipping subject {i}.")

if not connectivity_data_control or not connectivity_data_disease:
    print("Error: No connectivity data loaded for one or both groups. Cannot proceed.")
    exit()

avg_connectivity_control = np.mean(connectivity_data_control, axis=0)
avg_connectivity_disease = np.mean(connectivity_data_disease, axis=0)

def plot_connectivity_matrix(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.show()

plot_connectivity_matrix(avg_connectivity_control, abbreviated_roi_names, 'Average Connectivity - Control Group')
plot_connectivity_matrix(avg_connectivity_disease, abbreviated_roi_names, 'Average Connectivity - Disease Group')
plot_connectivity_matrix(avg_connectivity_control - avg_connectivity_disease, abbreviated_roi_names, 'Connectivity Difference (Control - Disease)')

def draw_all_connectivity_network(diff_matrix, node_labels, title):
    G_all = nx.from_numpy_array(diff_matrix)
    
    pos_all = nx.spring_layout(G_all, k=0.5, iterations=50, seed=42) 
    
    edge_weights_all = np.array([G_all[u][v]['weight'] for u, v in G_all.edges()])
    
    weights_norm = []
    if edge_weights_all.size > 0:
        if np.ptp(edge_weights_all) > 0:
            weights_norm = (edge_weights_all - np.min(edge_weights_all)) / (np.max(edge_weights_all) - np.min(edge_weights_all))
        else:
            weights_norm = np.ones_like(edge_weights_all) * 0.5
    
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G_all, pos_all, node_color='skyblue', node_size=300, edgecolors='k', alpha=0.8)
    
    if G_all.number_of_edges() > 0 and len(weights_norm) > 0 :
        edges = nx.draw_networkx_edges(G_all, pos_all, edge_color=weights_norm, edge_cmap=plt.cm.coolwarm, width=1.5, alpha=0.7, vmin=0, vmax=1)
        if edge_weights_all.size > 0 and np.ptp(edge_weights_all) > 0 :
             sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=edge_weights_all.min(), vmax=edge_weights_all.max()))
             sm.set_array([])
             plt.colorbar(sm, label='Connectivity Difference (Control - Disease)', shrink=0.8, aspect=30)
        elif edge_weights_all.size > 0:
            print(f"Note: All edge weights in the difference network are uniform: {edge_weights_all[0] if edge_weights_all.size > 0 else 'N/A'}")
    else:
        print("No edges to draw for the network graph or weights_norm is empty.")

    labels_dict = {i: label for i, label in enumerate(node_labels)}
    nx.draw_networkx_labels(G_all, pos_all, labels=labels_dict, font_size=8)
    
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

draw_all_connectivity_network(avg_connectivity_control - avg_connectivity_disease, abbreviated_roi_names, 'Complete Connectivity Differences Network (Control - Disease)')

