import numpy as np
from pathlib import Path
from collections import defaultdict

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

def calculate_recall(gt_file, gt_file_query, est_file, recall_n=[1, 5], 
                                   dist_threshold=10.0, rot_threshold_deg=30.0):
    """
    Computes Recall@N considering both translation (meters) and rotation (degrees).
    """
    # 1. Parse Ground Truth
    # GT Format: timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, ...
    poses = {}
    files_to_read = [gt_file, gt_file_query]
    for gt_file in files_to_read:
        with open(gt_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                
                img_path = parts[0]
                # Quaternions: indices 2, 3, 4, 5 (qw, qx, qy, qz)
                q = np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
                t = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
                
                # Normalize quaternion to ensure it's a unit quaternion
                q = q / np.linalg.norm(q)
                poses[img_path] = {'t': t, 'q': q}

    # 2. Parse Estimates (Image Pairs)
    queries = defaultdict(list)
    with open(est_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                queries[parts[0]].append(parts[1])

    # 3. Evaluation
    recall_counts = {n: 0 for n in recall_n}
    total_queries = 0

    for q_path, retrieved_list in queries.items():
        # Heuristic matching: find which GT key is in the long path
        q_data = next((v for k, v in poses.items() if k in q_path), None)
        if q_data is None: continue
        
        total_queries += 1
        
        for rank, r_path in enumerate(retrieved_list, 1):
            r_data = next((v for k, v in poses.items() if k in r_path), None)
            if r_data is None: continue

            # --- TRANSLATION ERROR ---
            dist = np.linalg.norm(q_data['t'] - r_data['t'])
            print(f"Query: {q_path}, Retrieved: {r_path}, Distance: {dist:.2f}m")
            
            # --- ROTATION ERROR ---
            # Angle = 2 * acos(|q1 dot q2|)
            dot_product = np.clip(np.abs(np.dot(q_data['q'], r_data['q'])), 0, 1.0)
            rot_error_rad = 2 * np.arccos(dot_product)
            rot_error_deg = np.degrees(rot_error_rad)

            # Standard InLoc Success
            if dist <= dist_threshold:
                for n in recall_n:
                    if rank <= n:
                        recall_counts[n] += 1
                break

    # 4. Print Results
    print(f"--- Evaluation ({dist_threshold}m, {rot_threshold_deg}°) ---")
    print(f"Total Queries: {total_queries}")
    rec_calcs = []
    for n in sorted(recall_n):
        rec = (recall_counts[n] / total_queries) * 100 if total_queries > 0 else 0
        rec_calcs.append(rec)
        print(f"Recall@{n}: {rec:.2f}%")
    return rec_calcs[0]

recal_results = []
for target_layer in range(24):
    # for target_layer in [21]:
    results_file = "/home/dominic/Documents/lamar-benchmark/outputs/HGE/pair_selection/query_val_phone/map/fusion-salad-5/pairs_" +str(target_layer) + ".txt"
    # results_file = "/home/dominic/Documents/lamar-benchmark/outputs/HGE/pair_selection/map/map/fusion-salad-10_frustum_pose-120-20-250/pairs.txt"
    gt_file = "/home/dominic/Documents/lamar-benchmark/data/HGE/sessions/map/trajectories.txt"
    gt_file_query = "/home/dominic/Documents/lamar-benchmark/data/HGE/sessions/query_val_phone/proc/alignment_trajectories.txt"

    recall = calculate_recall(gt_file, gt_file_query, results_file, recall_n=[1, 5, 10])
    recal_results.append(recall)
print(recal_results)

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(range(1,25), recal_results, marker='o', markersize=10, linewidth=4)
# plt.title('Recall@1 for each VGGT Layer', fontsize=22)
plt.xlabel('VGGT Layer', fontsize=20)
plt.ylabel('Recall@1 (%)', fontsize=20)
plt.xticks(range(1,25))
# Increase tick size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid()
plt.show()

sorted_pairs = sorted(zip(range(1, 25), recal_results), key=lambda x: x[1], reverse=True)
layers_sorted, recall_sorted = zip(*sorted_pairs)

# Convert layer numbers to strings for categorical x-axis labels
x_labels = [str(l) for l in layers_sorted]

# Create the bar plot
plt.bar(x_labels, recall_sorted, color='skyblue', edgecolor='navy')

# Labels and formatting using LaTeX for mathematical notation
plt.xlabel(r'$\text{VGGT Layer}$', fontsize=20)
plt.ylabel(r'$\text{Recall@}1\ (\%)$', fontsize=20)

# Increase tick size and handle overlap
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust the y-axis to focus on the min/max range
# Adding a small 5% padding so the bars don't touch the top/bottom edges
min_val, max_val = min(recall_sorted), max(recall_sorted)
padding = (max_val - min_val) * 0.05
plt.ylim(min_val - padding, max_val + padding)

# Adjust layout and save the figure
plt.tight_layout()
plt.show()