import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
from gtsam.symbol_shorthand import X

from foundit.slam_utils import decompose_camera, normalize_to_sl4

class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        inner_noise = 0.05*np.ones(15, dtype=float)
        intra_noise = 0.05*np.ones(15, dtype=float)
        self.inner_submap_noise = noiseModel.Diagonal.Sigmas(inner_noise) # TODO Dominic set these to different values
        self.intra_submap_noise = noiseModel.Diagonal.Sigmas(intra_noise) # TODO Dominic set these to different values
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        # Odometry factors are expected to be more accurate than the VGGT visual factors.
        # Sigma = 0.01 (5x tighter than the VGGT inner_submap_noise of 0.05).
        self.odom_noise = noiseModel.Diagonal.Sigmas(1e-6 * np.ones(15, dtype=float))
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis

    def add_homography(self, key, global_h):
        """Add a new homography node to the graph."""
        # print("det(global_h)", np.linalg.det(global_h))
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key in self.initialized_nodes:
            return
        self.values.insert(key, SL4(global_h))
        self.initialized_nodes.add(key)

    def add_between_factor(self, key1, key2, relative_h, noise):
        """Add a relative SL4 constraint between two nodes."""
        # relative_h = normalize_to_sl4(relative_h)
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(gtsam.BetweenFactorSL4(key1, key2, SL4(relative_h), noise))
    
    def add_prior_factor(self, key, global_h):
        # global_h = normalize_to_sl4(global_h)
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        factor_index = self.graph.size()
        self.graph.add(PriorFactorSL4(key, SL4(global_h), self.anchor_noise))
        return factor_index

    def update_homography(self, key, global_h):
        """Overwrite the value of an already-initialised node (no-op if absent)."""
        key = X(key)
        if key not in self.initialized_nodes:
            return
        self.values.update(key, SL4(global_h))

    def remove_factor(self, factor_index):
        """Null out a factor by index (used to release a temporary session anchor)."""
        if factor_index is not None:
            self.graph.remove(factor_index)

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node, or None if not yet initialised.
        """
        node_id = X(node_id)
        if node_id not in self.initialized_nodes:
            return None
        return self.values.atSL4(node_id).matrix()

    def get_projection_matrix(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node, or None if not yet initialised.
        """
        homography = self.get_homography(node_id)
        if homography is None:
            return None
        return np.linalg.inv(homography)

    # def optimize(self):
    #     """Optimize the graph incrementally using iSAM2 and print loss."""
    #     params = gtsam.ISAM2Params()
    #     isam = gtsam.ISAM2(params)

    #     # Compute and print initial error before update
    #     initial_error = self.graph.error(self.values)
    #     print(f"Initial error: {initial_error:.6f}")

    #     # Update iSAM2 with the current graph and initial values
    #     isam.update(self.graph, self.values)

    #     # Optionally perform an additional relinearization update
    #     isam.update()

    #     # Retrieve the optimized estimates
    #     self.values = isam.calculateEstimate()

    #     # Compute and print final error after optimization
    #     final_error = self.graph.error(self.values)
    #     print(f"Final error: {final_error:.6f}")

    #     # Optionally clear the graph to avoid re-adding old factors
    #     # self.graph.resize(0)


    def compute_between_factor_error(self, factor, values):
        """
        Compute residual r = Log(H_i^-1 * H_j * H_ij^-1) and Mahalanobis error r^T I r
        for a BetweenFactorPose3-like factor.
        Returns (error, r_vector (6,), info_matrix (6x6)).
        """
        # 1) extract keys
        keys = list(factor.keys())
        if len(keys) < 2:
            raise RuntimeError("Factor does not have two keys as expected for a between-pose factor")

        key_i, key_j = keys[0], keys[1]

        # 2) get Pose3 estimates Hi, Hj from values
        Hi = values.atSL4(key_i)
        Hj = values.atSL4(key_j)

        # 3) get measured Hij from the factor (common method names: measured(), measured or measured())
        Hij = factor.measured()           # common in many bindings


        # 4) compute relative: H_rel = Hi^-1 * Hj * Hij^-1
        H_rel = Hi.inverse().compose(Hj).compose(Hij.inverse())

        print("Hi:\n", Hi.matrix())
        print("Hj:\n", Hj.matrix())
        print("Hij:\n", Hij.matrix())
        print("H_rel:\n", H_rel.matrix())

        # 5) Logmap to get 15-vector residual
        r = np.asarray(gtsam.SL4.Logmap(H_rel)).reshape(-1)


        # 6) get information matrix from noise model
        nm = factor.noiseModel()
        info = np.asarray(nm.information())      # directly available for many Gaussian models

        # 7) Mahalanobis squared error
        e = float(r @ info @ r)
        return e, r, info

    
    def optimize(self, verbose=False):
        """Optimize the graph with Levenberg–Marquardt and print per-factor errors."""
        # Optional verbosity settings
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)

        # --- Initial total error ---
        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        # --- Per-factor initial error ---
        if verbose:
            print("\nInitial per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

            keys = [gtsam.DefaultKeyFormatter(k) for k in factor.keys()]
            print(f"Factor {i} connects to {keys} with error {e:.6f}")

        # --- Optimize ---
        result = optimizer.optimize()

        # --- Per-factor final error ---
        if verbose:
            print("\nFinal per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        # --- Store optimized values ---
        self.values = result


    def print_estimates(self):
        """Print the optimized poses."""
        for key in sorted(self.initialized_nodes):
            print(f"Homography{key}:\n{self.values.atSL4(key)}\n")
    
    def increment_loop_closure(self):
        """Increment the loop closure count."""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures."""
        return self.num_loop_closures
    def plot_poses(self):
        def extract_xz(poses):
            xz = []
            for p in poses:
                trans = decompose_camera(p)[2]
                xz.append([trans[0], trans[2]])
            return np.array(xz)

        # --- Collect all poses ---
        keys = sorted(self.initialized_nodes)
        poses = [np.linalg.inv(self.values.atSL4(key).matrix()) for key in keys]

        # --- Normalize so last element is 1 ---
        poses = [p / p[-1, -1] for p in poses]

        # --- Extract x-z positions ---
        xz = extract_xz(poses)

        plt.figure(figsize=(8, 6))
        plt.scatter(xz[:, 0], xz[:, 1], label='Poses', color='black', s=200, alpha=0.4)

        # --- Label each node with offset to avoid overlapping text ---
        n = len(xz)
        radius = 0.01  # small offset radius in meters
        angles = np.random.uniform(0, 2 * np.pi, n)

        for i, ((x, z), angle) in enumerate(zip(xz, angles), start=0):
            dx = radius * np.cos(angle)
            dz = radius * np.sin(angle)
            plt.text(x + dx, z + dz, str(i),
                    fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    color='blue', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.2))

        # --- Colormap setup ---
        num_factors = self.graph.size()
        cmap = cm.get_cmap("tab20", num_factors)
        norm = mcolors.Normalize(vmin=0, vmax=max(num_factors - 1, 1))

        # --- Draw edges based on BetweenFactorSL4 connections ---
        for i in range(num_factors):
            factor = self.graph.at(i)
            if isinstance(factor, gtsam.BetweenFactorSL4):
                key1 = factor.keys()[0]
                key2 = factor.keys()[1]
                if key1 in self.initialized_nodes and key2 in self.initialized_nodes:
                    idx1 = keys.index(key1)
                    idx2 = keys.index(key2)
                    p1 = xz[idx1]
                    p2 = xz[idx2]
                    color = cmap(norm(i))
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=color, linestyle='-', linewidth=2.0, alpha=1.0)

        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.title("Pose Graph with Node Labels and Colored Between Factors (X-Z Plane)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()