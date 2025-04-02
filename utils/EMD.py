import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def voxelize_point_cloud(points, voxel_size):
    if points.size == 0:
        raise ValueError()

    voxel_indices = np.floor(points / voxel_size).astype(int)

    unique_voxels, voxel_counts = np.unique(voxel_indices, axis=0, return_counts=True)

    weights = voxel_counts / voxel_counts.sum()

    voxel_centers = (unique_voxels + 0.5) * voxel_size

    return voxel_centers, weights

def sinkhorn_knopp_emd(source_points, target_points, source_weights, target_weights, epsilon=0.01, max_iter=3000, tol=1e-3):

    if source_points.shape[1] != target_points.shape[1]:
        raise ValueError()
    if not (np.isclose(source_weights.sum(), 1) and np.isclose(target_weights.sum(), 1)):
        raise ValueError()

    cost_matrix = cdist(source_points, target_points, metric='euclidean')

    # Sinkhorn-Knopp
    K = np.exp(-cost_matrix / epsilon)
    K += 1e-9  

    u = np.ones_like(source_weights)
    v = np.ones_like(target_weights)

    with tqdm(total=max_iter, desc="Running Sinkhorn", unit="iter") as pbar:
        for _ in range(max_iter):
            pbar.update(1)
            u_prev = u.copy()
            u = source_weights / (K @ v)
            v = target_weights / (K.T @ u)

            diff = np.linalg.norm(u - u_prev, 1)
            pbar.set_postfix({'diff': f"{diff:.5e}"})
            if diff < tol:
                break

    transport_matrix = np.outer(u, v) * K
    emd_approx = np.sum(transport_matrix * cost_matrix)

    return emd_approx

def calc_EMD_with_sinkhorn_knopp(point_cloud_1, point_cloud_2, voxel_size=0.5, epsilon=0.001, max_iter=3000, tol=1e-4):

    voxel_centers_1, weights_1 = voxelize_point_cloud(point_cloud_1, voxel_size)
    voxel_centers_2, weights_2 = voxelize_point_cloud(point_cloud_2, voxel_size)

    # clip
    if voxel_centers_1.shape[0] + voxel_centers_2.shape[0] > 130000:
        return None

    return sinkhorn_knopp_emd(voxel_centers_1, voxel_centers_2, weights_1, weights_2, epsilon, max_iter, tol)

if __name__ == "__main__":

    num_points_1 = 10000
    num_points_2 = 120000
    point_cloud_1 = np.random.rand(num_points_1, 3) * 15
    point_cloud_2 = (np.random.rand(num_points_2, 3)-0.5) * 15

    emd_result = calc_EMD_with_sinkhorn_knopp(point_cloud_1, point_cloud_2, voxel_size=0.5, epsilon=0.001, max_iter=3000, tol=1e-4)

    print(f"Approximate Earth Mover's Distance (Sinkhorn): {emd_result:.4f}")
