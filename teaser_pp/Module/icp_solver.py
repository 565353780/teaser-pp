import torch
import numpy as np
from typing import Union, Optional

import teaserpp_python
from teaser_pp.Method.data import toNumpy, toPcd

NOISE_BOUND = 0.05


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));


class ICPSolver(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def register_points(
        source_pts: Union[torch.Tensor, np.ndarray, list],
        target_pts: Union[torch.Tensor, np.ndarray, list],
        common_point_num: Optional[int]=None,
    ) -> torch.Tensor:
        source_pts = toNumpy(source_pts, np.float64).reshape(-1, 3)
        target_pts = toNumpy(target_pts, np.float64).reshape(-1, 3)

        if common_point_num is None:
            common_point_num = min(source_pts.shape[0], target_pts.shape[0])

        source_pcd = toPcd(source_pts)
        target_pcd = toPcd(target_pts)

        if common_point_num < source_pts.shape[0]:
            print('[INFO][ICPSolver::register_points]')
            print('\t start sample source pts from', source_pts.shape[0], 'to', common_point_num, '...')
            source_pcd = source_pcd.farthest_point_down_sample(common_point_num)
        if common_point_num < target_pts.shape[0]:
            print('[INFO][ICPSolver::register_points]')
            print('\t start sample target pts from', target_pts.shape[0], 'to', common_point_num, '...')
            target_pcd = target_pcd.farthest_point_down_sample(common_point_num)

        src = np.transpose(np.asarray(source_pcd.points))
        dst = np.transpose(np.asarray(target_pcd.points))

        # Populating the parameters
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = NOISE_BOUND
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver.solve(src, dst)

        solution = solver.getSolution()
        return solution
