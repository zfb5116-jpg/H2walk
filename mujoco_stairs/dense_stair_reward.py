"""Finite per-episode exploration budgets for ascending stairs."""
import numpy as np


class DenseStairReward:
    def __init__(self, geometry):
        self.geometry = geometry
        self.best_approach = np.zeros(geometry.step_count)
        self.best_lift = np.zeros(geometry.step_count)
        self.best_reach = np.zeros(geometry.step_count)
        self.completed = 0

    def step(self, soles, support, root_x, root_y, height, index, dt, active):
        result = dict(stair_approach=0., stair_lift=0., stair_reach=0., stair_completion=0.)
        if not active or height <= 0 or self.geometry.direction != "up":
            return result
        # Reward actual sequential support once, including the final tread.
        result["stair_completion"] = max(index - self.completed, 0) / dt
        self.completed = max(index, self.completed)
        if index >= self.geometry.step_count or abs(root_y) > self.geometry.lane_half_width:
            return result
        edge = self.geometry.first_edge + index * self.geometry.tread_depth
        approach = float(np.clip((root_x - (edge - .65)) / .50, 0., 1.))
        lift = reach = 0.
        for foot, points in enumerate(soles):
            center = points.mean(axis=0)
            if support[foot] or not support[1 - foot] or abs(center[1]) > self.geometry.lane_half_width:
                continue
            # Use the lowest sole sample, so a pitched toe cannot fake clearance.
            if edge - .45 <= center[0] <= edge + .25 and edge - .65 <= root_x <= edge + .35:
                clearance = float(np.clip((points[:, 2].min() - index * height) / (height + .04), 0., 1.))
                lift = max(lift, clearance)
                forward = float(np.clip((center[0] - (edge - .40)) / .50, 0., 1.))
                reach = max(reach, clearance * forward)
        for name, value, best in (("stair_approach", approach, self.best_approach),
                                  ("stair_lift", lift, self.best_lift),
                                  ("stair_reach", reach, self.best_reach)):
            result[name] = max(value - best[index], 0.) / dt
            best[index] = max(best[index], value)
        return result
