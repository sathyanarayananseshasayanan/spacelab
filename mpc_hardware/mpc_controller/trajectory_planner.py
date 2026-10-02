#!/usr/bin/env python3
"""
RRT-based trajectory planner for docking with collision avoidance.
Uses Rapidly-exploring Random Tree to generate collision-free paths.
"""

import numpy as np
import math
from scipy.interpolate import CubicSpline


class RRTNode:
    """Node in the RRT tree."""
    def __init__(self, pos: np.ndarray):
        self.pos = np.array(pos, dtype=float)  # (2,)
        self.parent = None
        self.children = []
    
    def distance_to(self, other_pos: np.ndarray) -> float:
        """Euclidean distance to another position."""
        return np.linalg.norm(self.pos - other_pos)


class SmoothTrajectoryPlanner:
    """
    Generates smooth reference trajectories between current pose and docking goal.
    Uses cubic spline interpolation and collision checking.
    """
    
    def __init__(self, safe_distance: float = 0.3, goal_shift: float = 0.2):
        """
        Args:
            safe_distance: Minimum allowed distance from collision point (docking target)
            goal_shift: Distance ahead of dock to place shifted goal (for approach phase)
        """
        self.safe_distance = safe_distance
        self.goal_shift = goal_shift
        self.rrt_max_iter = 500        # Max RRT iterations
        self.rrt_step_size = 0.1       # Max step size from RRT node
        self.rrt_goal_bias = 0.15      # Probability to sample goal directly
        self.path_samples = 20         # Waypoints to extract from RRT path
        
    def plan_trajectory(self, 
                       x_current: np.ndarray,
                       x_goal: np.ndarray,
                       dt: float,
                       horizon: int,
                       target_velocity: np.ndarray = None) -> np.ndarray:
        """
        Generate smooth reference trajectory using RRT for collision-free planning.
        Supports moving docking targets.
        
        Args:
            x_current: Current state [x, y, psi, vx, vy, r] (6,)
            x_goal: Goal state [x, y, psi, vx, vy, r] (6,)
            dt: Time step (seconds)
            horizon: MPC horizon (number of steps)
            target_velocity: Target velocity [vx, vy] (2,), None = stationary target
        
        Returns:
            Reference trajectory (6, horizon+1)
        """
        pos_current = x_current[:2]
        pos_goal = x_goal[:2]
        psi_current = x_current[2]
        psi_goal = x_goal[2]
        
        # Store target velocity for collision checking
        self.target_velocity = target_velocity if target_velocity is not None else np.zeros(2)
        self.dock_pos_initial = pos_goal.copy()
        
        # Plan collision-free path via RRT with moving target
        rrt_path = self._plan_rrt_path(pos_current, pos_goal)
        
        if rrt_path is None or len(rrt_path) < 2:
            # Fallback to direct path if RRT fails
            rrt_path = np.array([pos_current, pos_goal])
        
        # Smooth the RRT path via cubic spline
        waypoints_xy = self._smooth_path(rrt_path)
        
        # Generate headings along path
        headings = self._smooth_headings(waypoints_xy, psi_current, psi_goal)
        
        # Interpolate to full horizon
        ref_traj = self._interpolate_trajectory(
            waypoints_xy, headings, x_goal, dt, horizon
        )
        
        return ref_traj
    
    def _plan_rrt_path(self, pos_start: np.ndarray, pos_goal: np.ndarray) -> np.ndarray:
        """
        Plan collision-free path using RRT algorithm with moving target support.
        
        Args:
            pos_start: Starting position (2,)
            pos_goal: Goal position (2,) at t=0
        
        Returns:
            Path as array of waypoints (N, 2), or None if no path found
        """
        # Initialize tree with start node
        root = RRTNode(pos_start)
        nodes = [root]
        
        # Track cumulative time as we build path
        node_times = {id(root): 0.0}  # Time at each node
        
        goal_reached = False
        
        for iteration in range(self.rrt_max_iter):
            # Sample random point (with goal bias)
            if np.random.rand() < self.rrt_goal_bias:
                sample_pos = pos_goal.copy()
            else:
                # Random point in region around start/goal
                center = (pos_start + pos_goal) / 2.0
                radius = np.linalg.norm(pos_goal - pos_start)
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, radius * 1.5)
                sample_pos = center + r * np.array([np.cos(angle), np.sin(angle)])
            
            # Find nearest node in tree
            nearest_node = min(nodes, key=lambda n: n.distance_to(sample_pos))
            
            # Steer from nearest toward sample
            direction = sample_pos - nearest_node.pos
            dist = np.linalg.norm(direction)
            
            if dist < 1e-6:
                continue
            
            direction = direction / dist
            new_pos = nearest_node.pos + min(dist, self.rrt_step_size) * direction
            
            # Get time at nearest node for space-time collision check
            t_nearest = node_times[id(nearest_node)]
            
            # Check collision along edge (in space-time with moving target)
            if not self._edge_collision_free(nearest_node.pos, new_pos, pos_goal, time=t_nearest):
                continue
            
            # Add new node
            new_node = RRTNode(new_pos)
            new_node.parent = nearest_node
            nearest_node.children.append(new_node)
            nodes.append(new_node)
            
            # Estimate time at new node
            edge_dist = np.linalg.norm(new_pos - nearest_node.pos)
            t_new = t_nearest + edge_dist / max(0.1, self.rrt_step_size)
            node_times[id(new_node)] = t_new
            
            # Check if goal is reachable
            if new_node.distance_to(pos_goal) < self.rrt_step_size:
                goal_reached = True
                break
        
        if not goal_reached:
            return None
        
        # Extract path by backtracking from last node to root
        path = []
        current = nodes[-1]
        while current is not None:
            path.insert(0, current.pos)
            current = current.parent
        
        return np.array(path)
    
    def _edge_collision_free(self, pos1: np.ndarray, pos2: np.ndarray, 
                            dock_pos: np.ndarray, time: float = 0.0, dt: float = 0.1) -> bool:
        """
        Check if straight line from pos1 to pos2 maintains safe distance from moving dock.
        Checks collision in space-time.
        
        Args:
            pos1: Start position (2,)
            pos2: End position (2,)
            dock_pos: Initial dock position (2,) at time=0
            time: Time at which this edge is traversed (seconds)
            dt: Time step for collision checking
        
        Returns:
            True if edge is collision-free
        """
        if not hasattr(self, 'target_velocity'):
            self.target_velocity = np.zeros(2)
        if not hasattr(self, 'dock_pos_initial'):
            self.dock_pos_initial = dock_pos.copy()
        
        # Sample multiple points along edge and in time
        num_space_samples = 10
        num_time_samples = 5  # Check along time dimension too
        
        # Estimate trajectory duration through this edge
        edge_dist = np.linalg.norm(pos2 - pos1)
        edge_duration = edge_dist / max(0.1, self.rrt_step_size)  # Rough estimate
        
        for i in range(num_space_samples + 1):
            spatial_frac = i / num_space_samples
            sample_pos = (1 - spatial_frac) * pos1 + spatial_frac * pos2
            
            # Check collision at different times along this edge
            for j in range(num_time_samples + 1):
                time_frac = j / num_time_samples
                t_at_sample = time + time_frac * edge_duration
                
                # Predict dock position at this time
                dock_pos_predicted = self.dock_pos_initial + self.target_velocity * t_at_sample
                
                # Check distance
                dist_to_dock = np.linalg.norm(sample_pos - dock_pos_predicted)
                
                if dist_to_dock < self.safe_distance:
                    return False
        
        return True
    
    def _smooth_path(self, rrt_path: np.ndarray) -> np.ndarray:
        """
        Smooth RRT path using cubic spline.
        
        Args:
            rrt_path: RRT waypoints (N, 2)
        
        Returns:
            Smoothed waypoints (2, path_samples)
        """
        if len(rrt_path) < 2:
            return rrt_path.T
        
        # Parameterize path by arc length
        t_param = np.linspace(0, 1, len(rrt_path))
        
        try:
            cs_x = CubicSpline(t_param, rrt_path[:, 0])
            cs_y = CubicSpline(t_param, rrt_path[:, 1])
            
            # Sample smoothed path
            t_smooth = np.linspace(0, 1, self.path_samples)
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            
            return np.array([x_smooth, y_smooth])
        except:
            # Fallback if spline fails
            indices = np.linspace(0, len(rrt_path)-1, self.path_samples, dtype=int)
            return np.array([rrt_path[indices, 0], rrt_path[indices, 1]])
    
    def _generate_waypoints(self, pos_start: np.ndarray, pos_goal: np.ndarray) -> np.ndarray:
        """Legacy method - now handled by RRT. Kept for compatibility."""
        rrt_path = self._plan_rrt_path(pos_start, pos_goal)
        if rrt_path is None:
            rrt_path = np.array([pos_start, pos_goal])
        return self._smooth_path(rrt_path)
    
    def _smooth_headings(self, waypoints_xy: np.ndarray, 
                        psi_start: float, psi_goal: float) -> np.ndarray:
        """
        Generate smooth heading reference along the path.
        
        Args:
            waypoints_xy: Position waypoints (2, num_waypoints)
            psi_start: Starting yaw (radians)
            psi_goal: Goal yaw (radians)
        
        Returns:
            Heading array (num_waypoints,)
        """
        num_wp = waypoints_xy.shape[1]
        headings = np.zeros(num_wp)
        
        # Compute tangent heading at each waypoint
        for i in range(num_wp):
            if i == 0:
                # First point: use start heading
                headings[i] = psi_start
            elif i == num_wp - 1:
                # Last point: use goal heading
                headings[i] = psi_goal
            else:
                # Intermediate: compute tangent from surrounding points
                dx = waypoints_xy[0, i+1] - waypoints_xy[0, i-1]
                dy = waypoints_xy[1, i+1] - waypoints_xy[1, i-1]
                headings[i] = math.atan2(dy, dx)
        
        # Smooth heading transitions via cubic spline
        try:
            t = np.linspace(0, 1, num_wp)
            cs_psi = CubicSpline(t, headings)
            headings_smooth = cs_psi(t)
        except:
            headings_smooth = headings
        
        return headings_smooth
    
    def _interpolate_trajectory(self, waypoints_xy: np.ndarray, headings: np.ndarray,
                               x_goal: np.ndarray, dt: float, horizon: int) -> np.ndarray:
        """
        Interpolate waypoints to full MPC horizon trajectory.
        
        Args:
            waypoints_xy: Position waypoints (2, num_wp)
            headings: Heading at each waypoint (num_wp,)
            x_goal: Goal state (6,)
            dt: Time step
            horizon: MPC horizon
        
        Returns:
            Reference trajectory (6, horizon+1)
        """
        num_wp = waypoints_xy.shape[1]
        ref_traj = np.zeros((6, horizon + 1))
        
        # Map waypoints to horizon steps
        for k in range(horizon + 1):
            # Normalized position along trajectory [0, 1]
            progress = min(k / horizon, 1.0)
            wp_index = progress * (num_wp - 1)
            
            # Interpolate position
            wp_low = int(wp_index)
            wp_high = min(wp_low + 1, num_wp - 1)
            frac = wp_index - wp_low
            
            x = waypoints_xy[0, wp_low] + frac * (waypoints_xy[0, wp_high] - waypoints_xy[0, wp_low])
            y = waypoints_xy[1, wp_low] + frac * (waypoints_xy[1, wp_high] - waypoints_xy[1, wp_low])
            psi = headings[wp_low] + frac * (headings[wp_high] - headings[wp_low])
            
            # Smooth deceleration near goal
            vel_scale = max(0.0, 1.0 - progress)  # Reduces velocity as we approach goal
            vx = x_goal[3] * vel_scale
            vy = x_goal[4] * vel_scale
            r = x_goal[5] * vel_scale
            
            ref_traj[:, k] = np.array([x, y, psi, vx, vy, r])
        
        return ref_traj
    
    def is_collision_free(self, pos: np.ndarray, dock_pos: np.ndarray) -> bool:
        """
        Check if position is at safe distance from dock.
        
        Args:
            pos: Current position (2,)
            dock_pos: Dock position (2,)
        
        Returns:
            True if distance >= safe_distance
        """
        dist = np.linalg.norm(pos - dock_pos)
        return dist >= self.safe_distance
