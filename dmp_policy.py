import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID


class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, square_quat, demo_path='demos.npz', dt=0.01, n_bfs=20):
        self.dt = dt
        # Load and parse demo [DO NOT CHANGE]
        raw = np.load(demo_path)
        demos = defaultdict(dict)
        for key in raw.files:
            prefix, trial, field = key.split('_', 2)
            demos[f"{prefix}_{trial}"][field] = raw[key]

        # Determine which demo to use based on initial position & orientation (closest neighbor)
        min_dist = float('inf')
        closest_key = None
        for demo_key, demo in demos.items() :
            demo_obj_pos = demo['obs_object'][0, :3]
            demo_obj_quat = demo['obs_object'][0, 3:7]
            pos_dist = np.linalg.norm(demo_obj_pos - square_pos)
            quat_angle = self.quat_angle(demo_obj_quat, square_quat)
            weighted_dist = pos_dist + 0.5 * quat_angle 
            if weighted_dist < min_dist :
                min_dist = weighted_dist
                closest_key = demo_key

        demo = demos[closest_key]
        print(closest_key)

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        # Extract quaternions and convert to axis-angle [Project]
        ee_quat = demo['obs_robot0_eef_quat']  # (T,4)

        # Compute quaternion offset for first segment to new object orientation [Project]
        demo_obj_quat = demo['obs_object'][0, 3:7]
        demo_obj_quat_inv = self.quat_inverse(demo_obj_quat)
        new_obj_quat = square_quat
        rel_rot = self.quat_multiply(new_obj_quat, demo_obj_quat_inv)

        # TODO: Fit DMPs and generate segment trajectories
        # NOTE: you need a DMP for each segment
        self.trajectories = list()
        self.pids_trajectories = list()
        self.grasps = list()
        self.orientations = list()

        # Fit a DMP for each segment (positions, grasps, quaternions of EEF)
        for i, (start, end) in enumerate(segments):
            seg_traj = ee_pos[start:end]
            seg_grasp = ee_grasp[start:end].flatten()
            seg_quat = ee_quat[start:end]

            if i == 0:
                # Approach and grasp the new object (with z offset to be above)
                pregrasp = new_obj_pos.copy()
                pregrasp[2] += 0.03
                seg_traj = np.vstack([pregrasp, seg_traj])

                # Compute the grasp offset and retarget to new object, yielding new goal
                grasp_offset = ee_pos[end-1] - demo_obj_pos
                new_goal = new_obj_pos + grasp_offset

                # DMP fit and rollout
                dmp = DMP(n_dmps = 3, n_bfs = n_bfs, dt = dt)
                dmp.imitate(seg_traj)
                traj = dmp.rollout(new_goal = new_goal)

                # Retarget the orientation to the square nut (using relative rotation)
                seg_quat_retargeted = np.array([self.quat_multiply(rel_rot, q) for q in seg_quat])
                traj_ori = self.slerp_trajectory(seg_quat_retargeted[0], seg_quat_retargeted[-1], len(traj))
                prev_last_quat = seg_quat_retargeted[-1]
            else:
                # 
                seg_quat = np.vstack([prev_last_quat, seg_quat])
                dmp = DMP(n_dmps = 3, n_bfs = n_bfs, dt = dt)
                dmp.imitate(seg_traj)
                traj = dmp.rollout()

                # Lift for segment 1: ensure Z position is above (peg + clearance)
                if i == 1:
                    peg_height = 0.04
                    nut_height = 0.02
                    clearance = 0.03
                    min_z = new_obj_pos[2] + peg_height + nut_height + clearance
                    lift_point = traj[0].copy()
                    lift_point[2] = min_z
                    traj = np.vstack([lift_point, traj])
                    traj[:, 2] = np.maximum(traj[:, 2], min_z)
                traj_ori = self.slerp_trajectory(seg_quat[0], seg_quat[-1], len(traj))
                prev_last_quat = seg_quat[-1]

            # Store the trajectory, PID controller, orientation and grasp
            self.trajectories.append(traj)
            self.pids_trajectories.append(PID(kp = 5.0, ki = 0.0, kd = 0.2, target = traj[0]))
            self.orientations.append(traj_ori)
            self.grasps.append(seg_grasp)

        # Initialize current segment and step
        self.current_segment = 0
        self.current_step = 0
                
    def quat_angle(self, quat1, quat2) :
        """
        [Project]
        Compute angle between two quaternions.
        
        Args:
            quat1 (np.ndarray): First quaternion [x,y,z,w].
            quat2 (np.ndarray): Second quaternion [x,y,z,w].
            
        Returns:
            float: Angular difference
        """
        dot_prod = np.dot(quat1, quat2)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        quat_angle = np.arccos(2 * dot_prod**2 - 1)
        return quat_angle

    def quat_to_axis_angle(self, quat):
        """
        [Project]
        Convert a quaternion to axis-angle representation.

        Parameters:
            q (array-like): Quaternion [x, y, z, w]

        Returns:
            rotation_vector (np.array): Axis-angle representation (T, 3)
        """
        x, y, z, w = quat
        norm = np.linalg.norm(quat)
        x, y, z, w = np.array(quat) / norm

        angle = 2 * np.arccos(w)
        s = np.sqrt(1 - w * w)

        if s < 1e-8:
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([x, y, z]) / s

        return axis * angle
    
    def quat_to_axis_angle_vectors(self, quats):
        """
        [Project]
        Convert a vectors of quaternions to axis-angle representation.

        Parameters:
            quats (array-like): Quaternion array (T, 4)

        Returns:
            rot_vecs (np.ndarray): Axis-angle representation (T, 3)
        """
        norm = np.linalg.norm(quats, axis = 1, keepdims = True)
        quats = np.asarray(quats) / norm
        w = quats[:, 3]

        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        sin_half_theta = np.sqrt(1 - np.clip(w ** 2, 0, 1))

        small_angle = sin_half_theta < 1e-6
        axis = np.zeros_like(quats[:, :3])
        axis[~small_angle] = (quats[~small_angle, :3].T / sin_half_theta[~small_angle]).T
        axis[small_angle] = np.array([1, 0, 0])

        rot_vecs = axis * theta[:, np.newaxis]
        return rot_vecs
    
    def slerp(self, q0, q1, t):
        """
        [Project]
        Spherical linear interpolation between two quaternions.

        Args:
            q0 (np.ndarray): First quaternion [x,y,z,w].
            q1 (np.ndarray): Second quaternion [x,y,z,w].
            t (float): Interpolation factor in [0, 1].
        
        Returns:
            np.ndarray: Interpolated quaternion [x,y,z,w].
        """
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0) + (s1 * q1)
    
    def slerp_trajectory(self, q_start, q_goal, steps):
        """
        [Project]
        Generate a trajectory of quaternions using spherical linear interpolation.

        Args:
            q_start (np.ndarray): Start quaternion [x,y,z,w].
            q_goal (np.ndarray): Goal quaternion [x,y,z,w].
            steps (int): Number of interpolation steps.
        
        Returns:
            np.ndarray: Array of quaternions (steps, 4).
        """
        traj = []
        for i in range(steps):
            t = i / (steps - 1)
            q_interp = self.slerp(q_start, q_goal, t)
            traj.append(self.quat_to_axis_angle(q_interp))
        return np.array(traj)
    
    def quat_to_rotmat(self, quat):
        """
        [Project]
        Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix.

        Args:
            quat (np.ndarray): Quaternion [x,y,z,w].

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        x, y, z, w = quat
        n = x * x + y * y + z * z + w * w
        if n < 1e-8:
            return np.eye(3)
        s = 2.0 / n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = np.array([
            [1 - s*(yy + zz), s*(xy - wz), s*(xz + wy)],
            [s*(xy + wz), 1 - s*(xx + zz), s*(yz - wx)],
            [s*(xz - wy), s*(yz + wx), 1 - s*(xx + yy)]
        ])
        return rot

    def quat_inverse(self, quat) :
        """
        [Project]
        Compute the inverse of a quaternion.

        Args:
            quat (np.ndarray): Quaternion [x,y,z,w].

        Returns:
            np.ndarray: Inverse quaternion [x,y,z,w].
        """
        quat = np.asarray(quat)
        quat_conj = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        norm_sq = np.dot(quat, quat)
        return quat_conj / norm_sq
    
    def quat_multiply(self, quat1, quat2):
        """
        [Project]
        Multiply two quaternions.
        
        Args:
            quat1 (np.ndarray): First quaternion [x,y,z,w].
            quat2 (np.ndarray): Second quaternion [x,y,z,w].
        
        Returns:
            np.ndarray: Resulting quaternion [x,y,z,w].
        """
        x1, y1, z1, w1 = quat1
        x2, y2, z2, w2 = quat2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return np.array([x, y, z, w])
    
    def axis_angle_to_quat(self, rotvec):
        """
        [Project]
        Convert an axis-angle vector to a quaternion.

        Args:
            rotvec (np.ndarray): Axis-angle vector [x,y,z].
            
        Returns:
            np.ndarray: Quaternion [x,y,z,w].
        """
        angle = np.linalg.norm(rotvec)
        if angle < 1e-8:
            return np.array([0, 0, 0, 1])
        axis = rotvec / angle
        w = np.cos(angle / 2)
        xyz = axis * np.sin(angle / 2)
        return np.array([xyz[0], xyz[1], xyz[2], w])

    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        # TODO: implement boundary detection
        grasp_segments = list()
        start = 0
        prev_grasp = grasp_flags[start, 0]
        end = len(grasp_flags)
        for i in range(1, end) :
            if grasp_flags[i, 0] != prev_grasp :
                grasp_segments.append((start, i))
                start = i
                prev_grasp = grasp_flags[start, 0]
        grasp_segments.append((start, end))
        return grasp_segments

    def get_action(self, robot_eef_pos: np.ndarray, robot_eef_quat: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        # TODO: select current segment and step
        # TODO: compute PID-based delta_pos
        # TODO: assemble action (zero rotation + grasp)
        # Trajectories, orientations, and grasps of the current segment
        traj = self.trajectories[self.current_segment]
        pid_traj = self.pids_trajectories[self.current_segment]
        ori = self.orientations[self.current_segment]
        grasp_traj = self.grasps[self.current_segment]

        # Desired position, orientation, and grasp of the current step
        y_des = traj[self.current_step] if self.current_step < len(traj) else traj[-1]
        y_des_ori = ori[self.current_step] if self.current_step < len(ori) else ori[-1]
        grasp = grasp_traj[self.current_step] if self.current_step < len(grasp_traj) else grasp_traj[-1]
        
        # Update PID target and compute change in position
        pid_traj.target = y_des
        delta_pos = pid_traj.update(robot_eef_pos, dt = self.dt)

        # Maximum size allowed for position change at each step
        max_pos_step = 0.2

        # Vector-norm-based clipping (prevents any axis from dominating)
        norm = np.linalg.norm(delta_pos)
        if norm > max_pos_step:
            delta_pos = delta_pos / norm * max_pos_step

        # Orientation control using proportional SO(3) controller
        kp = 1.0
        max_ang_vel = 0.2

        q_des = self.axis_angle_to_quat(y_des_ori)
        q_cur = robot_eef_quat

        # Quaternions are aligned (same direction)
        if np.dot(q_cur, q_des) < 0:
            q_des = -q_des
        R_cur = self.quat_to_rotmat(q_cur)
        R_des = self.quat_to_rotmat(q_des)
        R_err = np.dot(R_des, R_cur.T)
        error_vec = 0.5 * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1]
        ])
        rotation_error_vec = error_vec

        # Rotation error clipping to avoid large rotations
        norm = np.linalg.norm(rotation_error_vec)
        max_angle = np.pi / 2
        if norm > max_angle:
            rotation_error_vec = rotation_error_vec / norm * max_angle

        delta_ori = np.clip(kp * rotation_error_vec, -max_ang_vel, max_ang_vel)

        # Compute errors for position and orientation for acceptable transition
        pos_error = np.linalg.norm(robot_eef_pos - traj[-1])
        ori_error = self.quat_angle(robot_eef_quat, self.axis_angle_to_quat(ori[-1]))

        self.current_step += 1
        if (self.current_step >= len(traj) 
            and self.current_segment < len(self.trajectories) - 1
            and ( (pos_error < 0.003 and ori_error < 0.01)
            | (self.current_step > 1000 and self.current_segment == 0) ) ):
            print(f"Transitioning to segment {self.current_segment + 1}")
            self.current_segment += 1
            self.current_step = 0

        # Pause before lifting the object (entered segment 1)
        if self.current_segment == 1 and self.current_step < 30:
            delta_pos[:] = 0

        # Z position scaled to be lower for segment 0, and higher for segment 1
        # So that the robot doesnt crash into the square or not lift high enough at peg
        if self.current_segment == 1 :
            delta_pos[2] *= 5
        else :
            delta_pos[2] *= 0.5

        action = np.zeros(7)
        action[:3] = delta_pos
        action[3:6] = delta_ori
        action[-1] = grasp

        return action
