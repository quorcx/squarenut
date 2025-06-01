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
    def __init__(self, square_pos, demo_path='demonstration_data.npz', dt=0.01, n_bfs=20):
        self.dt = dt
        # Load and parse demo [DO NOT CHANGE]
        raw = np.load(demo_path)
        demos = defaultdict(dict)
        for key in raw.files:
            prefix, trial, field = key.split('_', 2)
            demos[f"{prefix}_{trial}"][field] = raw[key]
        demo = demos['demo_98']

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        # TODO: Fit DMPs and generate segment trajectories
        # NOTE: you need a DMP for each segment
        self.trajectories = list()
        self.pids = list()
        self.grasps = list()

        for i, (start, end) in enumerate(segments):
            seg_traj = ee_pos[start:end]
            seg_grasp = ee_grasp[start:end].flatten()
            if i == 0:
                new_goal = new_obj_pos + offset
                dmp = DMP(n_dmps = 3, n_bfs = n_bfs, dt = dt)
                dmp.imitate(seg_traj)
                traj = dmp.rollout(new_goal=new_goal)
            else:
                dmp = DMP(n_dmps = 3, n_bfs = n_bfs, dt = dt)
                dmp.imitate(seg_traj)
                traj = dmp.rollout()
            self.trajectories.append(traj)
            pid = PID(kp = 5.0, ki = 0.0, kd = 0.1, target = traj[0])
            self.pids.append(pid)
            self.grasps.append(seg_grasp)

        self.current_segment = 0
        self.current_step = 0
        


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



    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
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
        traj = self.trajectories[self.current_segment]
        pid = self.pids[self.current_segment]

        y_des = traj[self.current_step] if self.current_step < len(traj) else traj[-1]
        
        pid.target = y_des
        delta_pos = pid.update(robot_eef_pos, dt = self.dt)

        grasp = -1 if self.current_segment % 2 == 0 else 1
                
        self.current_step += 1
        if (self.current_step >= len(traj) 
            and self.current_segment < len(self.trajectories) - 1):
            self.current_segment += 1
            self.current_step = 0
        
        action = np.zeros(7)
        action[:3] = delta_pos
        action[-1] = grasp
        return action
