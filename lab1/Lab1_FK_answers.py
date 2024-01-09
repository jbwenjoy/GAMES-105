import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    joint_channel = []
    joint_channel_count = []

    joint_reading_stack = []
    lines = []

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = [name for name in lines[i].split()]
        if line[0] == "HIERARCHY":
            continue
        elif line[0] == "MOTION":
            break
        elif line[0] == "CHANNELS":
            joint_channel_count.append(int(line[1]))
            joint_channel.append(line[2:len(line)])
        elif line[0] == "ROOT":
            joint_name.append(line[-1])
            joint_parent.append(-1)
        elif line[0] == "JOINT":
            joint_name.append(line[-1])
            joint_parent.append(joint_name.index(joint_reading_stack[-1]))
        elif line[0] == "End":
            joint_name.append(joint_name[-1] + "_end")
            joint_parent.append(joint_name.index(joint_reading_stack[-1]))
        elif line[0] == "OFFSET":
            joint_offset.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == "{":
            joint_reading_stack.append(joint_name[-1])
        elif line[0] == "}":
            joint_reading_stack.pop()
    joint_offset = np.array(joint_offset).reshape(-1, 3)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_names, joint_parents, joint_offsets, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []

    frame_motion_data = motion_data[frame_id]
    frame_motion_data = frame_motion_data.reshape(-1, 3)

    quat = R.from_euler('XYZ', frame_motion_data, degrees=True).as_quat()

    # 处理 end 节点
    end_joints = []
    # 记录 end 节点 index
    for i in joint_names:
        if "end" in i:
            end_joints.append(joint_names.index(i))
    # 为 end 节点插入四元数
    for i in end_joints:
        quat = np.insert(quat, i, [0, 0, 0, 1], axis=0)

    # 由根节点开始，遍历计算旋转和位置
    # 由于数组顺序的特殊性，parent 一定会比 child 先更新，从根节点开始走完一次即可确保完成所有节点的计算
    # Q_{i} = Q_{i-1} * R_{i}
    # p_{i+1} = p_{i} + Q_{i} * L_{i}
    for joint_index in range(len(joint_parents)):
        joint_parent = joint_parents[joint_index]
        joint_offset = joint_offsets[joint_index]

        if joint_parent == -1:  # 根节点
            joint_positions.append(frame_motion_data[0])
            joint_orientations.append(quat[0])
        else:
            quat_current = R.from_quat(quat[joint_index])  # R_{i}
            quat_parent = R.from_quat(quat[joint_parent])  # Q_{i-1}
            joint_orientations.append(R.as_quat(quat_current * quat_parent))  # Q_{i} = Q_{i-1} * R_{i}

            joint_orientations_quat = R.from_quat(joint_orientations)  # 用于计算偏移量的四元数
            offset_rotation = joint_orientations_quat[joint_parent].apply(joint_offset)  # Q_{i} * L_{i}
            joint_positions.append(joint_positions[joint_parent] + offset_rotation)  # p_{i+1} = p_{i} + Q_{i} * L_{i}
    # list 转 numpy 数组
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = []
    motion_dict = {}
    joint_remove_A = []
    joint_remove_T = []

    # 读取 T-pose A-pose 骨骼，A-pose 动作
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)

    root_position = A_motion_data[:, :3]
    A_motion_data = A_motion_data[:, 3:]
    T_motion_data = np.zeros(A_motion_data.shape)

    for i in A_joint_name:
        if "_end" not in i:
            joint_remove_A.append(i)

    for i in T_joint_name:
        if "_end" not in i:
            joint_remove_T.append(i)

    for index, name in enumerate(joint_remove_A):
        motion_dict[name] = A_motion_data[:, 3*index:3*(index+1)]

    for index, name in enumerate(joint_remove_T):
        # simple convert from A pose to T pose
        if name == "lShoulder":
            motion_dict[name][:, 2] -= 45
        elif name == "rShoulder":
            motion_dict[name][:, 2] += 45
        T_motion_data[:, 3*index:3*(index+1)] = motion_dict[name]

    T_motion_data = np.concatenate([root_position, T_motion_data], axis=1)

    return T_motion_data
