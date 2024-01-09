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
        if line[0] == "MOTION":
            break
        if line[0] == "CHANNELS":
            joint_channel_count.append(int(line[1]))
            joint_channel.append(line[2:len(line)])
        if line[0] == "ROOT":
            joint_name.append(line[-1])
            joint_parent.append(-1)
        if line[0] == "JOINT":
            joint_name.append(line[-1])
            joint_parent.append(joint_name.index(joint_reading_stack[-1]))
        if line[0] == "End":
            joint_name.append(joint_name[-1] + "_end")
            joint_parent.append(joint_name.index(joint_reading_stack[-1]))
        if line[0] == "OFFSET":
            joint_offset.append([float(line[1]), float(line[2]), float(line[3])])
        if line[0] == "{":
            joint_reading_stack.append(joint_name[-1])
        if line[0] == "}":
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
    frame_motion_data = frame_motion_data.reshape(-1, 3)  # shape: (M, 3)

    # 梳理每个节点的局部旋转，包含 end 节点
    joint_local_rotation = []
    end_joint_count = 0
    for i in range(len(joint_names)):
        if '_end' in joint_names[i]:
            joint_local_rotation.append(np.zeros(3))
            end_joint_count += 1
        else:
            joint_local_rotation.append(frame_motion_data[i - end_joint_count + 1, :])  # 根节点没有旋转，所以从第二个开始
    # temp = np.array(joint_local_rotation)

    # 由根节点开始，遍历计算旋转和位置
    # 由于数组顺序的特殊性，parent 一定会比 child 先更新，从根节点开始走完一次即可确保完成所有节点的计算
    for i in range(len(joint_names)):
        if joint_parents[i] == -1:  # 根节点
            joint_position = frame_motion_data[0]
            joint_orientation = R.from_euler('XYZ', joint_local_rotation[i], degrees=True).as_quat()
        else:
            Q_parent = R.from_quat(joint_orientations[joint_parents[i]])
            R_current = R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_orientation = (R_current * Q_parent).as_quat()  # Q_{i} = Q_{i-1} * R_{i}
            joint_position = joint_positions[joint_parents[i]] + R.from_quat(joint_orientations[joint_parents[i]]).apply(joint_offsets[i])  # p_{i+1} = p_{i} + Q_{i} * L_{i}

        joint_orientations.append(np.array(joint_orientation))
        joint_positions.append(np.array(joint_position))

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
