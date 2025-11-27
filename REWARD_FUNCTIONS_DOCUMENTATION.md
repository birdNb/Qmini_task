# Qmini 站立任务奖励函数详细说明

本文档详细说明所有奖励函数的具体功能、计算方式和设计目的。

---

## 一、辅助函数

### `ftol()` - 容差函数
**功能**: 平滑奖励塑形函数，用于将连续值映射到 [0, 1] 范围的奖励

**公式**: `reward = exp(-|value - center| / tolerance)`

**参数**:
- `value`: 输入值（如高度、角度等）
- `range_val`: 值的有效范围 [min, max]
- `center`: 目标中心值
- `tolerance`: 容差宽度（越小越严格）

**用途**: 当值接近目标中心时给予高奖励，远离时奖励指数衰减

---

## 二、多阶段检测

系统根据基座高度将训练分为3个阶段：

- **阶段1** (h < 0.25m): 身体扶正阶段 - 从倒地姿势开始调整身体姿态
- **阶段2** (0.25m ≤ h < 0.35m): 身体抬升阶段 - 将身体从低位置抬升
- **阶段3** (h ≥ 0.35m): 完全站立阶段 - 保持稳定站立姿态

---

## 三、任务奖励 (Task Rewards - rtask)

### 1. `rew_height_task` - 基座高度奖励
**功能**: 鼓励机器人达到目标高度

**计算方式**: 
```python
ftol(base_height, [0.0, target_height*1.5], target_height, 0.1)
```

**说明**:
- 使用 `ftol` 函数，当基座高度接近目标高度时给予奖励
- 目标高度: 0.43m（可在配置中修改）
- 容差: 0.1m，允许一定误差
- 奖励范围: 高度在 [0, 0.645m] 之间，中心在 0.43m

**权重**: `rew_scale_height_task = 1.0`

**设计目的**: 这是站立任务的核心目标，直接鼓励机器人站起来

---

### 2. `rew_orientation_task` - 身体朝向奖励
**功能**: 鼓励机器人保持身体垂直（直立姿态）

**计算方式**:
```python
ftol(orientation_z, [0.0, 1.0], 1.0, 0.05)
```

**说明**:
- `orientation_z`: 基座z轴在世界坐标系中的垂直分量，范围 [-1, 1]
  - 1.0 = 完全直立（z轴向上）
  - 0.0 = 水平
  - -1.0 = 倒置
- 目标值: 1.0（完全直立）
- 容差: 0.05（非常严格，要求接近完全直立）

**权重**: `rew_scale_orientation_task = 1.0`

**设计目的**: 确保机器人不仅站起来，还要保持正确的直立姿态

---

## 四、风格奖励 (Style Rewards - rstyle)

这些奖励用于约束动作风格，防止不自然或危险的动作。

### 3. `rew_waist_penalty` - 腰部扭转惩罚
**功能**: 防止腰部过度扭转（可能导致摔倒或损伤）

**计算方式**:
```python
waist_twist_penalty = (|yaw| > 1.4 rad) ? 1.0 : 0.0
rew_waist_penalty = -rew_scale_waist_penalty * waist_twist_penalty
```

**说明**:
- 阈值: 1.4 弧度（约 80 度）
- 当腰部扭转角度超过阈值时给予惩罚
- 使用 yaw 角作为腰部扭转的近似

**权重**: `rew_scale_waist_penalty = 10.0`（较大惩罚）

**设计目的**: 防止机器人做出不自然的扭转动作，保护关节

---

### 4. `rew_knee_penalty` - 膝盖角度惩罚
**功能**: 防止膝盖过度弯曲（可能导致关节损伤）

**计算方式**:
```python
max_knee_angle = max(|LL_knee_angle|, |RL_knee_angle|)
knee_penalty = (max_knee_angle > 2.85 rad) ? 1.0 : 0.0
rew_knee_penalty = -rew_scale_knee_penalty * knee_penalty
```

**说明**:
- 监控左右两个膝盖关节（LL_joint4, RL_joint4）
- 阈值: 2.85 弧度（约 163 度）
- 当任一膝盖角度超过阈值时给予惩罚

**权重**: `rew_scale_knee_penalty = 10.0`（较大惩罚）

**设计目的**: 保护膝盖关节，防止过度弯曲造成损伤

---

### 5. `rew_feet_distance_penalty` - 双脚距离惩罚
**功能**: 防止双脚距离过远（可能导致不稳定）

**计算方式**:
```python
feet_distance = |LL_ankle_angle - RL_ankle_angle|
feet_distance_penalty = (feet_distance > 0.9 rad) ? 1.0 : 0.0
rew_feet_distance_penalty = -rew_scale_feet_distance_penalty * feet_distance_penalty
```

**说明**:
- 使用左右脚踝关节角度差的绝对值作为双脚距离的近似
- 阈值: 0.9 弧度（约 51 度）
- 当双脚距离过远时给予惩罚

**权重**: `rew_scale_feet_distance_penalty = 10.0`（较大惩罚）

**设计目的**: 保持双脚合理间距，提高站立稳定性

---

### 6. `rew_shank_orientation` - 小腿朝向奖励
**功能**: 鼓励小腿保持垂直（良好的站立姿态）

**计算方式**:
```python
shank_angles = (LL_knee_angle + RL_knee_angle) / 2.0
rew_shank_orientation = rew_scale_shank_orientation * ftol(
    |shank_angles|, [0.0, π], 0.0, 0.1
)
```

**说明**:
- 使用左右膝盖角度的平均值估计小腿朝向
- 目标: 小腿垂直（角度 = 0）
- 容差: 0.1 弧度（约 5.7 度）

**权重**: `rew_scale_shank_orientation = 10.0`（较大奖励）

**设计目的**: 鼓励正确的腿部姿态，提高站立质量

---

## 五、正则化奖励 (Regularization Rewards - rregu)

这些奖励用于平滑动作，减少抖动和能量消耗。

### 7. `rew_joint_accel` - 关节加速度惩罚
**功能**: 抑制关节加速度，减少动作抖动

**计算方式**:
```python
joint_accel = (current_vel - prev_vel) / dt
joint_accel_norm = ||joint_accel||₂
rew_joint_accel = -rew_scale_joint_accel * joint_accel_norm
```

**说明**:
- 计算所有关节加速度的L2范数
- 加速度越大，惩罚越大
- 鼓励平滑、连续的动作

**权重**: `rew_scale_joint_accel = 2.5e-7`（很小，但累积效应重要）

**设计目的**: 减少动作抖动，提高实机部署的稳定性

---

### 8. `rew_action_rate` - 动作变化率惩罚
**功能**: 抑制动作突变，鼓励平滑的动作序列

**计算方式**:
```python
action_rate = ||actions_t - actions_{t-1}||₂
rew_action_rate = -rew_scale_action_rate * action_rate
```

**说明**:
- 计算当前动作与上一时刻动作的L2范数
- 动作变化越大，惩罚越大
- 鼓励连续、平滑的动作序列

**权重**: `rew_scale_action_rate = 1e-2`（0.01）

**设计目的**: 防止动作突变，提高控制平滑性

---

### 9. `rew_torque` - 扭矩惩罚
**功能**: 抑制高扭矩，减少能量消耗和关节负载

**计算方式**:
```python
torque_norm = ||joint_torques||₂
rew_torque = -rew_scale_torque * torque_norm
```

**说明**:
- 计算所有关节扭矩的L2范数
- 扭矩越大，惩罚越大
- 鼓励使用较小的力完成任务

**权重**: `rew_scale_torque = 2.5e-6`（很小）

**设计目的**: 减少能量消耗，延长机器人工作时间

---

### 10. `rew_power` - 功率惩罚
**功能**: 抑制高功率消耗，提高能效

**计算方式**:
```python
power = |Σ(torque_i * velocity_i)|
rew_power = -rew_scale_power * power
```

**说明**:
- 功率 = 扭矩 × 速度的绝对值
- 功率越大，惩罚越大
- 鼓励高效的动作策略

**权重**: `rew_scale_power = 2.5e-5`（很小）

**设计目的**: 优化能量效率，减少不必要的功率消耗

---

## 六、后任务奖励 (Post-task Rewards - rpost)

这些奖励仅在阶段3（完全站立后）激活，用于保持稳定站立。

### 11. `rew_ang_vel_post` - 站立后角速度奖励
**功能**: 站立后保持低角速度，提高稳定性

**计算方式**:
```python
ang_vel_xy = base_ang_vel[:, :2]  # 只考虑x, y方向
ang_vel_xy_norm = ||ang_vel_xy||₂
rew_ang_vel_post = rew_scale_ang_vel_post * exp(-2.0 * ang_vel_xy_norm) * stage3_mask
```

**说明**:
- 只考虑基座在x, y方向的角速度（忽略z轴旋转）
- 使用指数衰减函数：角速度越小，奖励越大
- 仅在阶段3（h ≥ 0.35m）激活

**权重**: `rew_scale_ang_vel_post = 10.0`（较大奖励）

**设计目的**: 站立后保持稳定，减少不必要的旋转

---

### 12. `rew_lin_vel_post` - 站立后线速度奖励
**功能**: 站立后保持低线速度，提高稳定性

**计算方式**:
```python
lin_vel_xy = base_lin_vel[:, :2]  # 只考虑x, y方向
lin_vel_xy_norm = ||lin_vel_xy||₂
rew_lin_vel_post = rew_scale_lin_vel_post * exp(-5.0 * lin_vel_xy_norm) * stage3_mask
```

**说明**:
- 只考虑基座在x, y方向的线速度（忽略z轴）
- 使用指数衰减函数：线速度越小，奖励越大
- 衰减系数 5.0 比角速度更严格
- 仅在阶段3激活

**权重**: `rew_scale_lin_vel_post = 10.0`（较大奖励）

**设计目的**: 站立后保持静止，避免不必要的移动

---

### 13. `rew_height_post` - 站立后高度维持奖励
**功能**: 站立后保持目标高度，防止下沉或过度上升

**计算方式**:
```python
height_error_post = |base_height - target_height|
rew_height_post = rew_scale_height_post * exp(-20.0 * height_error_post) * stage3_mask
```

**说明**:
- 计算当前高度与目标高度的绝对误差
- 使用指数衰减函数：误差越小，奖励越大
- 衰减系数 20.0 非常严格，要求精确维持高度
- 仅在阶段3激活

**权重**: `rew_scale_height_post = 10.0`（较大奖励）

**设计目的**: 站立后精确维持目标高度，提高站立质量

---

## 七、阶段特定奖励

### 14. `rew_joint` - 关节位置奖励
**功能**: 鼓励关节达到目标位置（仅在高度足够时激活）

**计算方式**:
```python
joint_error = ||joint_pos - target_pos||₂
height_ok_mask = (base_height >= 0.25m) ? 1.0 : 0.0
rew_joint = -rew_scale_joint * joint_error * height_ok_mask
```

**说明**:
- 计算关节位置与目标位置的L2范数误差
- 仅在基座高度 ≥ 0.25m 时激活（阶段2和3）
- 误差越小，奖励越大（负号表示惩罚误差）

**权重**: `rew_scale_joint = 1.0`

**设计目的**: 在身体抬升后，精确控制关节位置，达到正确的站立姿态

---

## 八、基础奖励

### 15. `rew_alive` - 存活奖励
**功能**: 鼓励机器人保持存活（未触发终止条件）

**计算方式**:
```python
rew_alive = rew_scale_alive * (1.0 - reset_terminated)
```

**说明**:
- 当机器人未触发终止条件时给予奖励
- 每步都给予，鼓励延长episode长度

**权重**: `rew_scale_alive = 0.1`（较小，但累积效应重要）

**设计目的**: 鼓励机器人保持存活，避免过早失败

---

### 16. `rew_term` - 终止惩罚
**功能**: 惩罚触发终止条件的情况

**计算方式**:
```python
rew_term = rew_scale_terminated * reset_terminated
```

**说明**:
- 当机器人触发终止条件（如摔倒、高度过低等）时给予惩罚
- 一次性惩罚，通常较大

**权重**: `rew_scale_terminated = -1.0`（负值，表示惩罚）

**设计目的**: 强烈惩罚失败，鼓励机器人避免触发终止条件

---

## 九、成功奖励

### 17. `rew_success` - 成功奖励
**功能**: 当机器人成功达到站立目标时给予大奖励

**成功条件**:
1. 所有关节位置误差 < `success_joint_tol` (0.05 rad)
2. 身体朝向误差 < `success_pitch_tol` (5度)
3. 基座高度 ≥ (target_height - 0.05m)

**计算方式**:
```python
success_mask = (
    (max_joint_error < success_joint_tol)
    & (|roll| < success_pitch_tol)
    & (|pitch| < success_pitch_tol)
    & (base_height >= target_height - 0.05)
)
rew_success = rew_scale_success * success_mask
```

**权重**: `rew_scale_success = 2.0`（较大奖励）

**设计目的**: 明确奖励成功完成站立任务，强化正确的行为

---

## 十、奖励总结

### 奖励分类汇总

| 类别 | 奖励项 | 权重范围 | 主要作用 |
|------|--------|----------|----------|
| **任务奖励** | 高度、朝向 | 1.0 | 核心目标：站起来并保持直立 |
| **风格奖励** | 腰部、膝盖、双脚、小腿 | 10.0 | 约束动作风格，防止危险动作 |
| **正则化奖励** | 加速度、动作率、扭矩、功率 | 1e-7 ~ 1e-2 | 平滑动作，减少抖动和能耗 |
| **后任务奖励** | 角速度、线速度、高度维持 | 10.0 | 站立后保持稳定 |
| **阶段特定** | 关节位置 | 1.0 | 精确控制关节 |
| **基础奖励** | 存活、终止 | 0.1, -1.0 | 鼓励存活，惩罚失败 |
| **成功奖励** | 成功 | 2.0 | 奖励完成任务 |

### 设计原则

1. **多阶段课程学习**: 不同阶段激活不同奖励，逐步引导学习
2. **多目标平衡**: 任务目标、动作风格、平滑性、稳定性综合考虑
3. **安全优先**: 风格奖励权重较大，防止危险动作
4. **实机部署**: 正则化奖励确保动作平滑，适合实机部署
5. **稳定性**: 后任务奖励确保站立后保持稳定

---

## 十一、权重调优建议

- **任务奖励权重**: 如果机器人无法站起来，可以适当增加 `rew_scale_height_task`
- **风格奖励权重**: 如果出现危险动作，增加相应的惩罚权重
- **正则化权重**: 如果动作抖动严重，增加 `rew_scale_action_rate` 和 `rew_joint_accel`
- **后任务权重**: 如果站立后不稳定，增加后任务奖励权重

---

*文档生成时间: 2025-01-27*
*基于 HoST (Humanoid Standing-up Control) 框架设计*

