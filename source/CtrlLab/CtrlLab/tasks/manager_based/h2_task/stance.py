"""Nominal two-foot support preload for the H2 USD home stance.

Static inverse-force balance uses this USD's 75.587 kg mass and level soles.
Physical effort limits remain enforced by each actuator. These torques are a
nominal PD target bias; active balance and stepping still require the policy.
"""

STANCE_PRELOAD_TORQUES = {'left_hip_pitch_joint': -1.0101489, 'left_hip_roll_joint': 4.97909581, 'left_hip_yaw_joint': -0.6675075, 'left_knee_joint': -27.07197389, 'left_ankle_roll_joint': -0.49265255, 'left_ankle_pitch_joint': 15.67942988, 'right_hip_pitch_joint': -1.14466532, 'right_hip_roll_joint': -4.76322872, 'right_hip_yaw_joint': 0.64115111, 'right_knee_joint': -26.20611724, 'right_ankle_roll_joint': 0.47841811, 'right_ankle_pitch_joint': 15.21599833}
