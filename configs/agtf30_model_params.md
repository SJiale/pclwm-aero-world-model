# AGTF30e 参数与工况记录

## 模型
- 文件: AGTF30SysDyn_PExPIn.slx (398 KB, 2025-11-20)
- 求解器: ode4, FixedStep = MWS.In.Ts = 0.015 s
- StopTime: MWS.In.tend = 120 s

## Setup 调用链
1. MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2)
2. 修改 MWS.In.Options.* 与 MWS.In.t_*/值
3. MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2)
4. 【可选】缩放 MWS.Cntrl.WfCntrl.Nom.Kp_array / Ki_array
5. assignin('base','MWS',MWS)
6. sim('AGTF30SysDyn_PExPIn.slx')

## Options（固定）
MWS.In.Options.HybridConfig = 1      # 标准模式
MWS.In.Options.EPSLosses = 2
MWS.In.Options.EngineEMInt = 1
MWS.In.Options.PGBConfig = 2
MWS.In.Options.WfTransientLogic = 2
MWS.In.Options.TEEM = 1

## 电气系统（全部关闭）
Boost = zeros(1,6); Charge = zeros(1,4); EPT = zeros(1,6)

## 工况点（6 组）
(Alt ft, MN, 增益缩放):
(0, 0.0, 1.0), (0, 0.6, 1.0), (10000, 0.3, 1.0),
(20000, 0.6, 2.0), (20000, 0.3, 0.5), (30000, 0.5, 1.0)

## 环境
t_Alt=[0 10], t_MN=[0 10], t_dTamb=[0 10], dTamb=[0 0]

## 隐藏健康参数（未使用，供 B 组实验）
MWS.FAN.s_Eff = 1.036257
MWS.HPC.s_Eff = 0.994014
MWS.HPT.s_Eff = 0.998392
MWS.Burn.Eff  = 0.9990
MWS.GearBox.Eff = 1.0
来源: Setup_FAN.m L6, setup_HPC.m L7, setup_HPT.m L6, setup_AllEng.m L29/44

## 输出变量（15 个 out_*）
out_S0, out_S2, out_S25, out_S36, out_S4, out_S45, out_S5, out_S17,
out_Dyn, out_FANmp, out_HPCmp, out_LPCmp, out_HPTmp, out_LPTmp, out_S21

## 动作来源
out_Dyn.act.Wfact           -> 实际燃油 (lb/s)
out_Dyn.cntrl.Dmd.Wfdmd     -> 燃油指令 (lb/s)

## 增益调度
MWS.Cntrl.WfCntrl.Nom.Kp_array  shape [21 10 9 17 14], 范围 [0, 0.00763]
MWS.Cntrl.WfCntrl.Nom.Ki_array  shape [21 10 9 17 14], 范围 [0, 0.00842]
维度含义: (PL, Alt, MN, PwrInLP, PwrInHP)
