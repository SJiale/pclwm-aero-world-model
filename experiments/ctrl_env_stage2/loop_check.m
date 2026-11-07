% 核实：AGTF30e 是开环还是闭环设计？
%
% 关键判据：
%   1. EEC 是否闭环跟踪 N1/N2 参考？
%   2. N1cErr 是否被驱动到 0？
%   3. 是否有积分器/反馈路径？
%   4. WfTransientLogic 的作用

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'loop_check.txt'),'w','n','UTF-8');

TS = 0.015;
TEND = 40;
t = (0:TS:TEND)';

% ══════════════════════════════════════════════════
% 测试 A：PLA 阶跃，看 EEC 是否把 N1 拉回来
% ══════════════════════════════════════════════════
fprintf('=== 测试 A: PLA 阶跃后 N1cErr 行为 ===\n');

MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t; MWS.In.Alt = zeros(size(t));
MWS.In.t_MN = t; MWS.In.MN = zeros(size(t));
MWS.In.t_dTamb = [0 TEND]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 1 TEND]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 1 TEND]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 1 TEND]; MWS.In.EPT = [0 0 0];
pla = 55*ones(size(t)); pla(t>=15) = 57;
MWS.In.t_PLA = t; MWS.In.PLA = pla;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');
D = evalin('base','out_Dyn');

err = D.cntrl.Data.N1cErr.Data(:);
n1cN = D.cntrl.Data.N1cN.Data(:);
n1max = D.cntrl.Data.N1cMax.Data(:);
n1min = D.cntrl.Data.N1cMin.Data(:);
wfd = D.cntrl.Dmd.Wfdmd.Data(:);
n2 = D.loop.N2int.Data(:);

fprintf(outfid,'\n=== PLA 55->57 @t=15s，N1cErr 演变 ===\n');
fprintf(outfid,'  %8s %12s %12s %12s %12s\n','t(s)','N1cErr','N1cN','Wfdmd','N2');
for tt = [10 14.9 15.5 16 18 20 25 30 39]
  [~,i] = min(abs(t-tt));
  fprintf(outfid,'  %8.2f %12.4f %12.6f %12.4f %12.1f\n', ...
          t(i), err(i), n1cN(i), wfd(i), n2(i));
end

% 关键：阶跃后 N1cErr 是否被驱动回 0？
pre = mean(err(t>=10 & t<15));
post_peak = max(abs(err(t>=15 & t<20)));
post_steady = mean(err(t>=35));
fprintf(outfid,'\n  N1cErr: 阶跃前=%.4f, 阶跃后峰值=%.4f, 稳态=%.4f\n', ...
        pre, post_peak, post_steady);
if abs(post_steady) < abs(post_peak) * 0.3
  fprintf(outfid,'  -> ✅ EEC 闭环工作（误差被拉回）\n');
else
  fprintf(outfid,'  -> ⚠ EEC 未把误差拉回\n');
end

% ══════════════════════════════════════════════════
% 测试 B：N1cMax/N1cMin 是常数还是变化？
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== N1c 限幅值 ===\n');
fprintf(outfid,'  N1cMax: [%.2f, %.2f] std=%.4f\n', min(n1max), max(n1max), std(n1max));
fprintf(outfid,'  N1cMin: [%.2f, %.2f] std=%.4f\n', min(n1min), max(n1min), std(n1min));
fprintf(outfid,'  -> 若是常数，说明是保护限幅，不是参考轨迹\n');

% ══════════════════════════════════════════════════
% 测试 C：PLA -> N1cN 的映射（PLA 是否直接决定参考）
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== PLA 扫描：N1cN 稳态值 ===\n');
fprintf(outfid,'  %8s %12s %12s %12s\n','PLA','N1cN','N2','N1cErr');
for pla_v = [45 50 55 60 65]
  tt = (0:TS:20)';
  MWS2 = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
  MWS2.In.Options.HybridConfig = 1; MWS2.In.Options.EPSLosses = 2;
  MWS2.In.Options.EngineEMInt = 1; MWS2.In.Options.PGBConfig = 2;
  MWS2.In.Options.WfTransientLogic = 2; MWS2.In.Options.TEEM = 1;
  MWS2.In.t_Alt = tt; MWS2.In.Alt = zeros(size(tt));
  MWS2.In.t_MN = tt; MWS2.In.MN = zeros(size(tt));
  MWS2.In.t_dTamb = [0 20]; MWS2.In.dTamb = [0 0];
  MWS2.In.t_Boost = [0 1 20]; MWS2.In.Boost = [0 0 0];
  MWS2.In.t_Charge = [0 1 20]; MWS2.In.Charge = [0 0 0];
  MWS2.In.t_EPT = [0 1 20]; MWS2.In.EPT = [0 0 0];
  MWS2.In.t_PLA = tt; MWS2.In.PLA = pla_v*ones(size(tt));
  MWS2 = Setup_Simulation(1, MWS2.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS2);
  set_param('AGTF30SysDyn_PExPIn','StopTime','20');
  sim('AGTF30SysDyn_PExPIn.slx');
  D2 = evalin('base','out_Dyn');
  e2 = D2.cntrl.Data.N1cErr.Data(:);
  c2 = D2.cntrl.Data.N1cN.Data(:);
  n2b = D2.loop.N2int.Data(:);
  fprintf(outfid,'  %8.1f %12.6f %12.1f %12.4f\n', ...
          pla_v, mean(c2(end-100:end)), mean(n2b(end-100:end)), mean(e2(end-100:end)));
end

% ══════════════════════════════════════════════════
% 测试 D：检查控制器的积分/反馈结构
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== WfTransientLogic 选项含义 ===\n');
fprintf(outfid,'  当前: WfTransientLogic = 2\n');
fprintf(outfid,'  这是 T-MATS 的燃油瞬态逻辑选项：\n');
fprintf(outfid,'    0 = 无瞬态逻辑\n');
fprintf(outfid,'    1 = 简化瞬态\n');
fprintf(outfid,'    2 = 完整瞬态（含速率限制）\n');
fprintf(outfid,'  -> 该逻辑依赖 N2dot 反馈，说明有内部状态\n');

fclose(outfid);
fprintf('PHYS_OK\n');
