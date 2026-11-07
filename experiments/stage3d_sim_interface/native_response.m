% 原生控制器（EEC）的转速跟踪响应
%
% 目的：与 MPC 对比
% 场景：PLA 55 -> 63.64 阶跃，观察 EEC 如何把 N2 跟踪到 5099
%
% 注意：这是【开环 PLA 阶跃 + EEC 内环】的响应
%       不是"上层闭环跟踪 N2_ref"——原生 AGTF30e 没有 N2 参考跟踪外环
%
% 两种情况都测：
%   A. PLA 阶跃 55 -> 63.64（EEC 当作内环）
%   B. 慢速 PLA 斜坡（模拟"手调油门"）
%
% 一次仿真即可，成本低

outDir = 'C:\pclwm\pclwm_stage3a\compare';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015;
TEND = 75;                 % 0-60 稳态，60-75 阶跃观测
T_STEP = 60;
t = (0:TS:TEND)';

PLA0 = 55.0;
PLA1 = 63.64;              % 插值得到的 5099 rpm 对应值
N2_TARGET = 5099.29;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL,0); catch, end

fid = fopen(fullfile(outDir,'native_response.txt'),'w','n','UTF-8');
fprintf(fid,'=== 原生 EEC 响应测试 ===\n');
fprintf(fid,'  PLA %.2f -> %.2f @ t=%.0fs\n', PLA0, PLA1, T_STEP);
fprintf(fid,'  目标 N2 = %.2f rpm\n\n', N2_TARGET);

% ══════════════════════════════════════════════════
% A. 阶跃响应
% ══════════════════════════════════════════════════
fprintf('=== A. PLA 阶跃响应 ===\n');
try, close_system(MDL,0); catch, end

MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig=1; MWS.In.Options.EPSLosses=2;
MWS.In.Options.EngineEMInt=1; MWS.In.Options.PGBConfig=2;
MWS.In.Options.WfTransientLogic=2; MWS.In.Options.TEEM=1;
MWS.In.t_Alt=t; MWS.In.Alt=zeros(size(t));
MWS.In.t_MN=t; MWS.In.MN=zeros(size(t));
MWS.In.t_dTamb=[0 TEND]; MWS.In.dTamb=[0 0];
MWS.In.t_Boost=[0 1 TEND]; MWS.In.Boost=[0 0 0];
MWS.In.t_Charge=[0 1 TEND]; MWS.In.Charge=[0 0 0];
MWS.In.t_EPT=[0 1 TEND]; MWS.In.EPT=[0 0 0];

% 阶跃剖面
pla_step = PLA0*ones(size(t));
pla_step(t >= T_STEP) = PLA1;
MWS.In.t_PLA=t; MWS.In.PLA=pla_step;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);

set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','off');
set_param(MDL,'StartTime','0','StopTime',num2str(TEND));
tic; simOut = sim(MDL,'ReturnWorkspaceOutputs','on'); tA = toc;
DA = simOut.get('out_Dyn');

n2A = DA.loop.N2int.Data(:);
n3A = DA.loop.N3int.Data(:);
wfA = DA.act.Wfact.Data(:);
wfdmdA = DA.cntrl.Dmd.Wfdmd.Data(:);
wfmaxA = DA.cntrl.Data.WfMax.Data(:);
y1A = DA.cntrl.Data.N1cN.Data(:);
errA = DA.cntrl.Data.N1cErr.Data(:);
T4A = DA.eng.S4.Tt.Data(:);
T45A = DA.eng.S45.Tt.Data(:);
P36A = DA.eng.S36.Pt.Data(:);

fprintf('  仿真完成 %.1f s\n', tA);

% 阶跃后分析
m = t >= T_STEP;
tt = t(m) - T_STEP;   % 相对阶跃时刻
n2 = n2A(m);
n2_0 = mean(n2A(t>=T_STEP-5 & t<T_STEP));
n2_end = mean(n2A(t>=TEND-3));

% 上升时间
i10 = find(n2 >= n2_0 + 0.1*(n2_end-n2_0), 1);
i90 = find(n2 >= n2_0 + 0.9*(n2_end-n2_0), 1);
tr = tt(i90) - tt(i10);

fprintf(fid,'  阶跃前 N2 = %.2f\n', n2_0);
fprintf(fid,'  稳态 N2 = %.2f\n', n2_end);
fprintf(fid,'  10%%-90%% 上升时间 = %.2f s\n', tr);
fprintf(fid,'  超调 = %.2f rpm\n', max(0, max(n2) - n2_end));
fprintf(fid,'  15s 末误差 = %.2f rpm\n', n2(end) - n2_end);
fprintf(fid,'  Wf 最大 = %.4f (上限 %.4f)\n', max(wfA(m)), wfmaxA(1));
fprintf(fid,'  Wf 饱和点 = %d / %d\n', sum(wfA(m) >= wfmaxA(m)*0.999), sum(m));
fprintf(fid,'  T45 最大 = %.1f R\n', max(T45A(m)));
fprintf(fid,'  N1cErr 范围 = [%.2f, %.2f]\n', min(errA(m)), max(errA(m)));

fprintf('\n  阶跃前 N2 = %.2f\n', n2_0);
fprintf('  稳态 N2   = %.2f\n', n2_end);
fprintf('  上升时间  = %.2f s\n', tr);
fprintf('  超调      = %.2f rpm\n', max(0, max(n2) - n2_end));

% ══════════════════════════════════════════════════
% B. 慢速斜坡（对比）
% ══════════════════════════════════════════════════
fprintf('\n=== B. PLA 慢速斜坡（5s 内线性）===\n');
try, close_system(MDL,0); catch, end

MWS2 = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS2.In.Options.HybridConfig=1; MWS2.In.Options.EPSLosses=2;
MWS2.In.Options.EngineEMInt=1; MWS2.In.Options.PGBConfig=2;
MWS2.In.Options.WfTransientLogic=2; MWS2.In.Options.TEEM=1;
MWS2.In.t_Alt=t; MWS2.In.Alt=zeros(size(t));
MWS2.In.t_MN=t; MWS2.In.MN=zeros(size(t));
MWS2.In.t_dTamb=[0 TEND]; MWS2.In.dTamb=[0 0];
MWS2.In.t_Boost=[0 1 TEND]; MWS2.In.Boost=[0 0 0];
MWS2.In.t_Charge=[0 1 TEND]; MWS2.In.Charge=[0 0 0];
MWS2.In.t_EPT=[0 1 TEND]; MWS2.In.EPT=[0 0 0];

T_RAMP = 5;
pla_ramp = PLA0*ones(size(t));
mask_ramp = t >= T_STEP & t < T_STEP+T_RAMP;
pla_ramp(mask_ramp) = PLA0 + (PLA1-PLA0)*(t(mask_ramp)-T_STEP)/T_RAMP;
pla_ramp(t >= T_STEP+T_RAMP) = PLA1;
MWS2.In.t_PLA=t; MWS2.In.PLA=pla_ramp;
MWS2 = Setup_Simulation(1, MWS2.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS2);

set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','off');
set_param(MDL,'StartTime','0','StopTime',num2str(TEND));
tic; simOut = sim(MDL,'ReturnWorkspaceOutputs','on'); tB = toc;
DB = simOut.get('out_Dyn');
n2B = DB.loop.N2int.Data(:);
fprintf('  仿真完成 %.1f s\n', tB);
n2B_0 = mean(n2B(t>=T_STEP-5 & t<T_STEP));
n2B_end = mean(n2B(t>=TEND-3));
fprintf(fid,'\n  斜坡: 阶跃前 %.2f, 稳态 %.2f\n', n2B_0, n2B_end);
fprintf('  斜坡: 阶跃前 %.2f, 稳态 %.2f\n', n2B_0, n2B_end);

% ══════════════════════════════════════════════════
% 保存
% ══════════════════════════════════════════════════
save(fullfile(outDir,'native_response.mat'), ...
     't', 'n2A', 'n3A', 'wfA', 'wfdmdA', 'wfmaxA', 'y1A', 'errA', ...
     'T4A', 'T45A', 'P36A', 'tA', 'pla_step', ...
     'n2B', 'tB', 'pla_ramp', ...
     'PLA0', 'PLA1', 'N2_TARGET', 'T_STEP', 'n2_0', 'n2_end', 'tr');

fclose(fid);
fprintf('PHYS_OK\n');
