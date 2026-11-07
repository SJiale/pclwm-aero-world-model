% B2 重跑（参考值已修正）
%
% 从 b2_ref.mat 读取经过单元测试的参考数组
% 若验证失败则拒绝启动

outDir = 'C:\pclwm\pclwm_stage3a\iiid';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));
addpath('C:\pclwm\pclwm_stage3a');

P = load('C:\pclwm\pclwm_stage3a\gru_pack2.mat');
MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105; H = 32;
T_SETTLE = 60;

% ══ 前置：加载并验证参考数组 ══
REFF = fullfile(outDir,'b2_ref.mat');
if ~isfile(REFF)
  error('参考文件不存在，先运行 b2_ref_unittest.m');
end
R = load(REFF);
n2_ref_arr = R.n2_ref;
t_step_arr = R.t_step;
N2_0 = R.N2_0; REF1 = R.REF1; REF2 = R.REF2;
N_CL = R.N_CL;

% ★ 再次验证（防止加载错误文件）
assert(abs(n2_ref_arr(1) - REF1) < 1e-6, '第1步参考错误');
assert(abs(n2_ref_arr(end) - REF2) < 1e-6, '末步参考错误');
assert(numel(unique(n2_ref_arr)) == 2, '参考值不是两个平台');
d = diff(n2_ref_arr);
assert(sum(abs(d) > 1e-6) == 1, '跳变数不是 1');
assert(abs(REF1 - REF2 - 500) < 1e-6, '下降量不是 500');
fprintf('✅ 参考数组二次验证通过\n');
fprintf('   REF1=%.2f, REF2=%.2f, N_CL=%d\n', REF1, REF2, N_CL);

T_END = T_SETTLE + (N_CL + H) * CTRL_DT;

% 动作约束（与上次 B2 相同，保证可比较）
ACT.pla_min = 48.0;
ACT.pla_max = 78.0;
ACT.dpla_max = 2.0;

fprintf('\n=== B2 重跑（参考已修正）===\n');
fprintf('  N2_0=%.2f\n', N2_0);
fprintf('  段1 0-6.3s:   %.2f (+750)\n', REF1);
fprintf('  段2 6.3-12.6s: %.2f (-500)\n', REF2);
fprintf('  %d 步, PLA∈[%.0f,%.0f], |dPLA|<=%.0f\n', N_CL, ...
        ACT.pla_min, ACT.pla_max, ACT.dpla_max);

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL,0); catch, end
outfid = fopen(fullfile(outDir,'b2_fixed.txt'),'w','n','UTF-8');
fprintf(outfid,'=== B2 (参考已修正) ===\n');
fprintf(outfid,'N2_0=%.2f  REF1=%.2f(+750)  REF2=%.2f(-500)\n', N2_0, REF1, REF2);
fprintf(outfid,'%d 步\n\n', N_CL);

% ══ MWS ══
t_full = (0:TS:T_END)';
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig=1; MWS.In.Options.EPSLosses=2;
MWS.In.Options.EngineEMInt=1; MWS.In.Options.PGBConfig=2;
MWS.In.Options.WfTransientLogic=2; MWS.In.Options.TEEM=1;
MWS.In.t_Alt=t_full; MWS.In.Alt=zeros(size(t_full));
MWS.In.t_MN=t_full; MWS.In.MN=zeros(size(t_full));
MWS.In.t_dTamb=[0 T_END]; MWS.In.dTamb=[0 0];
MWS.In.t_Boost=[0 1 T_END]; MWS.In.Boost=[0 0 0];
MWS.In.t_Charge=[0 1 T_END]; MWS.In.Charge=[0 0 0];
MWS.In.t_EPT=[0 1 T_END]; MWS.In.EPT=[0 0 0];
MWS.In.t_PLA=t_full; MWS.In.PLA=55*ones(size(t_full));
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);

% 稳态
fprintf('=== 稳态 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState0 = simOut.get('xState'); assignin('base','xState',xState0);
n2s = mean(D.loop.N2int.Data(end-600:end));
obs_steady = [mean(D.act.Wfact.Data(end-600:end)), n2s, ...
  mean(D.eng.S4.Tt.Data(end-600:end)), ...
  mean(D.eng.S45.Tt.Data(end-600:end)), ...
  mean(D.eng.S36.Pt.Data(end-600:end))];
fprintf('  N2_steady=%.2f\n', n2s);

% MPC 配置（完全不变）
CFG = struct();
CFG.H=H; CFG.n_cand=32; CFG.n_iter=6; CFG.n_short=8; CFG.w_short=3.0;
CFG.w_n2=1.0; CFG.w_du=1.0; CFG.w_coast=0.0;
CFG.pla_min=ACT.pla_min; CFG.pla_max=ACT.pla_max; CFG.dpla_max=ACT.dpla_max;
CFG.warm_start=true; CFG.mu_prev=[];

% 闭环
fprintf('=== 闭环 %d 步（约 %.0f 分钟）===\n', N_CL, N_CL*11/60);
assignin('base','xState',xState0);
obs_hist = repmat(obs_steady, 32, 1);
act_hist = repmat([55,0,0], 32, 1);
pla_prev = 55.0; mu_prev = [];

log = zeros(N_CL, 10);
% [k, t, n2, ref, err, pla, wf, T45, wfMax, opt]
fprintf(outfid,'%4s %7s %9s %9s %9s %7s %8s %8s %8s %7s\n', ...
        'k','t','N2','N2ref','err','PLA','Wf','T45','wfMax','opt');

for k = 1:N_CL
  tk = t_step_arr(k);
  n2_ref = n2_ref_arr(k);

  CFG.mu_prev = mu_prev;
  [pla_seq, info] = mpc_opt(obs_hist, act_hist, n2_ref, pla_prev, CFG, P, k*17);
  pla_apply = pla_seq(1);
  mu_prev = info.mu;

  pla_apply = min(ACT.pla_max, max(ACT.pla_min, pla_apply));
  pla_apply = min(pla_prev+ACT.dpla_max, max(pla_prev-ACT.dpla_max, pla_apply));
  pla_prev = pla_apply;

  t0 = T_SETTLE + (k-1)*CTRL_DT;
  t1 = T_SETTLE + k*CTRL_DT;
  mask = t_full > t0 & t_full <= t1;
  MWS.In.PLA(mask) = pla_apply;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  set_param(MDL,'LoadInitialState','on','InitialState','xState');
  set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
  set_param(MDL,'SaveCompleteFinalSimState','on');
  set_param(MDL,'StartTime',num2str(t0),'StopTime',num2str(t1));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  Dk = simOut.get('out_Dyn');
  xState = simOut.get('xState'); assignin('base','xState',xState);

  wf = Dk.act.Wfact.Data(end);
  wfmax = Dk.cntrl.Data.WfMax.Data(end);
  T4 = Dk.eng.S4.Tt.Data(end);
  T45 = Dk.eng.S45.Tt.Data(end);
  obs_new = [wf, Dk.loop.N2int.Data(end), T4, T45, Dk.eng.S36.Pt.Data(end)];
  obs_hist = [obs_hist(2:end,:); obs_new];
  act_hist = [act_hist(2:end,:); [pla_apply,0,0]];

  log(k,:) = [k, tk, obs_new(2), n2_ref, obs_new(2)-n2_ref, ...
              pla_apply, wf, T45, wfmax, info.t_total_ms];
  if mod(k,10)==1 || k==60 || k==N_CL
    fprintf('  k=%3d t=%5.2f N2=%8.2f ref=%8.2f err=%+8.2f PLA=%6.2f\n', ...
            k, tk, obs_new(2), n2_ref, obs_new(2)-n2_ref, pla_apply);
  end
  fprintf(outfid,'%4d %7.3f %9.2f %9.2f %+9.2f %7.3f %8.4f %8.1f %8.4f %7.0f\n', ...
          k, tk, obs_new(2), n2_ref, obs_new(2)-n2_ref, pla_apply, wf, T45, wfmax, ...
          info.t_total_ms);
end

save(fullfile(outDir,'b2_fixed.mat'), 'log', 'n2_ref_arr', 't_step_arr', ...
     'N2_0', 'REF1', 'REF2', 'ACT');
fclose(outfid);
fprintf('\n  -> %s\n', fullfile(outDir,'b2_fixed.txt'));
fprintf('PHYS_OK\n');
