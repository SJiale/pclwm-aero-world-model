% 关键测试：用 setBlockParameter 动态更新 PLA
%
% PLA 模块: AGTF30SysDyn_PExPIn/System Inputs/Model Source4
% Mask 参数: T_vec_M = 'MWS.In.t_PLA', input_array_M = 'MWS.In.PLA'
%
% 这是字符串引用，指向 base 工作区的 MWS 变量
% 关键问题：Simulation 对象初始化后，改 MWS 能否生效？
%
% 三种方案：
%   A. 改 base 的 MWS（之前的做法，失败）
%   B. setBlockParameter 改 input_array_M 为直接数值
%   C. setVariable 设置 MWS

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105; T_SETTLE = 60;
outDir = 'C:\pclwm\pclwm_stage3a\step_api';
try, close_system(MDL,0); catch, end

fid = fopen(fullfile(outDir,'setblockparam_test.txt'),'w','n','UTF-8');
fprintf(fid,'=== setBlockParameter 动态更新 PLA 测试 ===\n\n');

load_system(MDL);
PLA_BLK = [MDL '/System Inputs/Model Source4'];
fprintf(fid,'PLA 模块: %s\n', PLA_BLK);
fprintf(fid,'  T_vec_M     = %s\n', get_param(PLA_BLK,'T_vec_M'));
fprintf(fid,'  input_array_M = %s\n', get_param(PLA_BLK,'input_array_M'));

% 准备 MWS
T_END = T_SETTLE + 12*CTRL_DT;
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
fprintf(fid,'\n=== 稳态建立 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xState');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
D = simOut.get('out_Dyn');
xState = simOut.get('xState'); assignin('base','xState',xState);
n2_s = mean(D.loop.N2int.Data(end-600:end));
fprintf(fid,'  N2_steady = %.2f\n', n2_s);
fprintf('  稳态 N2 = %.2f\n', n2_s);

% ══ 测试计划 ══
% 5 步: PLA=55 (基准)
% 5 步: PLA=60 (大幅升，应该看到 N2 上升)
% 5 步: PLA=50 (大幅降)
N_STEP = 15;
PLA_SEQ = [55*ones(1,5), 60*ones(1,5), 50*ones(1,5)];

fprintf(fid,'\n=== 测试序列 ===\n');
fprintf(fid,'  步 1-5:   PLA=55\n');
fprintf(fid,'  步 6-10:  PLA=60 (升)\n');
fprintf(fid,'  步 11-15: PLA=50 (降)\n\n');

try
  s = simulation(MDL);
  s = setModelParameter(s, 'StopTime', num2str(T_END));
  s = setModelParameter(s, 'StartTime', num2str(T_SETTLE));
  s.InitialState = xState;

  tic; initialize(s); t_init = toc;
  fprintf(fid,'初始化: %.2f s\n\n', t_init);
  fprintf('  初始化 %.2f s\n', t_init);

  fprintf(fid,'  %3s %10s %8s %10s %10s %10s %8s\n', ...
          'k','t','PLA','N2','N3','Wf','dt_ms');
  fprintf('  %3s %10s %8s %10s %10s %10s %8s\n', ...
          'k','t','PLA','N2','N3','Wf','dt_ms');

  log = zeros(N_STEP, 7);
  t_all = zeros(N_STEP,1);

  for k = 1:N_STEP
    t_tgt = T_SETTLE + k*CTRL_DT;
    pla_now = PLA_SEQ(k);

    % ★ 方案 B：setBlockParameter 直接给数值
    tic;
    % 构造从当前时刻开始的 PLA 剖面（后续保持该值）
    t_rem = t_full(t_full >= T_SETTLE) - T_SETTLE;
    % 简单做法：给一个常数
    try
      s = setBlockParameter(s, PLA_BLK, 'input_array_M', pla_now);
      try
        s = setBlockParameter(s, PLA_BLK, 'T_vec_M', [T_SETTLE T_END]);
      catch
      end
    catch ME
      if k==1
        fprintf(fid,'  setBlockParameter 失败: %s\n', ME.message);
      end
    end
    t_setp = toc;

    % 推进
    tic;
    try
      step(s, PauseTime=t_tgt);
      t_step = toc;
    catch ME
      fprintf(fid,'  步 %d step 失败: %s\n', k, ME.message);
      break;
    end
    t_all(k) = t_setp + t_step;

    t_act = s.Time;
    try
      out = s.SimulationOutput;
      Dk = out.get('out_Dyn');
      n2 = Dk.loop.N2int.Data(end);
      n3 = Dk.loop.N3int.Data(end);
      wf = Dk.act.Wfact.Data(end);
    catch
      n2=NaN; n3=NaN; wf=NaN;
    end
    log(k,:) = [k, t_act, pla_now, n2, n3, wf, t_all(k)*1000];
    fprintf(fid,'  %3d %10.3f %8.2f %10.2f %10.2f %10.4f %8.0f\n', ...
            k, t_act, pla_now, n2, n3, wf, t_all(k)*1000);
    fprintf('  %3d %10.3f %8.2f %10.2f %10.2f %10.4f %8.0f\n', ...
            k, t_act, pla_now, n2, n3, wf, t_all(k)*1000);
  end

  try, stop(s); catch, end

  % 分析
  fprintf(fid,'\n=== 分析 ===\n');
  seg1 = log(1:5, 4);   % PLA=55
  seg2 = log(6:10, 4);  % PLA=60
  seg3 = log(11:15, 4); % PLA=50
  fprintf(fid,'  PLA=55: N2 = [%.2f, %.2f], 变化 %.2f\n', ...
          min(seg1), max(seg1), max(seg1)-min(seg1));
  fprintf(fid,'  PLA=60: N2 = [%.2f, %.2f], 变化 %.2f\n', ...
          min(seg2), max(seg2), max(seg2)-min(seg2));
  fprintf(fid,'  PLA=50: N2 = [%.2f, %.2f], 变化 %.2f\n', ...
          min(seg3), max(seg3), max(seg3)-min(seg3));
  fprintf(fid,'\n  PLA=60 时 N2 是否上升: %s\n', ...
          ternary(mean(seg2) > mean(seg1), '✅ 是', '❌ 否'));
  fprintf(fid,'  PLA=50 时 N2 是否下降: %s\n', ...
          ternary(mean(seg3) < mean(seg2), '✅ 是', '❌ 否'));

  tv = t_all(t_all>0);
  fprintf(fid,'\n  单步耗时: 中位=%.0f ms, 最大=%.0f ms\n', ...
          median(tv)*1000, max(tv)*1000);
  fprintf('  单步: 中位 %.0f ms\n', median(tv)*1000);

  save(fullfile(outDir,'setblock_log.mat'), 'log', 'PLA_SEQ', 'n2_s');

catch ME
  fprintf(fid,'\n❌ 失败: %s\n', ME.message);
  fprintf('  ❌ %s\n', ME.message);
end

fclose(fid);
fprintf('PHYS_OK\n');

function s = ternary(c,a,b)
if c, s=a; else, s=b; end
end

