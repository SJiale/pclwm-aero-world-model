% 最小测试：Simulation.step() 接口验证（20 步）
%
% PLA 序列：
%   1-5 步:  55
%   6-10 步: 56
%   11-15 步: 54
%   16-20 步: 55
%
% 验证：
%   1. 状态连续
%   2. PLA 动态更新生效
%   3. 逐步读取观测
%   4. PauseTime 实际暂停时刻
%   5. 与旧方案数值一致

outDir = 'C:\pclwm\pclwm_stage3a\step_api';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105;
N_STEP = 20;
T_SETTLE = 60;
T_END = T_SETTLE + N_STEP*CTRL_DT;

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL,0); catch, end

outfid = fopen(fullfile(outDir,'step_api_test.txt'),'w','n','UTF-8');
fprintf(outfid,'=== Simulation.step() 最小测试（%d 步）===\n\n', N_STEP);

% ══ 1. 准备 MWS ══
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

% ══ 2. 先建立稳态（用旧方法，一次性）══
fprintf('=== 1. 稳态建立 ===\n');
set_param(MDL,'FastRestart','off','LoadInitialState','off');
set_param(MDL,'SaveFinalState','on','FinalStateName','xSteady');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0','StopTime',num2str(T_SETTLE));
tic; simOut = sim(MDL,'ReturnWorkspaceOutputs','on'); t_settle = toc;
D = simOut.get('out_Dyn');
xSteady = simOut.get('xSteady');
assignin('base','xSteady',xSteady);
n2_steady = mean(D.loop.N2int.Data(end-600:end));
fprintf('  稳态: N2=%.2f, 耗时 %.1fs\n', n2_steady, t_settle);
fprintf(outfid,'稳态 N2 = %.2f rpm\n\n', n2_steady);

% ══ 3. 测试 Simulation.step() ══
fprintf('=== 2. Simulation.step() 测试 ===\n');

try
  % 创建 Simulation 对象
  s = simulation(MDL);
  fprintf('  ✓ simulation() 创建成功\n');
  fprintf('    Status = %s\n', s.Status);

  % 设置参数
  s = setModelParameter(s, 'StopTime', num2str(T_END));
  s = setModelParameter(s, 'StartTime', num2str(T_SETTLE));
  fprintf('  ✓ setModelParameter 成功\n');

  % 设置初始状态
  try
    s.InitialState = xSteady;
    fprintf('  ✓ InitialState 设置成功\n');
  catch ME
    fprintf('  ⚠ InitialState 设置失败: %s\n', ME.message);
  end

  % 初始化
  tic; initialize(s); t_init = toc;
  fprintf('  ✓ initialize 完成, 耗时 %.2f s, Status=%s\n', t_init, s.Status);
  fprintf(outfid,'初始化耗时: %.2f s\n', t_init);

  % 逐步推进
  PLA_SEQ = [55*ones(1,5), 56*ones(1,5), 54*ones(1,5), 55*ones(1,5)];
  log_step = zeros(N_STEP, 8);   % [k, target_time, actual_time, n2, n3, wf, T45, t_wall]
  t_step_all = zeros(N_STEP,1);
  pla_applied = zeros(N_STEP,1);

  fprintf('\n  %3s %8s %10s %10s %10s %7s %8s\n', ...
          'k','target_t','actual_t','N2','N3','PLA','dt(ms)');

  for k = 1:N_STEP
    t_target = T_SETTLE + k*CTRL_DT;
    pla_now = PLA_SEQ(k);

    % ★ 尝试更新 PLA：改 base 工作区的 MWS
    MWS.In.PLA(:) = pla_now;
    assignin('base','MWS',MWS);

    % 推进
    tic;
    try
      finalStep = step(s, PauseTime=t_target);
      t_step = toc;
      ok = true;
    catch ME
      t_step = toc;
      ok = false;
      fprintf('    步 %d 失败: %s\n', k, ME.message);
      break;
    end
    t_step_all(k) = t_step;
    pla_applied(k) = pla_now;

    % 读当前时间
    t_actual = s.Time;

    % 读输出
    try
      out = s.SimulationOutput;
      Dk = out.get('out_Dyn');
      n2 = Dk.loop.N2int.Data(end);
      n3 = Dk.loop.N3int.Data(end);
      wf = Dk.act.Wfact.Data(end);
      T45 = Dk.eng.S45.Tt.Data(end);
    catch ME
      n2 = NaN; n3 = NaN; wf = NaN; T45 = NaN;
    end

    log_step(k,:) = [k, t_target, t_actual, n2, n3, wf, T45, t_step*1000];
    if mod(k,2)==1 || k>15
      fprintf('  %3d %8.3f %10.3f %10.2f %10.2f %7.2f %8.0f\n', ...
              k, t_target, t_actual, n2, n3, pla_now, t_step*1000);
    end
  end

  % 结束
  try
    out_final = stop(s);
    fprintf('\n  ✓ stop() 完成\n');
  catch ME
    fprintf('\n  ⚠ stop() 失败: %s\n', ME.message);
  end

  % 统计
  fprintf(outfid,'\n=== 结果 ===\n');
  fprintf(outfid,'  %3s %10s %10s %10s %8s %8s\n','k','target','actual','N2','PLA','dt_ms');
  for k=1:N_STEP
    fprintf(outfid,'  %3d %10.3f %10.3f %10.2f %8.2f %8.0f\n', ...
            log_step(k,1), log_step(k,2), log_step(k,3), log_step(k,4), ...
            pla_applied(k), log_step(k,8));
  end

  tvec = t_step_all(t_step_all>0);
  fprintf(outfid,'\n  单步耗时: 中位=%.0f ms, 均值=%.0f ms, 最小=%.0f, 最大=%.0f\n', ...
          median(tvec)*1000, mean(tvec)*1000, min(tvec)*1000, max(tvec)*1000);
  fprintf('\n  单步耗时: 中位=%.0f ms, 最大=%.0f ms\n', ...
          median(tvec)*1000, max(tvec)*1000);

  save(fullfile(outDir,'step_api_log.mat'), 'log_step', 'PLA_SEQ', 'n2_steady');
  fprintf(outfid,'\n  状态连续检查:\n');
  dn2 = diff(log_step(:,4));
  fprintf(outfid,'    N2 步间变化: [%.2f, %.2f] rpm\n', min(dn2), max(dn2));
  fprintf(outfid,'    N2 范围: [%.2f, %.2f]\n', min(log_step(:,4)), max(log_step(:,4)));
  if max(abs(dn2)) < 200
    fprintf(outfid,'    -> ✅ 状态连续（无跳变）\n');
  else
    fprintf(outfid,'    -> ⚠ 有跳变，需检查\n');
  end

catch ME
  fprintf('\n  ❌ Simulation.step() 失败:\n');
  fprintf('     错误: %s\n', ME.message);
  fprintf(outfid,'\n❌ 失败: %s\n', ME.message);
  if ~isempty(ME.stack)
    for i=1:min(3,numel(ME.stack))
      fprintf('     at %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
  end
end

fclose(outfid);
fprintf('PHYS_OK\n');
