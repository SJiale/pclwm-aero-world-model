% 诊断单步仿真失败原因：Input_profiles 插值要求
%
% 错误：InputValues 在转换为其运行时数据类型之前和之后必须单调递增
% 推测：0.015s 步长在单精度下精度不足，导致插值点重复

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'step_probe.txt'),'w','n','UTF-8');

% ── 检查 Setup_Simulation 对时间向量的处理 ──
fprintf(outfid,'=== 时间精度诊断 ===\n');

% 不同 TEND 下的相邻差
for TEND = [0.105 0.12 0.2 0.3 1.0 3.36]
  t = (0:0.015:TEND)';
  d = diff(t);
  % 检查单精度转换后是否仍严格递增
  t_single = single(t);
  d_single = diff(t_single);
  n_dup = sum(d_single <= 0);
  fprintf(outfid,'  TEND=%-6.3f  n=%-4d  双精度最小差=%.3e  单精度重复点数=%d\n', ...
          TEND, numel(t), min(d), n_dup);
end

% ── 检查 Input_profiles 块要求 ──
fprintf(outfid,'\n=== Input_profiles 块参数 ===\n');
try
  blk = 'AGTF30SysDyn_PExPIn/System Inputs/Model Source24/Input_profiles';
  fprintf(outfid,'  块类型: %s\n', get_param(blk,'BlockType'));
  bp = get_param(blk,'DialogParameters');
  f = fieldnames(bp);
  for i=1:numel(f)
    try
      v = get_param(blk, f{i});
      if isnumeric(v) || ischar(v) || islogical(v)
        vs = mat2str(v);
        if numel(vs) > 100, vs = [vs(1:100) '...']; end
        fprintf(outfid,'    %s = %s\n', f{i}, vs);
      end
    catch
    end
  end
catch ME
  fprintf(outfid,'  读取失败: %s\n', ME.message);
end

% ── 尝试不同的时间步长 ──
fprintf(outfid,'\n=== 不同步长尝试（TEND=0.105）===\n');
for TS = [0.015 0.005 0.001]
  t = (0:TS:0.105)';
  try
    MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
    MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
    MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
    MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
    MWS.In.t_Alt = t; MWS.In.Alt = zeros(size(t));
    MWS.In.t_MN = t; MWS.In.MN = zeros(size(t));
    MWS.In.t_dTamb = [0 0.105]; MWS.In.dTamb = [0 0];
    MWS.In.t_Boost = [0 0.0525 0.105]; MWS.In.Boost = [0 0 0];
    MWS.In.t_Charge = [0 0.0525 0.105]; MWS.In.Charge = [0 0 0];
    MWS.In.t_EPT = [0 0.0525 0.105]; MWS.In.EPT = [0 0 0];
    MWS.In.t_PLA = t; MWS.In.PLA = 55*ones(size(t));
    MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
    assignin('base','MWS',MWS);
    set_param('AGTF30SysDyn_PExPIn','StopTime','0.105');
    tic; sim('AGTF30SysDyn_PExPIn.slx'); el=toc;
    fprintf(outfid,'  TS=%.4f (n=%d): ✅ 成功 %.3f s\n', TS, numel(t), el);
  catch ME
    msg = ME.message;
    if numel(msg)>60, msg=msg(1:60); end
    fprintf(outfid,'  TS=%.4f (n=%d): ❌ %s\n', TS, numel(t), msg);
  end
end

% ── 尝试用 sim 的 StartTime/StopTime ──
fprintf(outfid,'\n=== 用 sim 的 StartTime/StopTime（不用 Setup 重建）===\n');
% 先建一个 60s 标准配置
TEND = 60; t = (0:0.015:TEND)';
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
MWS.In.t_PLA = t; MWS.In.PLA = 55*ones(size(t));
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);

% 试 StartTime=0.9, StopTime=1.005（避开 0 附近的初始化）
for st = [0 0.9 1.0]
  try
    set_param('AGTF30SysDyn_PExPIn','StartTime',num2str(st));
    set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(st+0.105));
    tic; sim('AGTF30SysDyn_PExPIn.slx'); el=toc;
    fprintf(outfid,'  StartTime=%.3f: ✅ %.3f s\n', st, el);
  catch ME
    msg = ME.message;
    if numel(msg)>60, msg=msg(1:60); end
    fprintf(outfid,'  StartTime=%.3f: ❌ %s\n', st, msg);
  end
end
set_param('AGTF30SysDyn_PExPIn','StartTime','0');

fclose(outfid);
fprintf('PHYS_OK\n');
