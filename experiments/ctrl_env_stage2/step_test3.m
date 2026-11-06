% PLA -> EEC -> Wfdmd 因果链验证 + 阶跃响应（脚本版 v3）
%
% 关键修正：
%   1. sim 需要 base 工作区有名为 MWS 的变量（模型内部引用 MWS.In.Options.*）
%   2. 用 assignin('base','MWS',MWS) 而非 'MWS_run'
%   3. 每次 sim 前 close_system，避免锁

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

TEND = 30; TS = 0.015;
t = (0:TS:TEND)';
outfid = fopen(fullfile(outDir,'step_tests.txt'),'w','n','UTF-8');

% 关闭任何已打开模型
try, close_system('AGTF30SysDyn_PExPIn', 0); catch, end

% ══════════════════════════════════════════════════
% 基准 MWS 模板（只建一次）
% ══════════════════════════════════════════════════
fprintf('=== 建立基准 ===\n');
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
MWSbase = MWS;

% ══════════════════════════════════════════════════
% 测试 1：PLA 55 -> 57
% ══════════════════════════════════════════════════
fprintf('=== 测试 1: PLA 55 -> 57 ===\n');
MWS = MWSbase;
MWS.In.t_PLA = t;
MWS.In.PLA = 55*ones(size(t)); MWS.In.PLA(t >= 10) = 57;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');
D1 = evalin('base','out_Dyn');
wf1  = D1.cntrl.Dmd.Wfdmd.Data(:);
wfa1 = D1.act.Wfact.Data(:);
n21  = D1.loop.N2int.Data(:);
n31  = D1.loop.N3int.Data(:);

fprintf(outfid,'\n=== PLA 55->57 阶跃（t=10s）===\n');
fprintf(outfid,'  %-10s %11s %11s %11s %11s %11s\n','信号','阶跃前','+1s','+5s','稳态','变化');
for pair = {{'Wfdmd',wf1},{'Wfact',wfa1},{'N2int',n21},{'N3int',n31}}
  nm = pair{1}{1}; v = pair{1}{2};
  pre = mean(v(t>=8 & t<10));
  s1  = mean(v(t>=10.9 & t<11.1));
  s5  = mean(v(t>=14.9 & t<15.1));
  post = mean(v(t>=28));
  fprintf(outfid,'  %-10s %11.4f %11.4f %11.4f %11.4f %+11.4f\n', nm, pre, s1, s5, post, post-pre);
end

wf0 = mean(wf1(t>=8 & t<10)); wf_inf = mean(wf1(t>=28));
i_step = find(t>=10,1);
i63 = find(wf1(i_step:end) >= wf0+0.632*(wf_inf-wf0), 1);
i90 = find(wf1(i_step:end) >= wf0+0.9*(wf_inf-wf0), 1);
if ~isempty(i63), fprintf(outfid,'  Wfdmd 63%%响应: %.3f s\n', t(i_step+i63-1)-10); end
if ~isempty(i90), fprintf(outfid,'  Wfdmd 90%%响应: %.3f s\n', t(i_step+i90-1)-10); end

% ══════════════════════════════════════════════════
% 测试 2：PLA 55 -> 53
% ══════════════════════════════════════════════════
fprintf('=== 测试 2: PLA 55 -> 53 ===\n');
MWS = MWSbase;
MWS.In.t_PLA = t;
MWS.In.PLA = 55*ones(size(t)); MWS.In.PLA(t >= 10) = 53;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');
D2 = evalin('base','out_Dyn');
wf2  = D2.cntrl.Dmd.Wfdmd.Data(:);
n22  = D2.loop.N2int.Data(:);

fprintf(outfid,'\n=== PLA 55->53 阶跃 ===\n');
for pair = {{'Wfdmd',wf2},{'N2int',n22}}
  nm = pair{1}{1}; v = pair{1}{2};
  pre = mean(v(t>=8 & t<10)); post = mean(v(t>=28));
  fprintf(outfid,'  %-10s %.4f -> %.4f (%+.4f)\n', nm, pre, post, post-pre);
end

% ══════════════════════════════════════════════════
% 测试 3：PLA 55 -> 55.5（小幅）
% ══════════════════════════════════════════════════
fprintf('=== 测试 3: PLA 55 -> 55.5 ===\n');
MWS = MWSbase;
MWS.In.t_PLA = t;
MWS.In.PLA = 55*ones(size(t)); MWS.In.PLA(t >= 10) = 55.5;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');
D3 = evalin('base','out_Dyn');
wf3 = D3.cntrl.Dmd.Wfdmd.Data(:);
n23 = D3.loop.N2int.Data(:);

fprintf(outfid,'\n=== PLA 55->55.5 阶跃（+0.5 deg）===\n');
pre = mean(wf3(t>=8 & t<10)); post = mean(wf3(t>=28));
fprintf(outfid,'  Wfdmd: %.4f -> %.4f (%+.4f, %.4f%%)\n', pre, post, post-pre, (post-pre)/pre*100);
pre2 = mean(n23(t>=8 & t<10)); post2 = mean(n23(t>=28));
fprintf(outfid,'  N2int: %.2f -> %.2f (%+.2f rpm, %.4f%%)\n', pre2, post2, post2-pre2, (post2-pre2)/pre2*100);

% ══════════════════════════════════════════════════
% 测试 4：重复性
% ══════════════════════════════════════════════════
fprintf('=== 测试 4: 重复性 ===\n');
MWS = MWSbase;
MWS.In.t_PLA = t;
MWS.In.PLA = 55*ones(size(t)); MWS.In.PLA(t >= 10) = 57;
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');
D4 = evalin('base','out_Dyn');
wf4 = D4.cntrl.Dmd.Wfdmd.Data(:);

fprintf(outfid,'\n=== 重复性（同输入两次）===\n');
if numel(wf1)==numel(wf4)
  d = max(abs(wf1-wf4));
  fprintf(outfid,'  Wfdmd 最大差异: %.6e\n', d);
  if d < 1e-9
    fprintf(outfid,'  -> 完全可重复（确定性仿真）\n');
  elseif d < 1e-6
    fprintf(outfid,'  -> 高度可重复\n');
  else
    fprintf(outfid,'  -> 有差异\n');
  end
end

% ══════════════════════════════════════════════════
% 测试 5：PLA 扫描（静态增益+线性度）
% ══════════════════════════════════════════════════
fprintf('=== 测试 5: PLA 扫描 ===\n');
fprintf(outfid,'\n=== PLA -> Wfdmd/N2 静态增益 ===\n');
fprintf(outfid,'  %8s %12s %12s %12s\n','PLA','Wfdmd','N2int','N3int');
pla_list = [51 53 54 55 56 57 59];
res_wf = zeros(size(pla_list)); res_n2 = zeros(size(pla_list)); res_n3 = zeros(size(pla_list));
for k = 1:numel(pla_list)
  MWS = MWSbase;
  MWS.In.t_PLA = t;
  MWS.In.PLA = pla_list(k)*ones(size(t));
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);
  set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
  sim('AGTF30SysDyn_PExPIn.slx');
  Ds = evalin('base','out_Dyn');
  res_wf(k) = mean(Ds.cntrl.Dmd.Wfdmd.Data(:));
  res_n2(k) = mean(Ds.loop.N2int.Data(:));
  res_n3(k) = mean(Ds.loop.N3int.Data(:));
  fprintf(outfid,'  %8.1f %12.4f %12.2f %12.2f\n', pla_list(k), res_wf(k), res_n2(k), res_n3(k));
end

% 增益
fprintf(outfid,'\n=== 局部增益（相邻点差分）===\n');
fprintf(outfid,'  %12s %14s %14s\n','PLA区间','dWfdmd/dPLA','dN2/dPLA');
for k = 1:numel(pla_list)-1
  fprintf(outfid,'  %5.0f-%-6.0f %14.5f %14.3f\n', ...
    pla_list(k), pla_list(k+1), ...
    (res_wf(k+1)-res_wf(k))/(pla_list(k+1)-pla_list(k)), ...
    (res_n2(k+1)-res_n2(k))/(pla_list(k+1)-pla_list(k)));
end

fclose(outfid);
fprintf('PHYS_OK\n');
