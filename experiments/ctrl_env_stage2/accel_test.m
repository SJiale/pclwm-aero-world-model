% Fast Restart + 消除 Setup 的加速验证
%
% 已知：[1] Setup=3.53s, [2] sim 固定开销=8.22s, 数值积分≈0
% 目标：验证能否消除固定开销

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

outfid = fopen(fullfile(outDir,'accel.txt'),'w','n','UTF-8');
MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; CTRL_DT = 0.105;

try, close_system(MDL, 0); catch, end

T_TMPL = 1.05;
t_tmpl = (0:TS:T_TMPL)';
MWS0 = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS0.In.Options.HybridConfig = 1; MWS0.In.Options.EPSLosses = 2;
MWS0.In.Options.EngineEMInt = 1; MWS0.In.Options.PGBConfig = 2;
MWS0.In.Options.WfTransientLogic = 2; MWS0.In.Options.TEEM = 1;
MWS0.In.t_Alt = t_tmpl; MWS0.In.Alt = zeros(size(t_tmpl));
MWS0.In.t_MN = t_tmpl; MWS0.In.MN = zeros(size(t_tmpl));
MWS0.In.t_dTamb = [0 T_TMPL]; MWS0.In.dTamb = [0 0];
MWS0.In.t_Boost = [0 T_TMPL/2 T_TMPL]; MWS0.In.Boost = [0 0 0];
MWS0.In.t_Charge = [0 T_TMPL/2 T_TMPL]; MWS0.In.Charge = [0 0 0];
MWS0.In.t_EPT = [0 T_TMPL/2 T_TMPL]; MWS0.In.EPT = [0 0 0];
MWS0.In.t_PLA = t_tmpl; MWS0.In.PLA = 55*ones(size(t_tmpl));
InTemplate = MWS0.In;

fprintf(outfid,'=== 加速验证 ===\n\n');

% ══════════════════════════════════════════════════
% [A] 基线：无 FastRestart，每次 SetStart/Stop
% ══════════════════════════════════════════════════
Mtmp = Setup_Simulation(1, InTemplate, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',Mtmp);
set_param(MDL,'FastRestart','off');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));

N=10;
tA = zeros(N,1);
for k=1:N
  ts=(k-1)*CTRL_DT; te=k*CTRL_DT;
  if k==1, set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(ts));
  set_param(MDL,'StopTime',num2str(te));
  tic; sim(MDL); tA(k)=toc;
end
fprintf(outfid,'[A] 无 FastRestart（含状态传递）\n');
fprintf(outfid,'    median=%.3f s\n', median(tA));

% ══════════════════════════════════════════════════
% [B] Fast Restart（正确顺序：先 on，再 sim）
% ══════════════════════════════════════════════════
set_param(MDL,'FastRestart','off');
set_param(MDL,'LoadInitialState','off');
set_param(MDL,'SaveFinalState','on');
set_param(MDL,'FinalStateName','xFinal');
set_param(MDL,'SaveCompleteFinalSimState','on');
set_param(MDL,'StartTime','0');
set_param(MDL,'StopTime',num2str(CTRL_DT));

% 首次 sim 会触发编译
tic; sim(MDL); t_compile = toc;
fprintf(outfid,'\n[B] FastRestart\n');
fprintf(outfid,'    首次编译 sim: %.3f s\n', t_compile);

% 开启 FastRestart（之后 sim 复用编译结果）
set_param(MDL,'FastRestart','on');

tB = zeros(N,1); n2B = zeros(N,1);
for k=1:N
  ts=(k-1)*CTRL_DT; te=k*CTRL_DT;
  if k==1
    set_param(MDL,'LoadInitialState','off');
  else
    set_param(MDL,'LoadInitialState','on');
    set_param(MDL,'InitialState','xFinal');
  end
  set_param(MDL,'StartTime',num2str(ts));
  set_param(MDL,'StopTime',num2str(te));
  tic; sim(MDL); tB(k)=toc;
  try
    D = evalin('base','out_Dyn');
    n2B(k) = D.loop.N2int.Data(end);
  catch
  end
end
set_param(MDL,'FastRestart','off');

fprintf(outfid,'    每步 median=%.3f s, mean=%.3f s\n', median(tB), mean(tB));
fprintf(outfid,'    加速比: %.2fx\n', median(tA)/median(tB));

% ══════════════════════════════════════════════════
% [C] 检查 FastRestart 是否真的生效
% ══════════════════════════════════════════════════
fprintf(outfid,'\n[C] Fast Restart 完整性检查\n');
fr = get_param(MDL,'FastRestart');
fprintf(outfid,'    FastRestart 状态: %s\n', fr);

% ══════════════════════════════════════════════════
% 汇总
% ══════════════════════════════════════════════════
fprintf(outfid,'\n=== 汇总 ===\n');
fprintf(outfid,'  A. 无 FastRestart:  %.3f s/步\n', median(tA));
fprintf(outfid,'  B. 有 FastRestart:  %.3f s/步\n', median(tB));
fprintf(outfid,'  控制周期:           %.3f s\n', CTRL_DT);
fprintf(outfid,'\n  实时倍率:\n');
fprintf(outfid,'    A: %.1fx\n', median(tA)/CTRL_DT);
fprintf(outfid,'    B: %.1fx\n', median(tB)/CTRL_DT);

if median(tB) < CTRL_DT
  fprintf(outfid,'\n  ✅ FastRestart 后满足实时\n');
else
  fprintf(outfid,'\n  ❌ 仍不满足实时\n');
end

fclose(outfid);
fprintf('PHYS_OK\n');
