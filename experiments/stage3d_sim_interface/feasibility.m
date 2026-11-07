% 可行性检查：PLA 工作点 -> 可实现稳态 N2 范围
%
% 关键：不能用数据集 PLA 范围 [40,80] 当作安全控制范围
% 必须检查：燃油限幅、转速、温度、EEC 状态

outDir = 'C:\pclwm\pclwm_stage3a\iiid';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015;
TEND = 60;          % 每个工作点跑 60s 到稳态
t = (0:TS:TEND)';

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

outfid = fopen(fullfile(outDir,'feasibility.txt'),'w','n','UTF-8');
fprintf(outfid,'=== PLA -> 稳态 N2 可达范围检查 ===\n');
fprintf(outfid,'  每个工作点仿真 %.0f s\n\n', TEND);
fprintf(outfid,'  %8s %10s %10s %10s %10s %10s %10s %8s\n', ...
        'PLA','N2','N3','Wfdmd','T4(R)','T45(R)','P36','饱和');

% 扫描范围：从 40 到 80（数据集范围），看哪些能稳态
PLA_LIST = [40 45 50 52 55 58 60 62 65 70 75 80];
RESULTS = [];

for i = 1:numel(PLA_LIST)
  pla_v = PLA_LIST(i);
  try, close_system(MDL,0); catch, end

  MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
  MWS.In.Options.HybridConfig=1; MWS.In.Options.EPSLosses=2;
  MWS.In.Options.EngineEMInt=1; MWS.In.Options.PGBConfig=2;
  MWS.In.Options.WfTransientLogic=2; MWS.In.Options.TEEM=1;
  MWS.In.t_Alt = t; MWS.In.Alt = zeros(size(t));
  MWS.In.t_MN = t; MWS.In.MN = zeros(size(t));
  MWS.In.t_dTamb = [0 TEND]; MWS.In.dTamb = [0 0];
  MWS.In.t_Boost = [0 1 TEND]; MWS.In.Boost = [0 0 0];
  MWS.In.t_Charge = [0 1 TEND]; MWS.In.Charge = [0 0 0];
  MWS.In.t_EPT = [0 1 TEND]; MWS.In.EPT = [0 0 0];
  MWS.In.t_PLA = t; MWS.In.PLA = pla_v*ones(size(t));
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
  assignin('base','MWS',MWS);

  set_param(MDL,'FastRestart','off','LoadInitialState','off');
  set_param(MDL,'SaveFinalState','off');
  set_param(MDL,'StartTime','0','StopTime',num2str(TEND));
  simOut = sim(MDL,'ReturnWorkspaceOutputs','on');
  D = simOut.get('out_Dyn');

  n2 = D.loop.N2int.Data(:);
  n3 = D.loop.N3int.Data(:);
  wfd = D.cntrl.Dmd.Wfdmd.Data(:);
  wfmax = D.cntrl.Data.WfMax.Data(:);
  wfmin = D.cntrl.Data.WfMin.Data(:);
  n1max = D.cntrl.Data.N1cMax.Data(:);
  n1min = D.cntrl.Data.N1cMin.Data(:);
  n1err = D.cntrl.Data.N1cErr.Data(:);
  T4 = D.eng.S4.Tt.Data(:);
  T45 = D.eng.S45.Tt.Data(:);
  P36 = D.eng.S36.Pt.Data(:);

  % 稳态：最后 10s
  m = t >= (TEND-10);
  n2s = mean(n2(m)); n3s = mean(n3(m)); wfs = mean(wfd(m));
  t4s = mean(T4(m)); t45s = mean(T45(m)); p36s = mean(P36(m));
  n1errs = mean(n1err(m));

  % 饱和检测
  sat_wf_hi = sum(wfd >= wfmax*0.999);
  sat_wf_lo = sum(wfd <= wfmin*1.001);
  sat_n1_hi = sum(n3 >= n1max*0.999);   % 注意 N1c 对应哪路，先都看
  sat = sprintf('Wf:%d/%d', sat_wf_hi, sat_wf_lo);

  % 数值有效性
  ok = isfinite(n2s) && n2s > 1000 && n2s < 12000 && abs(n1errs) < 50;

  fprintf(outfid,'  %8.1f %10.2f %10.2f %10.4f %10.1f %10.1f %10.2f %8s\n', ...
          pla_v, n2s, n3s, wfs, t4s, t45s, p36s, sat);
  RESULTS(end+1,:) = [pla_v, n2s, n3s, wfs, t4s, t45s, p36s, ok];

  fprintf('  PLA=%5.1f -> N2=%8.2f  Wf=%.4f  T45=%7.1f  %s\n', ...
          pla_v, n2s, wfs, t45s, ternary(ok,'OK','FAIL'));
end

fprintf(outfid,'\n=== 可达范围分析 ===\n');
ok_mask = RESULTS(:,8) > 0;
if any(ok_mask)
  ok_res = RESULTS(ok_mask, :);
  [n2min, imin] = min(ok_res(:,2));
  [n2max, imax] = max(ok_res(:,2));
  fprintf(outfid,'  可达稳态 N2: [%.2f (PLA=%.1f), %.2f (PLA=%.1f)]\n', ...
          n2min, ok_res(imin,1), n2max, ok_res(imax,1));
  fprintf(outfid,'  初始 4349.29 rpm 相对位置: %.1f%%\n', ...
          (4349.29-n2min)/(n2max-n2min)*100);
  fprintf(outfid,'\n  ★ 目标检查:\n');
  targets = [4349.29+750, 4349.29+1500, 4349.29+1000, 4349.29+500, 4349.29];
  nms = {'+750 (5099)', '+1500 (5849)', '+1000 (5349)', '+500 (4849)', '恢复 (4349)'};
  for i=1:numel(targets)
    reach = targets(i) >= n2min && targets(i) <= n2max;
    fprintf(outfid,'    %-16s %.2f rpm -> %s\n', nms{i}, targets(i), ...
            ternary(reach,'可达','不可达'));
  end
else
  fprintf(outfid,'  无有效工作点\n');
end

save(fullfile(outDir,'feasibility.mat'), 'RESULTS', 'PLA_LIST');
fclose(outfid);
fprintf('PHYS_OK\n');

function s = ternary(c,a,b)
if c, s=a; else, s=b; end
end
