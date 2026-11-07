% 任务 2：采集 6 条 test_ctrl 独立验证轨迹
%
% 使用 Simulink.SimulationInput（推荐 API，替代 set_param）

outDir = 'C:\pclwm\ctrl\test_ctrl';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
TS = 0.015; TEND = 60;
t = (0:TS:TEND)';
N = numel(t);

warning('off','Simulink:blocks:AssumingDefaultSimStateForSFcn');
try, close_system(MDL, 0); catch, end

logfid = fopen(fullfile(outDir,'collect_log.txt'),'w','n','UTF-8');
fprintf(logfid,'=== test_ctrl 采集日志 ===\n');
fprintf(logfid,'  每条 %.0f s, %.3f s 采样, %d 点\n\n', TEND, TS, N);

% ══════════════════════════════════════════════════
% 定义 6 条轨迹
% ══════════════════════════════════════════════════
CASES = {};

p = 55 + 2*sin(2*pi*t/8);
CASES{end+1} = struct('name','V1_sin_e1_alt0','alt',0,'mn',0.0, ...
  'pla',p,'desc','正弦 53-57 deg, 周期 8s, Alt=0 MN=0');

tri = 55 + 3*sawtooth(2*pi*t/12, 0.5);
CASES{end+1} = struct('name','V2_tri_e2_alt10k','alt',10000,'mn',0.3, ...
  'pla',tri,'desc','三角 52-58 deg, 周期 12s, Alt=10k MN=0.3');

rng(42);
noise = randn(size(t));
a_lp = exp(-TS/2.0);
smooth = filter(1-a_lp, [1 -a_lp], noise);
smooth = smooth / std(smooth) * 2.5;
p3 = 55 + smooth;
p3 = max(48, min(62, p3));
CASES{end+1} = struct('name','V3_rand_e3_alt20k','alt',20000,'mn',0.5, ...
  'pla',p3,'desc','随机平滑 48-62 deg (tau=2s), Alt=20k MN=0.5');

p4 = 55*ones(size(t));
p4(t>=15 & t<30) = 54;
p4(t>=30 & t<45) = 56;
p4(t>=45) = 54;
CASES{end+1} = struct('name','V4_dstep_alt5k','alt',5000,'mn',0.15, ...
  'pla',p4,'desc','双向阶跃 55->54->56->54, Alt=5k MN=0.15');

p5 = 55 + 4*sin(2*pi*t/15);
CASES{end+1} = struct('name','V5_sinw_e2_alt15k','alt',15000,'mn',0.4, ...
  'pla',p5,'desc','正弦 51-59 deg, 周期 15s, Alt=15k MN=0.4');

p6 = 55 + 1.5*sin(2*pi*t/6);
p6(t>=20 & t<35) = p6(t>=20 & t<35) + 2.5;
p6(t>=40) = p6(t>=40) - 1.5;
p6 = max(48, min(62, p6));
CASES{end+1} = struct('name','V6_mix_alt8k','alt',8000,'mn',0.25, ...
  'pla',p6,'desc','混合（正弦周期6s + 阶跃偏移）, Alt=8k MN=0.25');

fprintf(logfid,'  共 %d 条轨迹\n', numel(CASES));
for i=1:numel(CASES)
  fprintf(logfid,'    %-24s %s\n', CASES{i}.name, CASES{i}.desc);
end
fprintf(logfid,'\n');

% ══════════════════════════════════════════════════
% 逐条采集
% ══════════════════════════════════════════════════
fprintf(logfid,'=== 采集 ===\n');
fprintf(logfid,'  %-24s %18s %18s %8s %10s\n','名称','N2范围','Wfdmd范围','耗时(s)','饱和(Wf/PLA)');

meta = struct();
meta.n_traj = numel(CASES);
meta.trajs = {};

for i = 1:numel(CASES)
  C = CASES{i};

  % 构建 MWS（通过 base 工作区变量）
  MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
  MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
  MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
  MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
  MWS.In.t_Alt = t; MWS.In.Alt = C.alt*ones(size(t));
  MWS.In.t_MN = t; MWS.In.MN = C.mn*ones(size(t));
  MWS.In.t_dTamb = [0 TEND]; MWS.In.dTamb = [0 0];
  MWS.In.t_Boost = [0 1 TEND]; MWS.In.Boost = [0 0 0];
  MWS.In.t_Charge = [0 1 TEND]; MWS.In.Charge = [0 0 0];
  MWS.In.t_EPT = [0 1 TEND]; MWS.In.EPT = [0 0 0];
  MWS.In.t_PLA = t; MWS.In.PLA = C.pla;
  MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);

  % ★ 用 SimulationInput
  in = Simulink.SimulationInput(MDL);
  in = in.setModelParameter('StopTime', num2str(TEND));
  in = in.setModelParameter('StartTime', '0');
  in = in.setModelParameter('FastRestart', 'off');
  in = in.setModelParameter('LoadInitialState', 'off');
  in = in.setModelParameter('SaveFinalState', 'off');
  in = in.setVariable('MWS', MWS);

  tic; out = sim(in); el = toc;

  % ★ 从 SimulationOutput 直接取（不用 evalin）
  D = out.get('out_Dyn');

  o = [ D.act.Wfact.Data(:), ...
        D.loop.N2int.Data(:), ...
        D.eng.S4.Tt.Data(:), ...
        D.eng.S45.Tt.Data(:), ...
        D.eng.S36.Pt.Data(:) ];
  a_exec = D.in.PLA.Data(:);
  a_cmd = C.pla(:);
  wf_dmd = D.cntrl.Dmd.Wfdmd.Data(:);
  wf_max = D.cntrl.Data.WfMax.Data(:);
  wf_min = D.cntrl.Data.WfMin.Data(:);
  n1c_max = D.cntrl.Data.N1cMax.Data(:);
  n1c_min = D.cntrl.Data.N1cMin.Data(:);
  n1c_err = D.cntrl.Data.N1cErr.Data(:);
  n3 = D.loop.N3int.Data(:);

  sat_wf_hi = sum(wf_dmd >= wf_max*0.999);
  sat_wf_lo = sum(wf_dmd <= wf_min*1.001);
  sat_pla_hi = sum(a_exec >= 80);
  sat_pla_lo = sum(a_exec <= 40);

  % 写 CSV
  name = C.name;
  fid = fopen(fullfile(outDir, [name '.csv']),'w');
  fprintf(fid, 'time_s,Wfact_lb_s,N2int,S4_Tt,S45_Tt,S36_Pt,N3int,');
  fprintf(fid, 'PLA_cmd_deg,PLA_exec_deg,Wfdmd_lb_s,WfMax,WfMin,');
  fprintf(fid, 'N1cMax,N1cMin,N1cErr,Alt_ft,MN\n');
  for k = 1:N
    fprintf(fid, '%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.2f,%.4f\n', ...
      t(k), o(k,1), o(k,2), o(k,3), o(k,4), o(k,5), n3(k), ...
      a_cmd(k), a_exec(k), wf_dmd(k), wf_max(k), wf_min(k), ...
      n1c_max(k), n1c_min(k), n1c_err(k), C.alt, C.mn);
  end
  fclose(fid);

  fprintf(logfid,'  %-24s [%7.1f,%7.1f] [%6.4f,%6.4f] %8.2f  Wf:%d/%d PLA:%d/%d\n', ...
    name, min(o(:,2)), max(o(:,2)), min(wf_dmd), max(wf_dmd), el, ...
    sat_wf_hi, sat_wf_lo, sat_pla_hi, sat_pla_lo);

  meta.trajs{end+1} = struct('name',C.name,'alt',C.alt,'mn',C.mn, ...
    'desc',C.desc,'pla_min',min(C.pla),'pla_max',max(C.pla), ...
    'sat_wf_hi',sat_wf_hi,'sat_wf_lo',sat_wf_lo, ...
    'sat_pla_hi',sat_pla_hi,'sat_pla_lo',sat_pla_lo);
end

% ══════════════════════════════════════════════════
% 元数据
% ══════════════════════════════════════════════════
meta.purpose = 'test_ctrl: 控制导向独立验证集，不参与训练/归一化/超参选择';
meta.sampling_s = TS;
meta.duration_s = TEND;
meta.n_points = N;
meta.created = datestr(now, 'yyyy-mm-dd HH:MM:SS');
meta.model = MDL;
meta.notes = '保留原 EEC；PLA 为绝对指令；记录执行值、限幅信息、N1cErr';

fid = fopen(fullfile(outDir,'metadata.json'),'w');
fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
fclose(fid);

fprintf(logfid,'\n=== 完成 ===\n');
fprintf(logfid,'  输出: %s\n', outDir);
fprintf(logfid,'  %d 条 x %d 点 x 17 列\n', numel(CASES), N);
fclose(logfid);

fprintf('PHYS_OK\n');
