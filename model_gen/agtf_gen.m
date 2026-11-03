% AGTF30e 数据生成：单次运行
% 外部设置: run_i (1-based index into config table)
% 同时改工况 (Alt/MN/PLA) 和增益缩放系数

if ~exist('run_i','var'), run_i = 1; end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
outDir = 'C:\pclwm\agtf30\dataset';
if ~isfolder(outDir), mkdir(outDir); end
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

% ===== 配置表（第一批：边界组合）=====
% {Alt[ft], MN, PLA_hi, gainScale}
CFG = {
   0      0     80   1.0
   0      0     80   2.0
   0      0     80   0.5
   0      0.6   80   1.0
   0      0.6   90   2.0
   20000  0     80   1.0
   20000  0.3   80   1.0
   20000  0.6   90   2.0
   20000  0.6   60   0.5
   30000  0     90   1.0
   30000  0.3   80   2.0
   30000  0.6   60   0.5
};
nCfg = size(CFG,1);
if run_i < 1 || run_i > nCfg
  fprintf('bad run_i\n'); return
end
alt = CFG{run_i,1};
mn  = CFG{run_i,2};
pla = CFG{run_i,3};
gs  = CFG{run_i,4};
tag = sprintf('r%02d_a%05d_m%02d_p%02d_g%03d', run_i, alt, round(mn*10), pla, round(gs*10));

fid = fopen(fullfile(outDir,[tag '.txt']),'w','n','UTF-8');
fprintf(fid,'=== %s ===\n', tag);
fprintf(fid,'  Alt=%g ft  MN=%g  PLA_hi=%g  gainScale=%g\n', alt, mn, pla, gs);

% ===== Setup =====
try
  MWS = Setup_Simulation(2,[],'AGTF30e_Inputs.xlsx',1,2);
  fprintf(fid,'  setup OK\n');
catch ME
  fprintf(fid,'  setup FAIL: %s\n', ME.message);
  fclose(fid); disp(['DONE_' tag '_FAIL']); return
end


% ===== 设置工况 =====
MWS.In.Options.HybridConfig = 1;
MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1;
MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2;
MWS.In.Options.TEEM = 1;

MWS.In.t_Alt = [0 10];   MWS.In.Alt = alt*[1 1];
MWS.In.t_MN  = [0 10];   MWS.In.MN  = mn*[1 1];
MWS.In.t_dTamb = [0 10]; MWS.In.dTamb = [0 0];

pla_lo = 40;
MWS.In.t_PLA = [0 20 20.015 50 50.015 120];
MWS.In.PLA   = [pla_lo pla_lo pla pla pla_lo pla_lo];
MWS.In.t_Boost = [0 3 3.015 70 70.015 120];
MWS.In.Boost   = [0 0 0 0 0 0];
MWS.In.t_Charge= [0 95 95.015 120];
MWS.In.Charge  = [0 0 0 0];
MWS.In.t_EPT   = [0 65 65.015 95 95.015 120];
MWS.In.EPT     = [0 0 0 0 0 0];

try
  MWS = Setup_Simulation(1,MWS.In,'AGTF30e_Inputs.xlsx',1,2);
  fprintf(fid,'  inputs OK. tend=%g Ts=%g\n', MWS.In.tend, MWS.In.Ts);
catch ME
  fprintf(fid,'  inputs FAIL: %s\n', ME.message);
  fclose(fid); disp(['DONE_' tag '_FAIL']); return
end

% ===== 关键：第二次 Setup 之后再缩放增益（否则会被覆盖）=====
try
  baseKp = MWS.Cntrl.WfCntrl.Nom.Kp_array;
  baseKi = MWS.Cntrl.WfCntrl.Nom.Ki_array;
  MWS.Cntrl.WfCntrl.Nom.Kp_array = baseKp * gs;
  MWS.Cntrl.WfCntrl.Nom.Ki_array = baseKi * gs;
  assignin('base','MWS',MWS);
  fprintf(fid,'  gain scaled x%g: Kp[%.5g,%.5g]->[%.5g,%.5g]\n', gs, ...
          min(baseKp(:)), max(baseKp(:)), ...
          min(MWS.Cntrl.WfCntrl.Nom.Kp_array(:)), ...
          max(MWS.Cntrl.WfCntrl.Nom.Kp_array(:)));
catch ME
  fprintf(fid,'  gain scale FAIL: %s\n', ME.message);
end

% ===== 运行 =====
try
  t0=tic; sim('AGTF30SysDyn_PExPIn.slx'); el=toc(t0);
  fprintf(fid,'  SIM OK %.1fs\n', el);
catch ME
  fprintf(fid,'  SIM FAIL: %s\n', ME.message);
  if isprop(ME,'cause') && ~isempty(ME.cause)
    fprintf(fid,'    cause: %s\n', ME.cause{1}.message);
  end
  fclose(fid); disp(['DONE_' tag '_SIMFAIL']); return
end

% ===== 收集 =====
vars = {'out_S0','out_S2','out_S25','out_S36','out_S4','out_S45','out_S5','out_Dyn'};
S = struct(); nok=0;
for k=1:numel(vars)
  try
    v = evalin('base', vars{k});
    S.(vars{k}) = v;
    nok = nok+1;
  catch
  end
end

% ===== 物理有效性检验 =====
fprintf(fid,'\n=== 物理有效性检验 ===\n');
ok_phys = true;
stats = struct();

% 提取关键量
getc = @(blk,fld) local_get(S, blk, fld);
t36 = getc('out_S36','Tt');  p36 = getc('out_S36','Pt');
t4  = getc('out_S4','Tt');   p4  = getc('out_S4','Pt');
t45 = getc('out_S45','Tt');  t5  = getc('out_S5','Tt');
w0  = getc('out_S0','W');    w36 = getc('out_S36','W');

% 1. 有限性
finite_ok = true;
for nm = {'t36','p36','t4','p4','t45','t5','w0','w36'}
  v = eval(nm{1});
  if isempty(v) || any(~isfinite(v))
    finite_ok = false;
    fprintf(fid,'  [FAIL] %s 含非有限值\n', nm{1});
  end
end
if finite_ok
  fprintf(fid,'  [OK] 数值有限\n');
else
  ok_phys = false;
end

% 2. 温度序
if ~isempty(t4) && ~isempty(t36) && ~isempty(t45) && ~isempty(t5)
  v1 = all(t4 > t36); v2 = all(t45 > t5); v3 = all(t4 > t45);
  fprintf(fid,'  [%s] S4>S36  %d/%d 点\n', ternary(v1,'OK','FAIL'), sum(t4>t36), numel(t4));
  fprintf(fid,'  [%s] S4>S45  %d/%d 点\n', ternary(v3,'OK','FAIL'), sum(t4>t45), numel(t4));
  fprintf(fid,'  [%s] S45>S5  %d/%d 点\n', ternary(v2,'OK','FAIL'), sum(t45>t5), numel(t45));
  if ~(v1 && v2 && v3), ok_phys = false; end
end

% 3. 有界性（放宽，允许激进）
if ~isempty(t4)
  t4max = max(t4); t4min = min(t4);
  fprintf(fid,'  T4 范围 [%.1f, %.1f] °R\n', t4min, t4max);
  if t4max > 4000 || t4min < 200
    fprintf(fid,'  [FAIL] T4 越界\n'); ok_phys = false;
  end
end
if ~isempty(p36)
  fprintf(fid,'  Pt36 范围 [%.1f, %.1f] psia\n', min(p36), max(p36));
  if max(p36) > 800 || min(p36) < 1
    fprintf(fid,'  [FAIL] Pt36 越界\n'); ok_phys = false;
  end
end

% 4. 动态幅度 + 振荡
if ~isempty(t4)
  rel4 = (max(t4)-min(t4))/max(t4)*100;
  d = diff(t4); sg=sign(d); sg(sg==0)=1;
  turns = sum(diff(sg)~=0);
  fprintf(fid,'  T4 相对变化 %.1f%%, 转折点 %d\n', rel4, turns);
  stats.t4_rel = rel4; stats.t4_turns = turns;
  stats.t4_max = max(t4); stats.t4_min = min(t4);
end
if ~isempty(p36)
  relp = (max(p36)-min(p36))/max(p36)*100;
  d = diff(p36); sg=sign(d); sg(sg==0)=1;
  turns_p = sum(diff(sg)~=0);
  fprintf(fid,'  Pt36 相对变化 %.1f%%, 转折点 %d\n', relp, turns_p);
  stats.p36_rel = relp; stats.p36_turns = turns_p;
end

fprintf(fid,'  >>> 判定: %s\n', ternary(ok_phys,'VALID','INVALID'));
stats.valid = ok_phys;
stats.alt=alt; stats.mn=mn; stats.pla=pla; stats.gainScale=gs;
stats.tag=tag;

% ===== 保存 =====
save(fullfile(outDir,[tag '.mat']),'-struct','S','-v7');
save(fullfile(outDir,[tag '_stats.mat']),'stats');
fprintf(fid,'  saved\n');
fclose(fid);
disp(['DONE_' tag '_' ternary(ok_phys,'VALID','INVALID')]);

% ---- helpers ----
function v = local_get(S, blk, fld)
  v = [];
  try
    if isfield(S,blk) && isfield(S.(blk),fld)
      x = S.(blk).(fld);
      if isa(x,'timeseries')
        d = x.Data; if size(d,2)>1, d=d(:,1); end
        v = d(:);
      end
    end
  catch
  end
end

function s = ternary(c,a,b)
  if c, s=a; else, s=b; end
end

