% AGTF30e 扩充数据生成 v2：多样 PLA 曲线 + 全工况覆盖
% 6 种 PLA 曲线类型 x 6 种工况组合 = 36 条轨迹

if ~exist('run_i','var'), run_i = 1; end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
outDir = 'C:\pclwm\agtf30\dataset2';
if ~isfolder(outDir), mkdir(outDir); end
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

% ===== 6 种 PLA 曲线（时间, 值）=====
% 统一 120 s，都在 20 s 开始动作
PROFILES = {
  'step80',  [0 20 20.015 50 50.015 120], [40 40 80 80 40 40]
  'step55',  [0 20 20.015 50 50.015 120], [40 40 55 55 40 40]
  'step95',  [0 20 20.015 50 50.015 120], [40 40 95 95 40 40]
  'ramp',    [0 20 50 80 110 120],        [40 40 80 80 40 40]
  'multi',   [0 20 20.015 50 50.015 80 80.015 120], [40 40 60 60 85 85 40 40]
  'down',    [0 20 20.015 50 50.015 120], [90 90 50 50 90 90]
};

% ===== 6 种工况组合 =====
CONDS = [
  0      0.0   1.0
  0      0.6   1.0
  10000  0.3   1.0
  20000  0.6   2.0
  20000  0.3   0.5
  30000  0.5   1.0
];

nP = size(PROFILES,1);
nC = size(CONDS,1);
nTot = nP*nC;
if run_i < 1 || run_i > nTot
  fprintf('bad run_i\n'); return
end
pIdx = floor((run_i-1)/nC) + 1;
cIdx = mod(run_i-1, nC) + 1;

pname = PROFILES{pIdx,1};
tp    = PROFILES{pIdx,2};
vp    = PROFILES{pIdx,3};
alt   = CONDS(cIdx,1);
mn    = CONDS(cIdx,2);
gs    = CONDS(cIdx,3);

tag = sprintf('d%02d_%s_a%05d_m%02d_g%03d', run_i, pname, alt, ...
              round(mn*10), round(gs*10));
fid = fopen(fullfile(outDir,[tag '.txt']),'w','n','UTF-8');
fprintf(fid,'=== %s ===\n', tag);
fprintf(fid,'  profile=%s  Alt=%g MN=%g gainScale=%g\n', pname, alt, mn, gs);
fprintf(fid,'  PLA t=%s\n  PLA v=%s\n', mat2str(tp), mat2str(vp));

try
  MWS = Setup_Simulation(2,[],'AGTF30e_Inputs.xlsx',1,2);
catch ME
  fprintf(fid,'  setup1 FAIL: %s\n', ME.message);
  fclose(fid); disp(['D2_' tag '_F1']); return
end

MWS.In.Options.HybridConfig = 1;
MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1;
MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2;
MWS.In.Options.TEEM = 1;

MWS.In.t_Alt = [0 10];   MWS.In.Alt = alt*[1 1];
MWS.In.t_MN  = [0 10];   MWS.In.MN  = mn*[1 1];
MWS.In.t_dTamb = [0 10]; MWS.In.dTamb = [0 0];
MWS.In.t_PLA = tp;
MWS.In.PLA   = vp;
MWS.In.t_Boost = [0 3 3.015 70 70.015 120]; MWS.In.Boost  = zeros(1,6);
MWS.In.t_Charge= [0 95 95.015 120];         MWS.In.Charge = zeros(1,4);
MWS.In.t_EPT   = [0 65 65.015 95 95.015 120]; MWS.In.EPT = zeros(1,6);

try
  MWS = Setup_Simulation(1,MWS.In,'AGTF30e_Inputs.xlsx',1,2);
catch ME
  fprintf(fid,'  setup2 FAIL: %s\n', ME.message);
  fclose(fid); disp(['D2_' tag '_F2']); return
end

% 增益缩放（第二次 Setup 之后）
try
  bKp = MWS.Cntrl.WfCntrl.Nom.Kp_array;
  bKi = MWS.Cntrl.WfCntrl.Nom.Ki_array;
  MWS.Cntrl.WfCntrl.Nom.Kp_array = bKp * gs;
  MWS.Cntrl.WfCntrl.Nom.Ki_array = bKi * gs;
  assignin('base','MWS',MWS);
  fprintf(fid,'  gain x%g\n', gs);
catch ME
  fprintf(fid,'  gain FAIL: %s\n', ME.message);
end

try
  t0=tic; sim('AGTF30SysDyn_PExPIn.slx'); el=toc(t0);
  fprintf(fid,'  SIM OK %.0fs\n', el);
catch ME
  fprintf(fid,'  SIM FAIL: %s\n', ME.message);
  fclose(fid); disp(['D2_' tag '_SF']); return
end

% ---- 收集全部 out_* ----
allv = evalin('base','whos');
S = struct(); nok=0;
for k=1:numel(allv)
  if startsWith(allv(k).name,'out_')
    try
      S.(allv(k).name) = evalin('base', allv(k).name);
      nok = nok+1;
    catch
    end
  end
end
fprintf(fid,'  收集 %d 个 out_* 变量\n', nok);

% ---- 物理检验 ----
getc = @(b,f) local_get2(S,b,f);
t36 = getc('out_S36','Tt'); t4 = getc('out_S4','Tt');
t45 = getc('out_S45','Tt'); t5 = getc('out_S5','Tt');
p36 = getc('out_S36','Pt');

ok = true;
if isempty(t4) || any(~isfinite(t4)), ok=false; fprintf(fid,'  [FAIL] T4 非有限\n'); end
if ~isempty(t4) && ~isempty(t36)
  v1 = all(t4>t36); if ~v1, fprintf(fid,'  [FAIL] S4<S36\n'); ok=false; end
end
if ~isempty(t4) && ~isempty(t45)
  v2 = all(t4>t45); if ~v2, fprintf(fid,'  [FAIL] S4<S45\n'); ok=false; end
end
if ~isempty(t45) && ~isempty(t5)
  v3 = all(t45>t5); if ~v3, fprintf(fid,'  [FAIL] S45<S5\n'); ok=false; end
end
if ~isempty(t4)
  if max(t4)>4000 || min(t4)<200
    fprintf(fid,'  [FAIL] T4 越界 [%.0f,%.0f]\n',min(t4),max(t4)); ok=false;
  end
  d = diff(t4); sg=sign(d); sg(sg==0)=1;
  fprintf(fid,'  T4 rel=%.1f%% turns=%d range=[%.0f,%.0f]\n', ...
          (max(t4)-min(t4))/max(t4)*100, sum(diff(sg)~=0), min(t4), max(t4));
end
if ~isempty(p36)
  fprintf(fid,'  Pt36 range=[%.1f,%.1f]\n', min(p36), max(p36));
  if max(p36)>800, fprintf(fid,'  [FAIL] Pt36 越界\n'); ok=false; end
end
n_t = numel(t4);
fprintf(fid,'  n=%d\n', n_t);
if n_t < 5000
  fprintf(fid,'  [FAIL] 提前终止\n'); ok=false;
end
fprintf(fid,'  >>> %s\n', ternary2(ok,'VALID','INVALID'));

save(fullfile(outDir,[tag '.mat']),'-struct','S','-v7');
fclose(fid);
disp(['D2_' tag '_' ternary2(ok,'OK','BAD')]);

function v = local_get2(S, blk, fld)
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

function s = ternary2(c,a,b)
  if c, s=a; else, s=b; end
end
