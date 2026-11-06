% 控制信号关系检查：PLA / EEC / 燃油指令 / 转速参考
%
% 目标：确定是否可采用上层监督控制 S 路线
%   1. PLA 如何影响 EEC
%   2. EEC 如何产生 Wfdmd
%   3. 转速参考 N1cSP / N1cErr 从哪来
%   4. 是否存在可注入的 S 层动作点

outDir = 'C:\pclwm\ctrl\probe';
if ~isfolder(outDir), mkdir(outDir); end

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

TEND = 20; TS = 0.015;
t = (0:TS:TEND)';

fprintf('=== 建立仿真 ===\n');
MWS = Setup_Simulation(2, [], 'AGTF30e_Inputs.xlsx', 1, 2);
MWS.In.Options.HybridConfig = 1; MWS.In.Options.EPSLosses = 2;
MWS.In.Options.EngineEMInt = 1; MWS.In.Options.PGBConfig = 2;
MWS.In.Options.WfTransientLogic = 2; MWS.In.Options.TEEM = 1;
MWS.In.t_Alt = t; MWS.In.Alt = zeros(size(t));
MWS.In.t_MN = t; MWS.In.MN = zeros(size(t));
MWS.In.t_PLA = t; MWS.In.PLA = 55*ones(size(t));
MWS.In.t_dTamb = [0 TEND]; MWS.In.dTamb = [0 0];
MWS.In.t_Boost = [0 1 TEND]; MWS.In.Boost = [0 0 0];
MWS.In.t_Charge = [0 1 TEND]; MWS.In.Charge = [0 0 0];
MWS.In.t_EPT = [0 1 TEND]; MWS.In.EPT = [0 0 0];
MWS = Setup_Simulation(1, MWS.In, 'AGTF30e_Inputs.xlsx', 1, 2);
assignin('base','MWS',MWS);
set_param('AGTF30SysDyn_PExPIn','StopTime',num2str(TEND));
sim('AGTF30SysDyn_PExPIn.slx');

D = evalin('base','out_Dyn');

% ══════════════════════════════════════════════════
% 1. cntrl 结构完整探查
% ══════════════════════════════════════════════════
fid = fopen(fullfile(outDir,'ctrl_structure.txt'),'w','n','UTF-8');
fprintf(fid,'=== out_Dyn.cntrl 结构 ===\n');

if isfield(D,'cntrl')
  C = D.cntrl;
  f1 = fieldnames(C);
  for i=1:numel(f1)
    v = C.(f1{i});
    fprintf(fid,'cntrl.%s : %s\n', f1{i}, class(v));
    if isa(v,'timeseries')
      d = v.Data;
      if ~isempty(d)
        fprintf(fid,'  n=%d  range=[%.4g, %.4g]  mean=%.4g\n', ...
                numel(d), min(d(:)), max(d(:)), mean(d(:)));
      end
    elseif isstruct(v)
      f2 = fieldnames(v);
      for j=1:numel(f2)
        w = v.(f2{j});
        if isa(w,'timeseries')
          d = w.Data;
          if ~isempty(d)
            fprintf(fid,'  .%s : n=%d [%.4g, %.4g] mean=%.4g\n', ...
                    f2{j}, numel(d), min(d(:)), max(d(:)), mean(d(:)));
          end
        else
          fprintf(fid,'  .%s : %s\n', f2{j}, class(w));
        end
      end
    end
  end
else
  fprintf(fid,'无 cntrl 字段\n');
end

% ══════════════════════════════════════════════════
% 2. act 结构（执行器）
% ══════════════════════════════════════════════════
fprintf(fid,'\n=== out_Dyn.act 结构 ===\n');
if isfield(D,'act')
  A = D.act;
  f1 = fieldnames(A);
  for i=1:numel(f1)
    v = A.(f1{i});
    if isa(v,'timeseries')
      d = v.Data;
      fprintf(fid,'act.%s : n=%d [%.4g, %.4g] mean=%.4g\n', ...
              f1{i}, numel(d), min(d(:)), max(d(:)), mean(d(:)));
    else
      fprintf(fid,'act.%s : %s\n', f1{i}, class(v));
    end
  end
end

% ══════════════════════════════════════════════════
% 3. in 结构（输入）
% ══════════════════════════════════════════════════
fprintf(fid,'\n=== out_Dyn.in 结构 ===\n');
if isfield(D,'in')
  I = D.in;
  f1 = fieldnames(I);
  for i=1:numel(f1)
    v = I.(f1{i});
    if isa(v,'timeseries')
      d = v.Data;
      fprintf(fid,'in.%s : n=%d [%.4g, %.4g]\n', ...
              f1{i}, numel(d), min(d(:)), max(d(:)));
    else
      fprintf(fid,'in.%s : %s\n', f1{i}, class(v));
    end
  end
end

% ══════════════════════════════════════════════════
% 4. 关键信号时序导出
% ══════════════════════════════════════════════════
fprintf(fid,'\n=== 关键信号 ===\n');

sig = {};
names_sig = {};
function [sig, names_sig] = addts(sig, names_sig, obj, pref)
  if isa(obj,'timeseries')
    d = obj.Data;
    if size(d,2)>1, d = d(:,1); end
    if numel(d)==2001  % TEND=20, TS=0.015 -> 1334 点
      sig{end+1} = d(:);
      names_sig{end+1} = pref;
    end
  end
end

% 用通用递归查找所有 timeseries
function walk(obj, pref, fid, depth)
  if depth > 4, return; end
  if ~isstruct(obj), return; end
  f = fieldnames(obj);
  for i=1:numel(f)
    try
      v = obj.(f{i});
      nm = [pref '.' f{i}];
      if isa(v,'timeseries')
        d = v.Data;
        if ~isempty(d)
          fprintf(fid,'TS %s : n=%d [%.4g, %.4g]\n', ...
                  nm, numel(d), min(d(:)), max(d(:)));
        end
      elseif isstruct(v)
        walk(v, nm, fid, depth+1);
      end
    catch
    end
  end
end

walk(D, 'Dyn', fid, 0);

fclose(fid);
fprintf('PHYS_OK\n');
