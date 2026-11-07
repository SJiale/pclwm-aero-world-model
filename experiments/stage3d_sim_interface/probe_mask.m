% 测试：用 setBlockParameter 动态更新 PLA
%
% 假设：PLA 通过 Model Source 子系统的 Mask 参数 input_array_M 输入
% 方法：找到 PLA 对应的 Model Source，更新其 input_array_M

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

fid = fopen(fullfile(outDir,'mask_probe.txt'),'w','n','UTF-8');
fprintf(fid,'=== Model Source Mask 参数探查 ===\n\n');

% 先建立 MWS 并 Setup
T_END = T_SETTLE + 5*CTRL_DT;
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

% 现在读每个 Model Source 的 Mask 参数值
load_system(MDL);
fprintf(fid,'=== 各 Model Source 的 Mask 参数 ===\n');
fprintf(fid,'  %-14s %-28s %-28s\n','源','T_vec_M','input_array_M');
fprintf(fid,'  %s\n', repmat('-',1,72));

for k = 0:27
  if k==0, nm='Model Source'; else, nm=sprintf('Model Source%d',k); end
  blk = [MDL '/System Inputs/' nm];
  try
    tv = get_param(blk,'T_vec_M');
    ia = get_param(blk,'input_array_M');
    % 只打印摘要
    tvs = mat2str(tv);
    ias = mat2str(ia);
    if numel(tvs)>26, tvs=[tvs(1:26) '...']; end
    if numel(ias)>26, ias=[ias(1:26) '...']; end
    fprintf(fid,'  %-14s %-28s %-28s\n', nm, tvs, ias);
    fprintf('  %s: T_vec_M=%s, input_array_M=%s\n', nm, tvs, ias);
  catch ME
    fprintf(fid,'  %-14s <无 mask 参数>\n', nm);
  end
end

% 找哪个的 input_array 值在 40-80 之间（PLA 特征）
fprintf(fid,'\n=== 按值域定位 PLA ===\n');
for k = 0:27
  if k==0, nm='Model Source'; else, nm=sprintf('Model Source%d',k); end
  blk = [MDL '/System Inputs/' nm];
  try
    ia = get_param(blk,'input_array_M');
    if isnumeric(ia) && ~isempty(ia)
      lo = min(ia(:)); hi = max(ia(:));
      if lo > 30 && hi < 90 && (hi-lo) < 10
        fprintf(fid,'  ★ %-14s 值域 [%.2f, %.2f] 可能是 PLA\n', nm, lo, hi);
        fprintf('  ★ %s: [%.2f, %.2f]  <- 可能 PLA\n', nm, lo, hi);
      end
    end
  catch
  end
end

fclose(fid);
fprintf('PHYS_OK\n');
