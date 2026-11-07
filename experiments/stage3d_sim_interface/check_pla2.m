% 精确定位 PLA 输入模块 + 检查可调性
root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
outDir = 'C:\pclwm\pclwm_stage3a\step_api';
fid = fopen(fullfile(outDir,'pla_input_mechanism.txt'),'w','n','UTF-8');

try, close_system(MDL,0); catch, end
load_system(MDL);

fprintf(fid,'=== PLA 输入机制核查 ===\n\n');

% 遍历所有 Model Source，看各自的 VariableName
fprintf(fid,'=== 所有 Model Source 的 From Workspace 变量 ===\n');
ms = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                 'regexp','on','Name','^Model Source');
fprintf(fid,'共 %d 个 Model Source\n\n', numel(ms));

PLA_BLK = '';
PLA_VAR = '';
for i = 1:numel(ms)
  sub = ms{i};
  fw = find_system(sub,'LookUnderMasks','all','FollowLinks','on', ...
                   'BlockType','FromWorkspace');
  if isempty(fw), continue; end
  for j = 1:numel(fw)
    vn = get_param(fw{j},'VariableName');
    bn = get_param(sub,'Name');
    fprintf(fid,'  %-20s -> %-30s (%s)\n', bn, vn, fw{j});
    if contains(upper(vn),'PLA')
      PLA_BLK = fw{j};
      PLA_VAR = vn;
    end
  end
end

fprintf(fid,'\n★ PLA 模块: %s\n', PLA_BLK);
fprintf(fid,'★ PLA 变量: %s\n', PLA_VAR);

% 检查 PLA From Workspace 的参数
if ~isempty(PLA_BLK)
  fprintf(fid,'\n=== PLA From Workspace 参数 ===\n');
  bp = get_param(PLA_BLK,'DialogParameters');
  fn = fieldnames(bp);
  for i=1:numel(fn)
    try
      v = get_param(PLA_BLK, fn{i});
      if ischar(v) || isstring(v) || isnumeric(v)
        vs = mat2str(v);
        if numel(vs)>80, vs=[vs(1:80) '...']; end
        fprintf(fid,'  %-20s = %s\n', fn{i}, vs);
      end
    catch
    end
  end
end

% 检查 MWS.In 的结构（PLA 怎么被包装）
fprintf(fid,'\n=== MWS.In 结构 ===\n');
try
  MWS = evalin('base','MWS');
  fprintf(fid,'  MWS.In 字段: %s\n', strjoin(fieldnames(MWS.In)', ', '));
  fprintf(fid,'  MWS.In.t_PLA 类型=%s size=%s\n', ...
          class(MWS.In.t_PLA), mat2str(size(MWS.In.t_PLA)));
  fprintf(fid,'  MWS.In.PLA 类型=%s size=%s\n', ...
          class(MWS.In.PLA), mat2str(size(MWS.In.PLA)));
catch ME
  fprintf(fid,'  MWS 不在 base: %s\n', ME.message);
end

% 检查是否有 MATLAB Function 或 S-Function 接收 MWS
fprintf(fid,'\n=== 模型如何读取 MWS ===\n');
mf = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                 'BlockType','MATLABFcn');
fprintf(fid,'  MATLAB Function 块: %d 个\n', numel(mf));
sf = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                 'BlockType','S-Function');
fprintf(fid,'  S-Function 块: %d 个\n', numel(sf));

% 检查 Setup_Simulation 怎么设置 PLA
fprintf(fid,'\n=== Setup_Simulation 中的 PLA 处理 ===\n');
setupF = fullfile(agtf,'SimSetup','Setup_Simulation.m');
if isfile(setupF)
  txt = fileread(setupF);
  lines = strsplit(txt, newline);
  for i=1:numel(lines)
    if contains(lines{i},'PLA') || contains(lines{i},'Input_profiles')
      fprintf(fid,'  L%d: %s\n', i, strtrim(lines{i}));
    end
  end
end

fclose(fid);
fprintf('PHYS_OK\n');
