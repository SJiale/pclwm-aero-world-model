% PLA 信号链完整拓扑 + 单点替换可行性
%
% 已知：
%   Model Source4 (Lookup, Mask: MWS.In.PLA) -> Bus Creator1 port 1
%
% 目标：找到"只替换一个信号源"的位置

root = 'D:\不太用的项目\AGTF30-e-main\AGTF30-e-main';
agtf = fullfile(root,'AGTF30e');
cd(agtf); addpath(agtf); addpath(fullfile(agtf,'SimSetup'));
addpath(fullfile(agtf,'Functions'));
addpath(genpath(fullfile(root,'TMATS')));
addpath(genpath(fullfile(root,'EMTAT-1.3.2')));

MDL = 'AGTF30SysDyn_PExPIn';
outDir = 'C:\pclwm\pclwm_stage3a\step_api';
try, close_system(MDL,0); catch, end
load_system(MDL);

fid = fopen(fullfile(outDir,'pla_chain_topology.txt'),'w','n','UTF-8');
fprintf(fid,'=== PLA 信号链完整拓扑 ===\n\n');

% ══ 1. Model Source4 输出 -> Bus Creator1 ══
SRC = [MDL '/System Inputs/Model Source4'];
BC = [MDL '/System Inputs/Bus Creator1'];

fprintf(fid,'[1] 信号链起点\n');
fprintf(fid,'  %s (T-MATS Model Source, Lookup)\n', SRC);
fprintf(fid,'    Mask: T_vec_M=MWS.In.t_PLA, input_array_M=MWS.In.PLA\n');
fprintf(fid,'    -> Bus Creator1, port 1\n\n');

% ══ 2. Bus Creator1 的其他输入 ══
fprintf(fid,'[2] Bus Creator1 的所有输入\n');
ph = get_param(BC,'PortHandles');
n_in = numel(ph.Inport);
fprintf(fid,'  输入端口数: %d\n', n_in);
for i=1:n_in
  l = get_param(ph.Inport(i),'Line');
  if l > 0
    sp = get_param(l,'SrcPortHandle');
    if ~isempty(sp)
      sb = get_param(sp(1),'Parent');
      fprintf(fid,'    port %d: [%s] %s\n', i, get_param(sb,'BlockType'), ...
              strrep(sb, [MDL '/'], ''));
    end
  end
end

% ══ 3. Bus Creator1 的输出 ══
fprintf(fid,'\n[3] Bus Creator1 输出\n');
if ~isempty(ph.Outport)
  l = get_param(ph.Outport(1),'Line');
  if l > 0
    dp = get_param(l,'DstPortHandle');
    for i=1:numel(dp)
      db = get_param(dp(i),'Parent');
      fprintf(fid,'    -> [%s] %s\n', get_param(db,'BlockType'), ...
              strrep(db, [MDL '/'], ''));
    end
  end
end

% ══ 4. 搜索谁消费 PLA ══
fprintf(fid,'\n[4] 搜索 PLA 的下游消费者\n');
% 找 Controller 子系统
ctrl = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                   'regexp','on','Name','Control');
fprintf(fid,'  名称含 Control 的子系统: %d 个\n', numel(ctrl));
for i=1:min(numel(ctrl),10)
  fprintf(fid,'    %s\n', strrep(ctrl{i}, [MDL '/'], ''));
end

% EEC 相关
eec = find_system(MDL,'LookUnderMasks','all','FollowLinks','on', ...
                  'regexp','on','Name','.*EEC.*|.*Controller.*|.*Cntrl.*');
fprintf(fid,'\n  名称含 EEC/Controller/Cntrl: %d 个\n', numel(eec));
for i=1:min(numel(eec),15)
  fprintf(fid,'    [%s] %s\n', get_param(eec{i},'BlockType'), ...
          strrep(eec{i}, [MDL '/'], ''));
end

% ══ 5. 单点替换可行性评估 ══
fprintf(fid,'\n[5] 单点替换可行性\n');
fprintf(fid,'  方案：把 Model Source4 替换为 Inport 或 Signal Builder\n');
fprintf(fid,'  \n');
fprintf(fid,'  优点：\n');
fprintf(fid,'    - 只需替换 1 个模块（28 个 Model Source 中的 1 个）\n');
fprintf(fid,'    - PLA 信号链路清晰：Lookup -> Bus Creator1 port 1\n');
fprintf(fid,'    - 下游不需改动\n');
fprintf(fid,'  \n');
fprintf(fid,'  风险：\n');
fprintf(fid,'    - Model Source4 的输出数据类型/采样时间需匹配\n');
fprintf(fid,'    - T-MATS 内可能有隐藏依赖\n');
fprintf(fid,'  \n');
fprintf(fid,'  验证方法：\n');
fprintf(fid,'    1. 检查 Model Source4 的输出端口数据类型\n');
fprintf(fid,'    2. 在模型副本中替换为 Inport\n');
fprintf(fid,'    3. 用 SimulationInput.setExternalInput 驱动\n');
fprintf(fid,'    4. 验证与旧方法数值一致\n');

% ══ 6. Model Source4 输出端口属性 ══
fprintf(fid,'\n[6] Model Source4 输出端口属性\n');
ph4 = get_param(SRC,'PortHandles');
if ~isempty(ph4.Outport)
  fprintf(fid,'  端口数: %d\n', numel(ph4.Outport));
  % 尝试读数据类型
  try
    dt = get_param(SRC,'OutDataTypeStr');
    fprintf(fid,'  OutDataTypeStr = %s\n', dt);
  catch
  end
  try
    st = get_param(SRC,'SampleTime');
    fprintf(fid,'  SampleTime = %s\n', st);
  catch
  end
end
% 查子系统输出
try
  op = find_system(SRC,'LookUnderMasks','all','SearchDepth',1, ...
                   'BlockType','Outport');
  for i=1:numel(op)
    fprintf(fid,'  Outport %s: %s\n', get_param(op{i},'Port'), ...
            get_param(op{i},'OutDataTypeStr'));
  end
catch
end

% ══ 7. 检查是否有现成的 Signal Builder / Inport 变体 ══
fprintf(fid,'\n[7] 检查 T-MATS Model Source 的内部实现\n');
inner = find_system(SRC,'LookUnderMasks','all','FollowLinks','on', ...
                    'MatchFilter',@Simulink.match.allVariants);
for i=1:numel(inner)
  bt = get_param(inner{i},'BlockType');
  nm = get_param(inner{i},'Name');
  fprintf(fid,'  [%s] %s\n', bt, nm);
end

fclose(fid);
fprintf('PHYS_OK\n');
