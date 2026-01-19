import json

notebook_path = r'D:\githubcode\SoilNet\dataset\Climate\ClimateInformationGEEpyAPI.ipynb'

# 读取
with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()
    nb = json.loads(content)

# 修改 cell 3
cell = nb['cells'][3]
source = cell['source']

# 方法：直接替换字符串内容
new_source = []
for line in source:
    # 修改 Climate_bands 列表
    if line.strip() == "'srad', 'tmmn', 'tmmx', 'LST_warm', 'vap', 'vpd', 'vs', 'swe'":
        new_source.append("    'srad', 'tmmn', 'tmmx', 'LST_warm', 'vap', 'vpd', 'vs', 'swe',\n")
        new_source.append("    'def', 'ro', 'soil'  # 新增：气候水分亏缺、径流、土壤湿度\n")
    # 修改主循环判断
    elif "if var in ['AET','PET','pr','pdsi','srad','tmmn','tmmx','vap','vpd','vs','swe']:" in line and line.strip().startswith("if var"):
        new_source.append("    if var in ['AET','PET','pr','pdsi','srad','tmmn','tmmx','vap','vpd','vs','swe','def','ro','soil']:\n")
    # 修改重试逻辑判断
    elif "if var in ['AET','PET','pr','pdsi','srad','tmmn','tmmx','vap','vpd','vs','swe']:" in line and "        if var" in line:
        new_source.append("        if var in ['AET','PET','pr','pdsi','srad','tmmn','tmmx','vap','vpd','vs','swe','def','ro','soil']:\n")
    else:
        new_source.append(line)

cell['source'] = new_source

# 保存（保持原始格式）
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Success!")









