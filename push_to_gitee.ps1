# Git 推送到 Gitee 脚本
# 使用方法: .\push_to_gitee.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "准备推送到 Gitee 仓库" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 切换到项目目录
Set-Location D:\SoilNet

# 1. 检查 Git 是否初始化
if (-not (Test-Path .git)) {
    Write-Host "错误: Git 仓库未初始化" -ForegroundColor Red
    Write-Host "请先运行: git init" -ForegroundColor Yellow
    exit 1
}

# 2. 检查当前状态
Write-Host "`n[1] 检查 Git 状态..." -ForegroundColor Yellow
git status

# 3. 添加 .gitignore
Write-Host "`n[2] 添加 .gitignore 文件..." -ForegroundColor Yellow
git add .gitignore

# 4. 添加关键代码文件
Write-Host "`n[3] 添加关键代码文件..." -ForegroundColor Yellow
git add config.py
git add train.py
git add train_utils.py
git add train_ssl.py
git add train_SimCLR_utils.py
git add soilnet/
git add dataset/*.py
git add requirements/
git add *.md
git add README.md
git add plot_utils/*.py
git add plot_utils/__init__.py

# 5. 查看将要提交的文件
Write-Host "`n[4] 查看将要提交的文件..." -ForegroundColor Yellow
$files = git status --short
Write-Host "将要提交的文件数量: $($files.Count)" -ForegroundColor Green
git status --short | Select-Object -First 30

# 6. 确认提交
Write-Host "`n[5] 准备提交..." -ForegroundColor Yellow
$commit = Read-Host "是否提交这些更改? (y/n)"
if ($commit -eq "y" -or $commit -eq "Y") {
    git commit -m "迁移到 Gitee: 核心代码和文档，排除数据和结果文件"
    Write-Host "提交完成!" -ForegroundColor Green
} else {
    Write-Host "已取消提交" -ForegroundColor Yellow
    exit 0
}

# 7. 检查远程仓库
Write-Host "`n[6] 检查远程仓库..." -ForegroundColor Yellow
$remotes = git remote -v
if ($remotes -match "gitee") {
    Write-Host "Gitee 远程仓库已存在" -ForegroundColor Green
} else {
    Write-Host "添加 Gitee 远程仓库..." -ForegroundColor Yellow
    git remote add gitee https://gitee.com/PurpleWWW/Soilnet.git
    Write-Host "Gitee 远程仓库已添加" -ForegroundColor Green
}

# 8. 检查当前分支
Write-Host "`n[7] 检查当前分支..." -ForegroundColor Yellow
$branch = git branch --show-current
if (-not $branch) {
    $branch = "main"
    git checkout -b main
    Write-Host "创建并切换到 main 分支" -ForegroundColor Yellow
}
Write-Host "当前分支: $branch" -ForegroundColor Green

# 9. 推送到 Gitee
Write-Host "`n[8] 推送到 Gitee..." -ForegroundColor Yellow
$push = Read-Host "是否推送到 Gitee? (y/n)"
if ($push -eq "y" -or $push -eq "Y") {
    git push -u gitee $branch
    Write-Host "`n推送完成!" -ForegroundColor Green
    Write-Host "可以在 https://gitee.com/PurpleWWW/Soilnet 查看你的代码" -ForegroundColor Cyan
} else {
    Write-Host "已取消推送" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "完成!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

