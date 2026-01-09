@echo off
chcp 65001 >nul
REM 使用conda环境运行交互式SHAP分析工具
REM 解决 "No module named 'shap'" 问题

echo ============================================================
echo 交互式SHAP分析工具
echo ============================================================
echo.
echo 使用conda环境运行...
echo.

REM 切换到脚本目录
cd /d "%~dp0"

REM 使用conda环境运行Python脚本
D:\Anaconda\Scripts\conda.exe run -p "d:\Pycharm_Project\Pytorch\NSFC\Data-test-3\.conda" --no-capture-output python interactive_shap.py

echo.
echo ============================================================
echo 完成！
echo ============================================================
pause
