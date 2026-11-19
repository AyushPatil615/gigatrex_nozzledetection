@echo off
echo ========================================
echo  Installing required packages...
echo ========================================
pip install opencv-python numpy Pillow
echo.
echo ========================================
echo  Running augmentation...
echo ========================================
cd /d "%~dp0"
python data_pipeline/augment_dataset.py
pause
