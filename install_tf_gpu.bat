@echo off
setlocal
chcp 65001 >nul

echo ============================================
echo TensorFlow GPU setup for Windows
echo Project: afm
echo ============================================
echo.
echo This script assumes:
echo 1. You are running on native Windows.
echo 2. The project venv is .\afm
echo 3. NVIDIA GPU is present.
echo.

if not exist ".\afm\Scripts\python.exe" (
    echo [ERROR] .\afm\Scripts\python.exe not found.
    echo Rebuild the venv first.
    exit /b 1
)

echo [1/3] Verify TensorFlow inside the afm venv
.\afm\Scripts\python.exe -c "import tensorflow as tf; print('TensorFlow', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"
echo.

echo [2/3] Required Windows native GPU runtime
echo TensorFlow 2.10.1 on Windows needs:
echo - CUDA 11.2
echo - cuDNN 8.1
echo.
echo Expected CUDA DLL examples:
echo - cudart64_110.dll
echo - cublas64_11.dll
echo - cudnn64_8.dll
echo.

echo [3/3] PATH entries that should exist after installation
echo C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
echo C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
echo.

echo If GPU is still not detected, run:
echo .\afm\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo.
pause
