@echo off
set "VENV=G:\universal_dub\.venv"
set "PATH=%VENV%\Scripts;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CMAKE_ARGS=-DGGML_CUDA=on"
echo CMAKE_ARGS=%CMAKE_ARGS% > G:\universal_dub\build.log
echo CUDA=%CUDA_PATH% >> G:\universal_dub\build.log
uv pip install --python %VENV%\Scripts\python.exe --force-reinstall --no-deps --no-binary :all: --no-cache-dir llama-cpp-python >> G:\universal_dub\build.log 2>&1
echo UV_EXIT=%errorlevel% >> G:\universal_dub\build.log
