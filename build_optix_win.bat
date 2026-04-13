@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
set PATH=%APPDATA%\Python\Python314\Scripts;%PATH%
python cuda\v5\build_optix_ext.py
