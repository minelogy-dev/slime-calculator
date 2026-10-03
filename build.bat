@echo off
REM Copyright 2026 minelogy
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM
REM Native Windows build. Usage: build.bat   (output: build\*.exe)
REM
REM This file is ASCII-only on purpose: cmd.exe parses .bat in the OEM codepage
REM (GBK/936 here), so UTF-8 Chinese text would be mis-parsed.

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---- Visual Studio environment (cl.exe / link.exe) -------------------------
REM vswhere ships with the Visual Studio Installer and its path is fixed.
set "VSPATH="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo [build] Visual Studio C++ toolset not found; install "Desktop development with C++".
    exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

REM ---- CUDA compatibility switches ------------------------------------------
REM A CUDA toolkit older than your MSVC makes the STL hard-error with
REM   yvals_core.h: error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer
REM and nvcc's cudafe++ then dies with 0xC0000005. These two switches are the
REM documented escape hatch; /utf-8 is required because the sources use UTF-8
REM Chinese comments. Remove them once your CUDA toolkit supports your MSVC.
set "CXXCOMPAT=/O2 /Oi /Ot /utf-8 /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
set "NVVCOMPAT=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

if not exist build mkdir build

echo === Building slime_cmp (C) ===
cl /O2 /Oi /Ot /Febuild\slime_cmp.exe src\slime_cmp.c /link /MACHINE:X64
if errorlevel 1 exit /b 1

if "%CUDA_ARCH%"=="" set CUDA_ARCH=-arch=all-major
echo === Building slime_main (CUDA) ===
nvcc -Xcompiler="%CXXCOMPAT%" -o build\slime_main.exe src\slime_main.cu -O3 -use_fast_math %NVVCOMPAT% %CUDA_ARCH% -std=c++20
if errorlevel 1 exit /b 1

echo === Building slime_circle (CUDA) ===
nvcc -Xcompiler="%CXXCOMPAT%" -o build\slime_circle.exe src\slime_circle.cu -O3 -use_fast_math %NVVCOMPAT% %CUDA_ARCH% -std=c++20
if errorlevel 1 exit /b 1

echo Done. Binaries in build\
endlocal
