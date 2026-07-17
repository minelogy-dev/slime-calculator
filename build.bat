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

setlocal enabledelayedexpansion

if not exist build mkdir build

echo === Building slime_cmp (C) ===
cl /O2 /Oi /Ot /Febuild\slime_cmp.exe src\slime_cmp.c /link /MACHINE:X64

if "%CUDA_ARCH%"=="" set CUDA_ARCH=-arch=all-major
echo === Building slime_main (CUDA) ===
nvcc -Xcompiler="/O2 /Oi /Ot" -o build\slime_main.exe src\slime_main.cu -O3 -use_fast_math %CUDA_ARCH% -std=c++20

echo === Building slime_circle (CUDA) ===
nvcc -Xcompiler="/O2 /Oi /Ot" -o build\slime_circle.exe src\slime_circle.cu -O3 -use_fast_math %CUDA_ARCH% -std=c++20

echo Done. Binaries in build\
endlocal
