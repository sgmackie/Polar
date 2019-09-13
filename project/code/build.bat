call build_cuda.bat

echo Clang x64

@echo off

:: Set name of the platform .cpp file for unity building
set Platform=win32_polar

:: Set build directory relative to current drive and path
set BuildDir="%~dp0..\build\win32"

:: Create build path if it doesn't exist
if not exist %BuildDir% mkdir %BuildDir%

:: Move to build directory
pushd %BuildDir%

:: Set compiler arguments
set PlatformFiles="%~dp0%Platform%.cpp"
set ObjDir=.\obj\

:: Create Object path if it doesn't exist
if not exist %ObjDir% mkdir %ObjDir%

:: Set CUDA include paths
set CUDAPaths=-I"%CUDA_PATH%\include" -L="%CUDA_PATH%\lib\x64"

:: Set compiler flags:
set CompilerFlags=-g -gcodeview -pedantic

:: Set warning labels:
set CommonWarnings=-Wall -Werror -Wno-language-extension-token -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function -Wno-writable-strings -Wno-gnu-anonymous-struct

:: Set Compiler optimsation level
REM set CompilerOpt=-O3 -march=native
set CompilerOpt=-O0

:: Set CUDA flags
set CUDAFlags=-DCUDA=1 -DPARTIALS_GPU=0 -DBUBBLES_GPU=0

:: Set logging flags
set LogFlags=-DLOGGER_ERROR=1 -DLOGGER_PROFILE=0

:: Set profile flags
set ProfileFlags=-DCORE_PROFILE=1

:: Set win32 libraries
set Libs=-lUser32.lib -lOle32.lib -lAvrt.lib -lWinmm.lib -ld3d9.lib -lGdi32.lib -lShell32.lib -lws2_32.lib -lAdvapi32.lib

:: Set path for CUDA function library
set CUDAFunctions=-lcudart.lib -lcurand.lib -lpolar_cuda.lib
REM set CUDAFunctions=

:: Run Clang compiler
clang %CompilerFlags% %CUDAPaths% %CommonWarnings% %CompilerOpt% %CUDAFlags% %LogFlags% %ProfileFlags% %Libs% %CUDAFunctions% %PlatformFiles% -o %Platform%.exe

:: Exit
popd