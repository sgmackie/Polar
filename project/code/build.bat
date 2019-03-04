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

:: Create Object and Mapfile directories if they don't exist
if not exist %ObjDir% mkdir %ObjDir%

:: Set compiler flags:
set CompilerFlags=-g -gcodeview -pedantic -std=c++14 -DCUDA=1

:: Set warning labels:
set CommonWarnings=-Wall -Wextra -Werror -Wno-unused-function -Wno-language-extension-token -Wno-vla-extension -Wno-deprecated-declarations -Wno-sign-compare

:: Set Compiler optimsation level
set CompilerOpt=-O0

:: Set win32 libraries
set Libs=-lUser32.lib -lOle32.lib -lAvrt.lib -lWinmm.lib

:: Run Clang compiler
clang %CompilerFlags% %CommonWarnings% %CompilerOpt% %Libs% %PlatformFiles% -o %Platform%.exe

:: Exit
popd
