@echo off

:: Set name of the main .cpp file for unity building
set Main=win32_polar

:: Begin CTime on main .cpp file
ctime -begin %Main%.ctm

:: Set build directory relative to current drive and path
set BuildDir=%~dp0..\build

:: Create build path if it doesn't exist
if not exist %BuildDir% mkdir %BuildDir%

:: Move to build directory
pushd %BuildDir%

:: Set compiler arguments
set Files=..\..\project\code\%Main%.cpp
set Libs=User32.lib Ole32.lib Avrt.lib Gdi32.lib Ws2_32.lib Winmm.lib
set ObjDir=.\obj\

:: Set Visual compiler flags:
:: -Zi enable debugging info
:: -FC use full path in diagnostics
:: -MT for multi-threading
:: -EHsc exception handling where "extern "C"" does not throw C++ exceptions
:: -W4 warning level (prefer -Wall but difficult when including Windows.h)
:: -Fo path to store Object files
:: -DWIN32 for Windows builds
:: -DDEBUG to toggle debug macros
:: -DDEBUG_CRT for C Runtime function debugging
:: -DDEBUG_WIN32 for Windows API function debugging
:: -DWIN32_METRICS for frame timing information printed to Visual Studio/Code debug console
:: -Fe to set .exe name
set CompilerFlags=-Zi -FC -MT -EHsc -W4 -Fo%ObjDir% -DWIN32=1 -DDEBUG -DDEBUG_CRT=1 -DDEBUG_WIN32=0 -DWIN32_METRICS=1 -DPOLAR_LOOP=1 -Fe"polar.exe"

:: Set debug path for logging files
set DebugDir=%~dp0..\build\debug

:: Create Object and Debug directories if they don't exist
if not exist %ObjDir% mkdir %ObjDir%
if not exist %DebugDir% mkdir %DebugDir%

:: Run Visual Studio compiler
cl %CompilerFlags% %Files% %Libs%

:: Jump out of build directory
popd

:: End CTime on main .cpp file
ctime -end %Main%.ctm