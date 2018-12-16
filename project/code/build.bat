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
set ObjDir=.\obj\

:: Set link flags:
:: /link            to enable linker options
:: -INCREMENTAL:NO  to disable incremental linking (building everything from scratch everytime anyway)
:: -OPT:REF         to eleminate functions that are never referenced
:: User32.lib       Windows general library
:: Ole32.lib        WASAPI COM objects
:: Avrt.lib         WASAPI multithreading
:: Gdi32.lib        Windows graphics display library (Bitmaps, StrechDIBits)
:: Winmm.lib        Windows multimedia library to set sleep granularity
set Libs=/link -INCREMENTAL:NO -OPT:REF User32.lib Ole32.lib Avrt.lib Gdi32.lib Winmm.lib

:: Set Visual compiler flags:
:: -Z7              generate debugging info to .pdb file
:: -FC              use full path in diagnostics
:: -MTd             for multi-threading (debug)
:: -Od              to disable compiler optimisations
:: -Oi              to generate intrinsic functions when applicable
:: -Gm              to enable minimal rebuilds
:: -GR              enable runtime type information for insepcting objects
:: -EHa             to enable C++ exception handling
:: -WX              to treat compiler warnings as errors
:: -W4              warning level (prefer -Wall but difficult when including Windows.h)
:: -wd4201          to disable unnamed union/struct warning
:: -Fo path         to store Object files
:: -DWIN32          for Windows builds
:: -DDEBUG          to toggle debug macros
:: -DDEBUG_CRT      for C Runtime function debugging
:: -DDEBUG_WIN32    for Windows API function debugging
:: -DWIN32_METRICS  for frame timing information printed to Visual Studio/Code debug console
:: -DPOLAR_LOOP     for live code editing
:: -Fe              to set .exe name
set CompilerFlags=-Z7 -FC -MTd -Od -Oi -GR -EHa -WX -W4 -wd4201 -Fo%ObjDir% -DWIN32=1 -DDEBUG -DDEBUG_CRT=1 -DDEBUG_WIN32=0 -DWIN32_METRICS=1 -DPOLAR_LOOP=1 -Fe"polar.exe"

:: Set debug path for logging files
set DebugDir=%~dp0..\build\debug

:: Create Object and Debug directories if they don't exist
if not exist %ObjDir% mkdir %ObjDir%
if not exist %DebugDir% mkdir %DebugDir%

:: Delete previous .pdb files
del *.pdb > NUL 2> NUL

:: Run Visual Studio compiler
cl %CompilerFlags% %Files% %Libs%

:: Jump out of build directory
popd

:: End CTime on main .cpp file
ctime -end %Main%.ctm