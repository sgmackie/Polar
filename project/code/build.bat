echo Clang x64 (Compatibility)

@echo off

:: Set name of the Platform .cpp file for unity building
set Platform=win32_polar
set Engine=polar

:: Set CTime directory relative to current drive and path
set CTimeDir="%~dp0..\build\ctime"

:: Create CTime path if it doesn't exist
if not exist %CTimeDir% mkdir %CTimeDir%

:: Move to CTime directory
pushd %CTimeDir%

:: Begin CTime on .cpp files
ctime -begin %Engine%.ctm
ctime -begin %Platform%.ctm

:: Step out of CTime directory
popd

:: Set build directory relative to current drive and path
set BuildDir="%~dp0..\build\win32"

:: Create build path if it doesn't exist
if not exist %BuildDir% mkdir %BuildDir%

:: Move to build directory
pushd %BuildDir%

:: Set compiler arguments
set PlatformFiles="%~dp0%Platform%.cpp"
set EngineFiles="%~dp0%Engine%.cpp"
set ObjDir=.\obj\
set MapDir=.\map\

:: Create Object and Mapfile directories if they don't exist
if not exist %ObjDir% mkdir %ObjDir%
if not exist %MapDir% mkdir %MapDir%

:: Set Visual compiler flags:
:: -nologo          remove MSVC terminal banner
:: -Z7              generate debugging info to .pdb file
:: -FC              use full path in diagnostics
:: -MTd             for multi-threading (debug)
:: -Gm              to enable minimal rebuilds
:: -GR              enable runtime type information for insepcting objects
:: -EHa             to enable C++ exception handling
:: -Fo path         to store Object files
:: -DWIN32_METRICS  for frame timing information printed to Visual Studio/Code debug console
:: -DWASAPI_INFO    for WASAPI device and format info
set CompilerFlags=-nologo -Z7 -FC -MTd -GR -EHa -Fo%ObjDir% -DWIN32_METRICS=1 -DWASAPI_INFO=1

:: Set warning labels:
:: -Wall            warning level (-W4 for Windows due to include errors)
:: -WX              to treat compiler warnings as errors
:: -Wno             to disable Clang warnings:  unused-function
set CommonWarnings=-W4 -WX -Wno-unused-function

:: Set Compiler optimsation level for debug or release builds
:: -Oi              to generate intrinsic functions when applicable
:: -Od              to disable compiler optimisations               /   -O2                 to enable "fast code" (compiler optimisations)
:: -fp:precise      for default floating point operations           /   -fp:fast            for fastest floating point operations
set CompilerOpt=-Od -Oi -fp:precise

:: Set win32 libraries
:: User32.lib       Windows general library
:: Ole32.lib        WASAPI COM objects
:: Avrt.lib         WASAPI multithreading
:: Gdi32.lib        Windows graphics display library (Bitmaps, StrechDIBits)
:: Winmm.lib        Windows multimedia library to set sleep granularity
set Libs=User32.lib Ole32.lib Avrt.lib Gdi32.lib Winmm.lib

:: Set link flags:
:: /link            to enable linker options
:: -INCREMENTAL:NO  to disable incremental linking (not needed when building from scratch)
set LinkerFlags=/link -INCREMENTAL:NO

:: Set Linker optimsation level for debug or release builds
:: -OPT:REF         to eliminate functions that are never referenced /   -OPT:NOREF         to disable
:: -OPT:ICF         to enable COMDAT function folding                /   -OPT:NOICF         to disable
set LinkerOpt=-OPT:REF -OPT:ICF

:: Delete previous .pdb files to create unique ones for each build
del *.pdb > NUL 2> NUL

:: Run Visual Studio compiler
:: Polar:
:: -LD              to create DLL for export functions
:: -PDB:Filename    define name of .pdb file (with random used to generate unique ID)
:: -MAP:Filename    to store Mapfiles that list all elements in a given .exe or .dll file
:: -EXPORT          export "extern" functions
clang-cl %CompilerFlags% %CommonWarnings% %CompilerOpt% %EngineFiles% -LD %LinkerFlags% %LinkerOpt% -PDB:polar_%random%.pdb -MAP:%MapDir%%Engine%.map -EXPORT:Update -EXPORT:Render
set PolarLastError=%ERRORLEVEL%

:: Win32:
:: -SUBYSTEM        define subsystem for application (Window, Console)
clang-cl %CompilerFlags% %CommonWarnings% %CompilerOpt% %PlatformFiles% %LinkerFlags% %LinkerOpt% -MAP:%MapDir%%Platform%.map -SUBSYSTEM:windows %Libs%
set PlatformLastError=%ERRORLEVEL%

:: Jump out of build directory
popd

:: Move back to CTime path
pushd %CTimeDir%

:: End CTime on .cpp files
ctime -end %Engine%.ctm %PolarLastError%
ctime -end %Platform%.ctm %PlatformLastError%

:: Exit
popd