@echo off

:: Set directories to read into
set BuildDir=..\build\win32
set ObjDir=.\obj\
set MapDir=.\map\
set LibExpDir=.\libexp\
set VisualStudio=.\.vs\

:: Delete files if build directory exists
if exist %BuildDir% (  
  :: Move to build path
  pushd %BuildDir%

  :: Force delete files with the following extentions, without printing output to the terminal
  del /q /f *.exe *.pdb *.ilk *.dll *.map *.lib *.exp *.pli
  
  :: Remove directory and sub-folders without printing output to the terminal
  if exist %ObjDir% (
    rd /q /s %ObjDir%
  )

  if exist %MapDir% (
    rd /q /s %MapDir%
  )

  if exist %LibExpDir% (
    rd /q /s %LibExpDir%
  )

  if exist %VisualStudio% (
    rd /q /s %VisualStudio%
  )

  popd
)