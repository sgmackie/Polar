@echo off

:: Set directories to read into
set BuildDir=..\build
set ObjDir=.\obj\
set LibExpDir=.\libexp\
set Debug=.\Debug\
set VisualStudio=.\.vs\

:: Delete files if build directory exists
if exist %BuildDir% (  
  :: Move to build path
  pushd %BuildDir%

  :: Force delete files with the following extentions, without printing output to the terminal
  del /q /f *.exe *.pdb *.ilk *.dll *.map *.lib *.exp
  
  :: Remove directory and sub-folders without printing output to the terminal
  if exist %ObjDir% (
    rd /q /s %ObjDir%
  )

  if exist %LibExpDir% (
    rd /q /s %LibExpDir%
  )

  if exist %Debug% (
    rd /q /s %Debug%
  )

  if exist %VisualStudio% (
    rd /q /s %VisualStudio%
  )

  popd
)