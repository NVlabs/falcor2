set NO_CMAKE_BUILD=1

echo Installing development python packages...
pip install -r %~dp0\..\requirements-dev.txt
if errorlevel 1 goto :fail

echo Installing slangpy (editable, no CMake build)...
pip install --editable %~dp0\..\external\slangpy
if errorlevel 1 goto :fail

echo Installing falcor2 (editable, no CMake build)...
pip install --editable %~dp0\..\
if errorlevel 1 goto :fail

echo Building extensions...
python %~dp0\build.py
if errorlevel 1 goto :fail

echo Installing slangpy_torch (if torch is available)...
python %~dp0\install_slangpy_torch.py
if errorlevel 1 goto :fail

set NO_CMAKE_BUILD=
exit /b 0

:fail
set NO_CMAKE_BUILD=
exit /b 1
