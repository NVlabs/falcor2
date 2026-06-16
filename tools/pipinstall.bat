set NO_CMAKE_BUILD=1

echo Installing development python packages...
pip install -r %~dp0\..\requirements-dev.txt

echo Installing slangpy (editable, no CMake build)...
pip install --editable %~dp0\..\external\slangpy

echo Installing falcor2 (editable, no CMake build)...
pip install --editable %~dp0\..\

echo Building extensions...
python %~dp0\build.py

echo Installing slangpy_torch (if torch is available)...
python %~dp0\install_slangpy_torch.py

set NO_CMAKE_BUILD=
