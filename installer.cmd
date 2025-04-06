
cd ./python-3.13.2-embed-amd64
python get-pip.py --no-index --find-links=%str1%pip_whl
python -m pip install --upgrade pip
pause

set "str1=%cd%"
set str11=%str1:~0,-1%
set "str2=%cd%\Scripts"
echo %str11%
echo %str2%
pause

for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH`) do set my_user_path=%%B
echo %my_user_path%
setx PATH "%str11%;%str2%;%my_user_path%"

pause

