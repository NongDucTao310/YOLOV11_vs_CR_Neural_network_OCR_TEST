@echo on
set "str1=%~dp0"
set str11=%str1:~0,-1%
set "str2=%~dp0Scripts"
echo %str11%
echo %str2%
pause

for /f "usebackq tokens=2,*" %%A in (`reg query HKCU\Environment /v PATH`) do set my_user_path=%%B
echo %my_user_path%
setx PATH "%str11%;%str2%;%my_user_path%"

pause