if "%USER_NAME%" == "" (
	ECHO "usage: set USER_NAME=mikhailu connect.bat"

) else (
	putty %USER_NAME%@login.leonhard.ethz.ch
)
