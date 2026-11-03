@echo off
setlocal
call "%~dp0run_vision.cmd" -m vision.review_keypoints %*
if errorlevel 1 pause
