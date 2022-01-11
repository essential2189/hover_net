#!/bin/bash
PID=`ps -ef | grep "run_infer.py" | grep -v grep | awk '{print $2}'`
kill -9 $PID
