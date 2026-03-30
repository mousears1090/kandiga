#!/bin/bash
cd "$(dirname "$0")"
pkill -f chat_server.py 2>/dev/null
lsof -ti:8899 | xargs kill -9 2>/dev/null
sleep 0.5
echo "Starting Kandiga Chat..."
/Volumes/Crucial/Users/mousears1090/projects/WebApp/bakan/.venv/bin/python chat_server.py &
sleep 3
open http://localhost:8899
wait
