#!/bin/bash

echo "Starting server"
python simple_server.py True &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in 5 7; do
    echo "Starting client $i"
    python simple_client.py $i True &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
