ps -ef | grep '[r]ay' | grep -v grep | awk '{print $2}' | xargs kill -9
rm /tmp/ray/ray_current_cluster