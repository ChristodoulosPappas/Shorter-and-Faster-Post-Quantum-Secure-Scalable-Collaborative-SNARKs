#!/bin/bash

echo "=================================================="
echo "    OS Network Tuning Verification Script"
echo "=================================================="

# 1. Check Core Buffer Limits
echo -e "\n--- 1. Core Socket Limits ---"
echo "EXPECTED: >= 1342177280"
sysctl net.core.rmem_max
sysctl net.core.wmem_max

# 2. Check TCP Auto-Tuning Limits
echo -e "\n--- 2. TCP IPv4 Buffer Limits ---"
echo "EXPECTED: Third number should be >= 1342177280"
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem

# 3. Check Congestion Control Algorithm
echo -e "\n--- 3. TCP Congestion Control ---"
echo "EXPECTED: bbr"
sysctl net.ipv4.tcp_congestion_control

# 4. Check Queuing Discipline (Required for BBR)
echo -e "\n--- 4. Network Queuing Discipline ---"
echo "EXPECTED: fq (BBR requires 'fq' to work properly)"
sysctl net.core.default_qdisc

# 5. Check Routing Table for initcwnd
echo -e "\n--- 5. Route Initial Congestion Window ---"
echo "EXPECTED: Should contain 'initcwnd 1000' and 'initrwnd 1000'"
ip route show default

echo -e "\n=================================================="
