#!/usr/bin/env python3
"""
File Descriptor Monitor Script

This script helps monitor file descriptor usage and identify potential leaks.
Run this alongside your gateway server to track FD usage over time.
"""

import os
import time
import psutil
import threading
from datetime import datetime

def get_fd_count(pid=None):
    """Get the number of open file descriptors for a process"""
    if pid is None:
        pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
        return process.num_fds()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def get_system_fd_info():
    """Get system-wide file descriptor information"""
    try:
        with open('/proc/sys/fs/file-nr', 'r') as f:
            data = f.read().strip().split()
            allocated = int(data[0])
            available = int(data[1])
            max_fds = int(data[2])
            return allocated, available, max_fds
    except:
        return None, None, None

def monitor_fds(pid=None, interval=5):
    """Monitor file descriptors for a process"""
    print(f"Monitoring file descriptors for PID {pid or os.getpid()}")
    print(f"Interval: {interval} seconds")
    print("-" * 60)
    
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fd_count = get_fd_count(pid)
            allocated, available, max_fds = get_system_fd_info()
            
            print(f"[{timestamp}] Process FDs: {fd_count}")
            if allocated is not None:
                print(f"[{timestamp}] System FDs: {allocated}/{max_fds} (available: {available})")
            
            # Check for potential issues
            if fd_count and fd_count > 1000:
                print(f"⚠️  WARNING: High FD count detected: {fd_count}")
            
            if allocated and allocated > max_fds * 0.8:
                print(f"⚠️  WARNING: System FD usage high: {allocated}/{max_fds}")
            
            print("-" * 60)
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error monitoring FDs: {e}")
            time.sleep(interval)

def list_open_files(pid=None):
    """List open files for a process"""
    if pid is None:
        pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
        open_files = process.open_files()
        
        print(f"\nOpen files for PID {pid}:")
        print("-" * 60)
        
        # Group by file type
        file_types = {}
        for file in open_files:
            if file.path:
                ext = os.path.splitext(file.path)[1] or 'no_extension'
                if ext not in file_types:
                    file_types[ext] = []
                file_types[ext].append(file.path)
        
        for ext, files in file_types.items():
            print(f"\n{ext} files ({len(files)}):")
            for file_path in files[:10]:  # Show first 10
                print(f"  {file_path}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
                
    except Exception as e:
        print(f"Error listing open files: {e}")

def list_connections(pid=None):
    """List network connections for a process"""
    if pid is None:
        pid = os.getpid()
    
    try:
        process = psutil.Process(pid)
        connections = process.connections()
        
        print(f"\nNetwork connections for PID {pid}:")
        print("-" * 60)
        
        for conn in connections:
            status = conn.status
            local_addr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A"
            remote_addr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A"
            print(f"  {conn.fd} | {conn.type} | {status} | {local_addr} -> {remote_addr}")
            
    except Exception as e:
        print(f"Error listing connections: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor file descriptors")
    parser.add_argument("--pid", type=int, help="PID to monitor (default: current process)")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--list-files", action="store_true", help="List open files")
    parser.add_argument("--list-connections", action="store_true", help="List network connections")
    
    args = parser.parse_args()
    
    if args.list_files:
        list_open_files(args.pid)
    elif args.list_connections:
        list_connections(args.pid)
    else:
        monitor_fds(args.pid, args.interval)
