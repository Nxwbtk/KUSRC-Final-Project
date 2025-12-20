import socket
import time
from datetime import datetime

host = '192.168.4.1'
host_port = 4242

sleep_list = [2]
pose_list = ['forward', 'forward', 'forward', 'forward', 'stop']
# pose_list = ['forward', 'backward', 'left', 'right', 'stop']

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(datetime.now().strftime("%H:%M:%S"), 'Start Testing!')
    while (1):
        for sleep_duration in sleep_list:
            for pose in pose_list:
                # print(datetime.now().strftime("%H:%M:%S"), 'test:', pose, 'for', sleep_duration, 'sec')
                message = pose.encode()
                sock.sendto(message, (host, host_port))
                sock.settimeout(4)

                try:
                    data, server = sock.recvfrom(1024)
                    # print('return:', data.decode())
                except socket.timeout:
                    print(datetime.now().strftime("%H:%M:%S"), 'Warning: No response received (timeout)')
                    exit()

                time.sleep(sleep_duration if pose != 'stop' else 1)

    print('test passed!')

except Exception as e:
    print('test failed!', repr(e))

finally:
    if 'sock' in locals():
        sock.close()
