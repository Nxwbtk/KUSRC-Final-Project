import socket
import time

host = '192.168.4.1'
host_port = 4242

sleep_list = [2, 2, 2, 2, 2, 2, 2, 2]
pose_list = ['forward', 'backward', 'left', 'right', 'stop']

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for sleep_duration in sleep_list:
        for pose in pose_list:
            print('test:', pose, 'for', sleep_duration, 'sec')
            message = pose.encode()
            sock.sendto(message, (host, host_port))
            sock.settimeout(2)

            try:
                data, server = sock.recvfrom(1024)
                print('return:', data.decode())
            except socket.timeout:
                print('Warning: No response received (timeout)')

            time.sleep(sleep_duration)

    print('test passed!')

except Exception as e:
    print('test failed!', repr(e))

finally:
    if 'sock' in locals():
        sock.close()
