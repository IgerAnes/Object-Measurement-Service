import pickle
import socket
import cv2
import threading
from datetime import datetime

# setup server parameter
HOST = "0.0.0.0"
PORT = 6000
HEADERSIZE = 64
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

def systemTime():
    now = datetime.now()
    timeFormat = "%Y-%m-%d %H:%M:%S"
    formatTime = datetime.strftime(now, timeFormat)
    return formatTime

# setup socket server with parameter
socketServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    socketServer.bind((HOST, PORT))
except socket.error as e:
    print(f"[{systemTime()}][ERROR] {str(e)}")

def handleClient(conn, addr):
    print(f"[{systemTime()}][NEW CONNECTION] New user connects from {addr[0]}:{addr[1]}")
    connectedFlag = True
    imageCounter = 0
    while connectedFlag:
        data = b""
        data_len = conn.recv(HEADERSIZE).decode(FORMAT)
        # print(f"[{systemTime()}][SERVER] Receive data length")
        
        if data_len != None and data_len != "":
            data_len = int(data_len)
            remain_len = data_len
            print(f"[{systemTime()}][{addr[0]}:{addr[1]}] data length: {data_len}")
            # check remain data length, and avoid to get data from the other frame
            while len(data) < data_len:
                if remain_len >= 4096:
                    data += conn.recv(4096)
                    remain_len -= 4096
                else:
                    data += conn.recv(remain_len)
            
            # convert byte data to python object by pickle
            origin_data = pickle.loads(data[:data_len])
            if isinstance(origin_data, str):
                if origin_data == DISCONNECT_MESSAGE:
                    connectedFlag = False
                    print(f"[{systemTime()}][SERVER] Disconnect from client {addr[0]}:{addr[1]}")
            else:
                imageCounter += 1
                cv2.putText(origin_data, f"Counter: {imageCounter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 100, 50), 1 , cv2.LINE_AA)
                cv2.imshow("Receive", origin_data)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"[{systemTime()}][SERVER] Start to close client {addr[0]}:{addr[1]} connection")
    conn.close()
    print(f"[{systemTime()}][SERVER] Client {addr[0]}:{addr[1]} connection closed")
    cv2.destroyAllWindows()
    print(f"[{systemTime()}][SERVER] Destroy all cv2 windows")


def serverStart():
    socketServer.listen()
    print(f"[{systemTime()}][SERVER] Server Address: {HOST}:{PORT}")
    print(f"[{systemTime()}][SERVER] Start to listen to client connection ...")
    while True:
        # accept the connectio form outside
        print("Start to accept connection")
        conn, addr = socketServer.accept() # this cmd. will wait until receive next connection
        thread = threading.Thread(target=handleClient, args=(conn, addr))
        thread.start()
        print(f"[{systemTime()}][ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

print(f"[{systemTime()}][SERVER] Server is starting ...")
serverStart()