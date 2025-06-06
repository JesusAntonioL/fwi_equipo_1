from pymavlink import mavutil

# Replace with your actual serial port and baud rate (e.g., 57600 or 115200)
master = mavutil.mavlink_connection('COM5', baud=115200)  # Windows example
# master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)  # Linux example

master.wait_heartbeat()
print("Connected to system:", master.target_system)

while True:
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    if msg:
        lat = msg.lat / 1e7 #in degE7
        lon = msg.lon / 1e7 #in degE7
        alt = msg.alt #in mm
        print(f"Latitude: {lat}, Longitude: {lon}, Altura: {alt}")
