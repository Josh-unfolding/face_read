import serial
import time

class ArduinoInterface:

    def __init__(self, port="COM3", baudrate=9600):
        self.arduino = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        print("Arduino connected")

    def send_command(self, command):
        self.arduino.write((command + "\n").encode())
        print("Sent to Arduino:", command)

    def read_message(self):
        if self.arduino.in_waiting:
            message = self.arduino.readline().decode().strip()
            return message
        return None

    def buzzer_on(self):
        self.send_command("BUZZER_ON")

    def buzzer_off(self):
        self.send_command("BUZZER_OFF")

    def test_buzzer(self):
        self.send_command("BUZZER_TEST")

    def get_gps(self):
        self.send_command("GPS")

    def close(self):
        self.arduino.close()
