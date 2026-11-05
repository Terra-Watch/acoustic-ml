"""Wait for the board's USB serial port, then print the BENCH lines. Run: ~/.platformio/penv/bin/python read_results.py"""
import glob, time
import serial

end = time.time() + 240; ser = None
while time.time() < end and ser is None:
    for p in glob.glob("/dev/cu.usbmodem*"):
        try: ser = serial.Serial(); ser.port, ser.baudrate, ser.timeout, ser.dtr, ser.rts = p, 115200, 1, False, False; ser.open(); print("open", p); break
        except Exception: ser = None
    time.sleep(0.5)
assert ser, "no serial port: is the board plugged in?"
buf = b""
while time.time() < end and b"BENCH_DONE" not in buf:
    d = ser.read(256)
    if d: buf += d; print(d.decode(errors="replace"), end="", flush=True)
