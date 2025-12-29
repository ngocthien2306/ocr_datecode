import Jetson.GPIO as GPIO

# Danh sách các chân đọc được (không bị busy)
readable_pins = {
    7: "MCLK05 (Audio Master Clock)",
    11: "UART1_RTS (UART #1 Request to Send)",
    12: "I2S2_CLK (Audio I2S #2 Clock)",
    19: "SPI1_MOSI (SPI #1 Master Out/Slave In)",
    21: "SPI1_MISO (SPI #1 Master In/Slave Out)",
    23: "SPI1_SCK (SPI #1 Shift Clock)",
    26: "SPI1_CS1_N (SPI #1 Chip Select #1)",
    32: "GPIO8",
    36: "UART1_CTS (UART #1 Clear to Send)"
}

def read_available_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
    print("\n" + "="*70)
    print("JETSON ORIN - GPIO STATUS (Các chân đọc được)")
    print("="*70)
    
    for pin, description in readable_pins.items():
        try:
            GPIO.setup(pin, GPIO.IN)
            value = GPIO.input(pin)
            status = "█ HIGH (1)" if value else "░ LOW  (0)"
            print(f"Pin {pin:2d} | {status} | {description}")
        except Exception as e:
            print(f"Pin {pin:2d} | ERROR    | {description}")
    
    print("="*70)
    GPIO.cleanup()

if __name__ == "__main__":
    read_available_gpio()