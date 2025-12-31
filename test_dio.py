#!/usr/bin/env python3
"""
Test script for Digital I/O functions
"""
import sys
sys.path.insert(0, '/home/demo/Source/ocr_datecode/ai_services')

from camera_shm_producer import read_di_value, write_do_value, check_trigger_edge

def test_read_all_di():
    """Test reading all DI ports"""
    print("\n" + "="*60)
    print("Testing: Read all Digital Inputs (DI 0-3)")
    print("="*60)

    for di in range(4):
        value = read_di_value(di)
        print(f"DI {di}: {value}")

def test_write_do():
    """Test writing to DO ports"""
    print("\n" + "="*60)
    print("Testing: Write to Digital Outputs (DO 0-3)")
    print("="*60)

    # Test setting DO 0
    print("\n--- Setting DO 0 = 1 ---")
    result = write_do_value(1, 1)
    print(f"Result: {'Success' if result else 'Failed'}")

    print("\n--- Setting DO 0 = 0 ---")
    result = write_do_value(0, 0)
    print(f"Result: {'Success' if result else 'Failed'}")

def test_trigger_edge():
    """Test edge detection logic"""
    print("\n" + "="*60)
    print("Testing: Trigger Edge Detection")
    print("="*60)

    test_cases = [
        # (previous, current, activation, expected)
        (0, 1, "RisingEdge", True),
        (1, 0, "RisingEdge", False),
        (0, 0, "RisingEdge", False),

        (1, 0, "FallingEdge", True),
        (0, 1, "FallingEdge", False),
        (1, 1, "FallingEdge", False),

        (0, 1, "AnyEdge", True),
        (1, 0, "AnyEdge", True),
        (0, 0, "AnyEdge", False),
        (1, 1, "AnyEdge", False),
    ]

    for prev, curr, activation, expected in test_cases:
        result = check_trigger_edge(curr, prev, activation)
        status = "✅" if result == expected else "❌"
        print(f"{status} {prev}→{curr} | {activation:12s} | Expected: {expected}, Got: {result}")

def test_hardware_trigger_simulation():
    """Simulate hardware trigger monitoring"""
    print("\n" + "="*60)
    print("Testing: Hardware Trigger Simulation (DI 0, RisingEdge)")
    print("="*60)
    print("Reading DI 0 continuously. Toggle DI to test trigger detection.")
    print("Press Ctrl+C to stop.\n")

    di_number = 0
    activation = "RisingEdge"
    previous_value = read_di_value(di_number)
    print(f"Initial DI {di_number} value: {previous_value}")

    try:
        import time
        trigger_count = 0

        for i in range(20):  # Run for 20 iterations
            time.sleep(0.5)  # Check every 500ms
            current_value = read_di_value(di_number)

            if check_trigger_edge(current_value, previous_value, activation):
                trigger_count += 1
                print(f"⚡ TRIGGER #{trigger_count} detected! DI {di_number}: {previous_value}→{current_value}")
            else:
                print(f"   DI {di_number} = {current_value}")

            previous_value = current_value

        print(f"\nTotal triggers detected: {trigger_count}")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")

if __name__ == "__main__":
    print("Digital I/O Test Suite")
    print("="*60)

    while True:
        print("\n\nSelect test:")
        print("1. Read all DI values (DI 0-3)")
        print("2. Write DO values (toggle DO 0)")
        print("3. Test edge detection logic")
        print("4. Hardware trigger simulation (monitor DI 0)")
        print("5. Run all tests")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            test_read_all_di()
        elif choice == "2":
            test_write_do()
        elif choice == "3":
            test_trigger_edge()
        elif choice == "4":
            test_hardware_trigger_simulation()
        elif choice == "5":
            test_read_all_di()
            test_write_do()
            test_trigger_edge()
        elif choice == "0":
            print("\nExiting...")
            break
        else:
            print("Invalid choice!")
