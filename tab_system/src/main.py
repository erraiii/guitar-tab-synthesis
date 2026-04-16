import sys
import traceback
from config import VIDEO_PATH
from fusion.tab_generator import TabGenerator


def main():
    try:
        print("[main] Starting tab generation...")
        print(f"[main] Video: {VIDEO_PATH}")
        
        tg = TabGenerator(VIDEO_PATH)
        tabs_content = tg.generate()
        
        print("[main] Tab generation completed successfully")
        return 0
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"[ERROR] Runtime error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
