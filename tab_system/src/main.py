import sys
import traceback
import logging
import argparse
from pathlib import Path
from config import PROJECT_ROOT, LOG_FORMAT
from fusion.tab_generator import TabGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose=False, log_file=None):
    """
    Configure logging with console and optional file handlers.
    
    Args:
        verbose: If True, set level to DEBUG, otherwise WARNING
        log_file: Optional path to log file. If None and file logging is needed,
                  use default path PROJECT_ROOT/tab_synthesis.log
    """
    # Determine log level
    level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file is not None:
        # If log_file is True (flag without argument), use default path
        if log_file is True:
            log_file = PROJECT_ROOT / "tab_synthesis.log"
        else:
            log_file = Path(log_file)
        
        # Create parent directories if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def create_parser():
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(
        description="Guitar Tab Synthesis from Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --output ./results --format txt
  python main.py video.mp4 --verbose --log-file
  python main.py video.mp4 --quiet --log-file ./logs/debug.log
        """
    )
    
    parser.add_argument(
        "input",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for results (default: output/)"
    )
    
    parser.add_argument(
        "--format",
        choices=["txt", "pdf", "both"],
        default="both",
        help="Output format (default: both)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging (verbose output)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Explicitly set quiet mode (this is the default)"
    )
    
    parser.add_argument(
        "--log-file",
        nargs="?",
        const=True,
        metavar="PATH",
        help="Save logs to file (default path: PROJECT_ROOT/tab_synthesis.log)"
    )
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate verbose/quiet flags
    if args.verbose and args.quiet:
        print("Error: Cannot use --verbose and --quiet together", file=sys.stderr)
        return 1
    
    # Setup logging (verbose=True means DEBUG, verbose=False means WARNING)
    verbose = args.verbose
    setup_logging(verbose=verbose, log_file=args.log_file)
    
    try:
        logger.info("Starting tab generation...")
        logger.info(f"Input video: {args.input}")
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = PROJECT_ROOT / "output"
        
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Format: {args.format}")
        
        tg = TabGenerator(str(input_path), output_dir=output_dir, format=args.format)
        tabs_content = tg.generate()
        
        logger.info("Tab generation completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
