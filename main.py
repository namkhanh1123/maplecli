"""
MapleCLI - A secure, feature-rich CLI for OpenAI-compatible APIs.
"""
import sys
from cli import CLI

def main() -> None:
    """Main function for the CLI."""
    try:
        cli = CLI()
        cli.run()
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install required libraries: pip install -e .")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()