import sys
import runpy
import os
import traceback

def set_limits():
    try:
        import resource
        # Limit CPU time to 5 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        # Limit memory to 256MB
        try:
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        except ValueError:
            pass
    except ImportError:
        # resource module is Unix-only; gracefully skip on Windows
        pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sandbox_wrapper.py <script.py>")
        sys.exit(1)

    script_path = sys.argv[1]

    # Primitive network blocking
    os.environ['http_proxy'] = 'http://127.0.0.1:1/'
    os.environ['https_proxy'] = 'http://127.0.0.1:1/'

    set_limits()

    try:
        # We need to clean up sys.argv so the script thinks it's running natively
        sys.argv = [script_path]
        runpy.run_path(script_path, run_name="__main__")
    except MemoryError:
        print("Sandbox Error: Memory Limit Exceeded", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
