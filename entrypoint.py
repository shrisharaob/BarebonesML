import sys
import os


def run_script():
    if len(sys.argv) != 2:
        print("Usage: entrypoint.py <sub_folder/script_name.py>")
        sys.exit(1)

    script_name = sys.argv[1]

    try:
        os.execvp("python", ["python", f"{script_name}"])
    except FileNotFoundError:
        print(f"Error: Script '{script_name}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    run_script()

# docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix my-python-app
