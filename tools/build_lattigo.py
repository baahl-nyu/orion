import sys
import platform
import subprocess
from pathlib import Path

def build(setup_kwargs=None):
    """Build the Go shared library for Lattigo."""
    print("=== Building Go shared library ===")
    
    # Determine the output filename based on platform
    if platform.system() == "Windows":
        output_file = "lattigo-win.dll"
    elif platform.system() == "Darwin":  # macOS
        output_file = "lattigo-mac.dylib"
    else:  
        output_file = "lattigo-linux.so"
    
    # Set up paths
    root_dir = Path(__file__).parent.parent
    backend_dir = root_dir / "orion" / "backend" / "lattigo"
    output_path = backend_dir / output_file
    
    # Build command
    build_cmd = [
        "go", "build", 
        "-buildmode=c-shared",
        "-o", str(output_path),
        str(backend_dir)
    ]
    
    # Run the build command
    try:
        print(f"Running: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, cwd=str(backend_dir), check=True)
        print(f"Successfully built {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Go build failed with exit code {e.returncode}")
        sys.exit(1)
    
    # Return setup_kwargs for Poetry
    return setup_kwargs or {}


if __name__ == "__main__":
    build()
