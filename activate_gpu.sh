source .venv/bin/activate

for cuda_lib_dir in "$VIRTUAL_ENV"/lib/python3.11/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

unset cuda_lib_dir
