rm -rf target/wheels
maturin build --release --features pytorch
uv publish target/wheels/*.whl
