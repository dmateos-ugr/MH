set -e

if [ ! -e "zig" ]; then
	echo -e "Downloading zig compiler\n"

	VERSION="0.11.0-dev.3106+3062f9b02"

	wget -P /tmp https://ziglang.org/builds/zig-linux-x86_64-${VERSION}.tar.xz
	tar -xf /tmp/zig-linux-x86_64-${VERSION}.tar.xz
	mv zig-linux-x86_64-${VERSION} zig
fi

echo "Building project"
./zig/zig build -p . -Doptimize=ReleaseSafe

echo "Done. You can run resulting binary with './bin/MH'."
