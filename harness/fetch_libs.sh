#!/bin/sh
# Clone the three libraries at the exact commits measured into libs_src/.
set -e
cd "$(dirname "$0")/.."
mkdir -p libs_src && cd libs_src
get() { [ -d "$1" ] || git clone -q "$2" "$1"; git -C "$1" checkout -q "$3"; }
get python-json-pointer https://github.com/stefankoegl/python-json-pointer f2641f09b20dfde10f068e492531f4d914427110
get isodate https://github.com/gweis/isodate 17cb25eb7bc3556a68f3f7b241313e9bb8b23760
get parse https://github.com/r1chardj0n3s/parse 0d2d28fbccf023d0485be21aaccc95de88e92eaa
