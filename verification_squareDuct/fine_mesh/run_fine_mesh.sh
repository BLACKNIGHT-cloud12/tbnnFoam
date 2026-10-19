#!/bin/sh
cd "${0%/*}" || exit
set -e

rm -rf [1-9]* log.* constant/polyMesh

blockMesh > log.blockMesh 2>&1
tbnnFoam > log.tbnnFoam 2>&1
