#!/bin/bash
. /usr/lib/openfoam/openfoam2606/etc/bashrc || true

ROOT_DIR="/mnt/c/Users/soumi/tbnnFoam/verification_squareDuct"
CASE_DIR="${ROOT_DIR}/baseline_sst"

mkdir -p "${CASE_DIR}"
cp -r "${ROOT_DIR}/0" "${CASE_DIR}/"
cp -r "${ROOT_DIR}/constant" "${CASE_DIR}/"
cp -r "${ROOT_DIR}/system" "${CASE_DIR}/"

cd "${CASE_DIR}"
sed -i 's/application .*/application simpleFoam;/' system/controlDict
sed -i 's/endTime .*/endTime 600;/' system/controlDict
sed -i 's/writeInterval .*/writeInterval 100;/' system/controlDict

blockMesh > log.blockMesh 2>&1
simpleFoam > log.simpleFoam 2>&1

tail -n 25 log.simpleFoam
