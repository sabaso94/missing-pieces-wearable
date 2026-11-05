#!/usr/bin/env bash
# Download the three data sources into data/ (each file is skipped if already present):
#   - PhysioNet "Minute-level step counts and physical activity data from NHANES 2011-2014" v1.0.2
#     (only the minute-level MIMS, wear prediction and quality flag files), checked against SHA256SUMS.txt
#   - NHANES 2011-12 (_G) and 2013-14 (_H) public-use XPT files for demographics, exam, labs and questionnaire
#   - NCHS public-use linked mortality files (2019 linkage)
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

fetch() {  # fetch URL DEST [extra curl args...]
    local url="$1" dest="$2"
    shift 2
    if [ -s "$dest" ]; then
        return
    fi
    mkdir -p "$(dirname "$dest")"
    curl -L --fail --silent --show-error "$@" -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
    echo "downloaded $dest"
}

# 1. PhysioNet minute-level activity
PN="https://physionet.org/files/minute-level-step-count-nhanes/1.0.2"
fetch "$PN/SHA256SUMS.txt" data/physionet/SHA256SUMS.txt
fetch "$PN/subject-info.csv" data/physionet/subject-info.csv
for var in log10PAXMTSM PAXPREDM PAXFLGSM; do
    fetch "$PN/csv/nhanes_1440_$var.csv.xz" "data/physionet/nhanes_1440_$var.csv.xz"
done
(
    cd data/physionet
    for f in subject-info.csv nhanes_1440_log10PAXMTSM.csv.xz nhanes_1440_PAXPREDM.csv.xz nhanes_1440_PAXFLGSM.csv.xz; do
        grep -E "[ *](csv/)?$f\$" SHA256SUMS.txt | awk -v f="$f" '{print $1"  "f}'
    done | sha256sum --check --quiet
)

# 2. NHANES public-use files (suffix G = 2011-12, H = 2013-14)
for cycle in "2011 G" "2013 H"; do
    set -- $cycle
    for name in DEMO BMX BPX MGX GHB TCHOL HDL BIOPRO CBC SMQ HUQ MCQ DIQ; do
        fetch "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/$1/DataFiles/${name}_$2.xpt" "data/nhanes/${name}_$2.xpt"
    done
done

# 3. Linked mortality. ftp.cdc.gov chains to a root (Sectigo Public Server Authentication Root R46) that older
#    system CA stores lack, so verify against the current Mozilla bundle shipped with certifi.
CA="$("$PYTHON" -c 'import certifi; print(certifi.where())')"
MORT="https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"
for years in 2011_2012 2013_2014; do
    fetch "$MORT/NHANES_${years}_MORT_2019_PUBLIC.dat" "data/mortality/NHANES_${years}_MORT_2019_PUBLIC.dat" --cacert "$CA"
done

echo "all data present in data/"
