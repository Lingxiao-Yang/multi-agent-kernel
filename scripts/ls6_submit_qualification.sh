#!/bin/bash
set -euo pipefail
umask 077
unset OPENAI_API_KEY
exec 9>submit.lock
flock -x 9
submit() {
    local name=$1
    local receipt="submit-$name.receipt" id
    if [[ ! -s "$receipt" ]]; then
        if [[ ! -s "$receipt.tmp" ]] || ! tail -n 1 "$receipt.tmp" | grep -Eq '^[0-9]+$'; then
            sbatch --parsable -A ASC26098 -J "qual-$name" project/scripts/ls6_qualification.sbatch "$name" > "$receipt.tmp" || return
        fi
        tail -n 1 "$receipt.tmp" | grep -Eq '^[0-9]+$' || return 1
        mv "$receipt.tmp" "$receipt" || return
    fi
    id=$(tail -n 1 "$receipt")
    [[ $id =~ ^[0-9]+$ ]] || return 1
    echo "$id"
}
for name in softmax layernorm; do
    id=$(submit "$name")
    printf '%s %s\n' "$name" "$id"
done
